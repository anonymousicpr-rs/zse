# zayan.py - ZAYAN model with CL pretraining + Transformer classifier

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.utils.class_weight import compute_class_weight


# ------------------------------------------------------------------------------
# ZAYAN-CL: Feature-Level Contrastive Learning Module (with Dropout)
# ------------------------------------------------------------------------------

class ZAYAN_CL(nn.Module):
    def __init__(self,
                 emb_dim,
                 hidden_dim=256,
                 tau=0.1,
                 lambd=1.0,
                 sigma=0.1,
                 mask_prob=0.1,
                 dropout=0.1,
                 device='cuda'):
        super().__init__()
        self.device = device
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.tau = tau
        self.lambd = lambd
        self.sigma = sigma
        self.mask_prob = mask_prob
        self.dropout = dropout

        # encoder will be built later once we know num_samples
        self.encoder = None

    def build_encoder(self, num_samples):
        """
        num_samples: number of training samples (N)
        Encoder takes feature-vectors across samples: shape (m, N)
        """
        self.encoder = nn.Sequential(
            nn.Linear(num_samples, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.emb_dim)
        ).to(self.device)

    def augment(self, X):
        """
        X: (N, m) tensor of samples x features.
        Returns two augmented views X1, X2 with noise + masking.
        """
        # Gaussian noise
        X1 = X + torch.randn_like(X) * self.sigma

        # Feature masking
        mask = (torch.rand_like(X) > self.mask_prob).float()
        X2 = X * mask

        return X1, X2

    def forward(self, X):
        """
        X: (N, m)
        Returns: (contrastive + redundancy loss, normalized feature embeddings Z1n)
                 Z1n: (m, emb_dim)
        """
        X1, X2 = self.augment(X)            # (N, m) each
        F1 = X1.T.to(self.device)           # (m, N)
        F2 = X2.T.to(self.device)           # (m, N)

        Z1 = self.encoder(F1)               # (m, emb_dim)
        Z2 = self.encoder(F2)               # (m, emb_dim)

        Z1n = F.normalize(Z1, dim=1)        # (m, emb_dim)
        Z2n = F.normalize(Z2, dim=1)        # (m, emb_dim)

        m, _ = Z1n.shape

        # InfoNCE-style contrastive loss (features as "instances")
        sim = (Z1n @ Z2n.T) / self.tau      # (m, m)
        labels = torch.arange(m, device=self.device)
        loss_contrast = F.cross_entropy(sim, labels)

        # Redundancy reduction (decorrelation)
        G = Z1n @ Z1n.T                     # (m, m)
        I = torch.eye(m, device=self.device)
        loss_redundancy = torch.norm(G - I, p='fro') ** 2

        loss = loss_contrast + self.lambd * loss_redundancy
        return loss, Z1n

    def pretrain(self, X, epochs=100, lr=1e-3, weight_decay=1e-4):
        """
        Pretrains the encoder on X (N, m), returns feature embeddings Z (m, emb_dim).
        """
        # Prepare data
        if isinstance(X, pd.DataFrame):
            X = X.values.astype(np.float32)
        elif isinstance(X, np.ndarray):
            X = X.astype(np.float32)

        X = torch.tensor(X, dtype=torch.float32).to(self.device)

        # Build encoder with num_samples = N
        N, _ = X.shape
        self.build_encoder(N)

        # Optimizer + scheduler
        optimizer = optim.Adam(
            self.encoder.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )

        # Training loop
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss, _ = self.forward(X)
            loss.backward()
            optimizer.step()
            scheduler.step(loss)

            print(f"[ZAYAN-CL] Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

        # Final embeddings
        with torch.no_grad():
            _, Z = self.forward(X)

        return Z  # (m, emb_dim)


# ------------------------------------------------------------------------------
# ZAYAN-T: Transformer Classifier Module (with Dropout)
# ------------------------------------------------------------------------------

class ZAYAN_T(nn.Module):
    def __init__(self,
                 emb_dim,
                 num_features,
                 num_classes,
                 nhead=4,
                 num_layers=2,
                 gamma=1.0,
                 dropout=0.1,
                 classification_type='multiclass',
                 device='cuda'):
        super().__init__()
        self.device = device
        self.gamma = gamma
        self.classification_type = classification_type

        # Positional embeddings for features
        self.pos_embed = nn.Parameter(
            torch.zeros(num_features, emb_dim)
        )

        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=nhead,
            dropout=dropout,
            batch_first=False    # default PyTorch behavior (S, B, E)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        out_dim = 1 if classification_type == 'binary' else num_classes

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, out_dim)
        )

        # Move whole module to device
        self.to(self.device)

    def forward(self, X, Z):
        """
        X: (batch_size, m) raw features
        Z: (m, emb_dim) feature embeddings from CL
        Returns:
            logits: (batch_size, out_dim)
            preserve_loss: scalar MSE between transformer outputs and Z
        """
        bs, m = X.shape

        # Broadcast raw features over embedding dimension
        # X.unsqueeze(-1) : (bs, m, 1)
        # Z.unsqueeze(0)  : (1, m, emb_dim)
        H = X.unsqueeze(-1).to(self.device) * Z.unsqueeze(0)  # (bs, m, emb_dim)

        # Add positional encoding
        H = H + self.pos_embed.unsqueeze(0)  # (bs, m, emb_dim)
        H = self.dropout(H)

        # Transformer expects (S, B, E)
        H = H.permute(1, 0, 2)  # (m, bs, emb_dim)
        H_out = self.transformer(H)
        H_out = H_out.permute(1, 0, 2)  # (bs, m, emb_dim)

        # Global pooling over features
        rep = H_out.mean(dim=1)         # (bs, emb_dim)
        logits = self.mlp(rep)          # (bs, out_dim)

        # Preserve feature embedding structure
        Z_exp = Z.unsqueeze(0).expand(bs, -1, -1)  # (bs, m, emb_dim)
        preserve_loss = F.mse_loss(H_out, Z_exp)

        return logits, preserve_loss


# ------------------------------------------------------------------------------
# ZAYAN: Orchestrates CL + T + Evaluation
# ------------------------------------------------------------------------------

class ZAYAN:
    def __init__(self,
                 cl_params: dict,
                 t_params: dict,
                 classification_type='multiclass',
                 num_classes=None,
                 device='cuda'):
        self.device = device

        # Contrastive module
        self.zcl = ZAYAN_CL(device=device, **cl_params)

        # Transformer classifier
        self.zt = None

        # Learned feature embeddings from CL
        self.Z = None  # (m, emb_dim)

        self.classification_type = classification_type
        self.num_classes = num_classes
        self.t_params = t_params

    def fit(self,
            X_train, y_train,
            X_val=None, y_val=None,
            X_test=None, y_test=None,
            cl_epochs=50, cl_lr=1e-3, cl_weight_decay=1e-4,
            t_epochs=20, t_lr=1e-4, t_weight_decay=1e-4,
            batch_size=32):

        # ------------------------------------------------------------------
        # 1) Pretrain CL on training features
        # ------------------------------------------------------------------
        self.Z = self.zcl.pretrain(
            X_train,
            epochs=cl_epochs,
            lr=cl_lr,
            weight_decay=cl_weight_decay
        )  # (m, emb_dim)

        # ------------------------------------------------------------------
        # 2) Initialize Transformer
        # ------------------------------------------------------------------
        m, d = self.Z.shape
        if self.zt is None:
            params = {
                'emb_dim': d,
                'num_features': m,
                'num_classes': self.num_classes,
                'classification_type': self.classification_type,
                **self.t_params
            }
            self.zt = ZAYAN_T(device=self.device, **params)

        # ------------------------------------------------------------------
        # 3) Prepare training tensors
        # ------------------------------------------------------------------
        X_tr = X_train.values if isinstance(X_train, pd.DataFrame) else np.array(X_train)
        y_tr = y_train.values if hasattr(y_train, 'values') else np.array(y_train)

        Xt = torch.tensor(X_tr, dtype=torch.float32).to(self.device)

        if self.classification_type == 'multiclass':
            yt = torch.tensor(y_tr, dtype=torch.long).to(self.device)
        else:
            yt = torch.tensor(y_tr, dtype=torch.float32).to(self.device)

        loader = DataLoader(
            TensorDataset(Xt, yt),
            batch_size=batch_size,
            shuffle=True
        )

        # ------------------------------------------------------------------
        # 4) Class-balanced loss
        # ------------------------------------------------------------------
        if self.classification_type == 'multiclass':
            weights = compute_class_weight(
                class_weight='balanced',
                classes=np.unique(y_tr),
                y=y_tr
            )
            weight_tensor = torch.tensor(weights, dtype=torch.float32).to(self.device)
            loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            loss_fn = nn.BCEWithLogitsLoss()

        # ------------------------------------------------------------------
        # 5) Optimizer + scheduler for Transformer
        # ------------------------------------------------------------------
        optimizer = optim.Adam(
            self.zt.parameters(),
            lr=t_lr,
            weight_decay=t_weight_decay
        )
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )

        # ------------------------------------------------------------------
        # 6) Fine-tune Transformer
        # ------------------------------------------------------------------
        for epoch in range(t_epochs):
            self.zt.train()
            total_loss = 0.0

            for xb, yb in loader:
                optimizer.zero_grad()

                logits, pres = self.zt(xb, self.Z)

                if self.classification_type == 'binary':
                    yb = yb.unsqueeze(1)  # (batch, 1)

                loss = loss_fn(logits, yb) + self.zt.gamma * pres
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            scheduler.step(avg_loss)

            print(f"[ZAYAN-T] Epoch {epoch+1}/{t_epochs} | Avg Loss: {avg_loss:.4f}")

        # ------------------------------------------------------------------
        # 7) Evaluate (optional)
        # ------------------------------------------------------------------
        results = {}
        if X_val is not None and y_val is not None:
            results['validation'] = self.evaluate(X_val, y_val)
        if X_test is not None and y_test is not None:
            results['test'] = self.evaluate(X_test, y_test)

        return results

    def evaluate(self, X, y):
        """
        Evaluate model on given data.
        Returns a dict with metrics (accuracy, macro F1, etc.).
        """
        self.zt.eval()

        X_ev = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        y_ev = y.values if hasattr(y, 'values') else np.array(y)

        Xt = torch.tensor(X_ev, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits, _ = self.zt(Xt, self.Z)

            if self.classification_type == 'binary':
                probs = torch.sigmoid(logits).cpu().numpy().ravel()
                preds = (probs > 0.5).astype(int)

                return {
                    'accuracy':  accuracy_score(y_ev, preds),
                    'precision': precision_score(y_ev, preds, zero_division=0),
                    'recall':    recall_score(y_ev, preds, zero_division=0),
                    'f1':        f1_score(y_ev, preds, zero_division=0),
                }
            else:
                probs = F.softmax(logits, dim=1).cpu().numpy()
                preds = probs.argmax(axis=1)

                return {
                    'accuracy':        accuracy_score(y_ev, preds),
                    'macro_precision': precision_score(y_ev, preds, average='macro', zero_division=0),
                    'macro_recall':    recall_score(y_ev, preds, average='macro', zero_division=0),
                    'macro_f1':        f1_score(y_ev, preds, average='macro', zero_division=0),
                }