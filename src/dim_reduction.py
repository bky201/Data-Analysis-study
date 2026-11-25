"""Dimensionality reduction: PCA, SVD, Random Projections, Autoencoder."""

import numpy as np
import copy
import torch
import torch.nn as nn
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection


class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, bottleneck_dim):
        super().__init__()
        enc_layers, prev = [], input_dim
        for h in hidden_dims:
            enc_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        enc_layers.append(nn.Linear(prev, bottleneck_dim))
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers, prev = [], bottleneck_dim
        for h in reversed(hidden_dims):
            dec_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x):
        return self.encoder(x)

def _fit_autoencoder(X_train: np.ndarray, config: dict):
    """Train AE on X_train, return (X_train_encoded, X_test_encoded_fn)."""
    ae_cfg = config["dim_reduction"]["autoencoder"]
    k = config["dim_reduction"]["n_components"]
    hidden = ae_cfg["hidden_dims"]
    max_epochs = ae_cfg["epochs"]
    lr = ae_cfg["lr"]
    batch_size = ae_cfg["batch_size"]

    patience = ae_cfg.get("patience", 10)
    min_delta = ae_cfg.get("min_delta", 1e-4)
    val_fraction = ae_cfg.get("val_fraction", 0.1)
    max_train_rows = ae_cfg.get("max_train_rows", 2000)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if X_train.shape[0] > max_train_rows:
        rng = np.random.default_rng(42)
        idx = rng.choice(X_train.shape[0], max_train_rows, replace=False)
        X_sub = X_train[idx]
    else:
        X_sub = X_train

    n_val = int(len(X_sub) * val_fraction)
    X_val = X_sub[:n_val]
    X_tr = X_sub[n_val:]

    X_tr_t = torch.FloatTensor(X_tr).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)

    model = Autoencoder(X_train.shape[1], hidden, k).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=patience // 2
    )
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    no_improve_count = 0
    actual_epochs = 0

    model.train()
    for epoch in range(max_epochs):
        perm = torch.randperm(X_tr_t.size(0))
        for start in range(0, X_tr_t.size(0), batch_size):
            idx = perm[start:start + batch_size]
            batch = X_tr_t[idx]
            optimizer.zero_grad()
            loss = loss_fn(model(batch), batch)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val_t), X_val_t).item()
        model.train()

        actual_epochs = epoch + 1
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        if (epoch + 1) % 10 == 0:
            print(f"    AE epoch {epoch+1}  lr={current_lr:.6f}  val_loss={val_loss:.6f}")

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve_count = 0
        else:
            no_improve_count += 1

        if (epoch + 1) % 20 == 0 or no_improve_count == patience:
            print(f"    AE epoch {epoch+1}/{max_epochs}  "
                  f"train_loss={loss.item():.6f}  val_loss={val_loss:.6f}  "
                  f"lr={optimizer.param_groups[0]['lr']:.6f}")

        if no_improve_count >= patience:
            print(f"    AE early stopping at epoch {actual_epochs} "
                  f"(best val_loss={best_val_loss:.6f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        X_train_enc = model.encode(torch.FloatTensor(X_train).to(device)).cpu().numpy()

    def _encode_test(X_test):
        Xt = torch.FloatTensor(X_test).to(device)
        with torch.no_grad():
            return model.encode(Xt).cpu().numpy()

    return X_train_enc, _encode_test, actual_epochs  # ← return actual epochs used


def reduce(X_train: np.ndarray, X_test: np.ndarray, config: dict):
    """
    Fit reducer on X_train, transform both.
    Returns (X_train_reduced, X_test_reduced, reducer_name).
    """
    method = config["dim_reduction"]["method"]  # set per-iteration by caller
    k = min(config["dim_reduction"]["n_components"],
            min(X_train.shape[0], X_train.shape[1]))

    if method == "autoencoder":
        X_train_red, _encode_test, actual_epochs = _fit_autoencoder(X_train, config)
        X_test_red = _encode_test(X_test)
        print(f"    [AE] Trained for {actual_epochs} epochs "
              f"(cap: {config['dim_reduction']['autoencoder']['epochs']})")

    elif method == "pca":
        reducer = PCA(n_components=k)
        X_train_red = reducer.fit_transform(X_train)
        X_test_red = reducer.transform(X_test)

    elif method == "svd":
        reducer = TruncatedSVD(n_components=k, random_state=42)
        X_train_red = reducer.fit_transform(X_train)
        X_test_red = reducer.transform(X_test)

    elif method == "random":
        reducer = GaussianRandomProjection(n_components=k, random_state=42)
        X_train_red = reducer.fit_transform(X_train)
        X_test_red = reducer.transform(X_test)

    else:
        raise ValueError(f"Unknown dim_reduction method: {method}")

    return X_train_red, X_test_red, method
