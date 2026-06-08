import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "commercial_twin" / "dt_dataset"
MODEL_DIR = PROJECT_ROOT / "models" / "digital_twin_sanity"
RESULT_DIR = PROJECT_ROOT / "results" / "step0_6"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_NAMES = ["V_embb", "V_urllc", "V_total", "management_utility"]

# To keep the sanity training fast, we sample from the very large dataset.
MAX_TRAIN_SAMPLES = 800_000
MAX_EVAL_SAMPLES = 300_000

BATCH_SIZE = 4096
EPOCHS = 10
LR = 1e-3
WEIGHT_DECAY = 1e-5
SEED = 20260604


class DigitalTwinMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(0.05),

            nn.Linear(128, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(0.05),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_npy(split):
    X = np.load(DATA_DIR / f"X_{split}.npy")
    Y = np.load(DATA_DIR / f"Y_{split}.npy")
    return X.astype(np.float32), Y.astype(np.float32)


def sample_arrays(X, Y, max_samples, seed):
    if len(X) <= max_samples:
        return X, Y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=max_samples, replace=False)
    return X[idx], Y[idx]


def make_loader(X, Y, batch_size, shuffle):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


@torch.no_grad()
def predict(model, X, device, batch_size=8192):
    model.eval()
    preds = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).to(device)
        pred = model(xb).cpu().numpy()
        preds.append(pred)
    return np.vstack(preds)


def regression_metrics(y_true, y_pred, target_names):
    metrics = {}
    eps = 1e-9

    for j, name in enumerate(target_names):
        yt = y_true[:, j]
        yp = y_pred[:, j]

        err = yp - yt
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        bias = float(np.mean(err))

        ss_res = float(np.sum((yt - yp) ** 2))
        ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
        r2 = float(1.0 - ss_res / (ss_tot + eps))

        metrics[name] = {
            "mae": mae,
            "rmse": rmse,
            "bias": bias,
            "r2": r2,
            "true_mean": float(np.mean(yt)),
            "pred_mean": float(np.mean(yp)),
            "true_std": float(np.std(yt)),
            "pred_std": float(np.std(yp)),
        }

    return metrics


def save_prediction_csv(path, y_true, y_pred, target_names, max_rows=300000):
    n = min(len(y_true), max_rows)
    data = {}
    for j, name in enumerate(target_names):
        data[f"true_{name}"] = y_true[:n, j]
        data[f"pred_{name}"] = y_pred[:n, j]
        data[f"abs_err_{name}"] = np.abs(y_pred[:n, j] - y_true[:n, j])
    pd.DataFrame(data).to_csv(path, index=False, encoding="utf-8-sig")


def main():
    set_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    X_train, Y_train = load_npy("train")
    X_val, Y_val = load_npy("validation")
    X_test, Y_test = load_npy("test")

    X_train_s, Y_train_s = sample_arrays(X_train, Y_train, MAX_TRAIN_SAMPLES, SEED)
    X_val_s, Y_val_s = sample_arrays(X_val, Y_val, MAX_EVAL_SAMPLES, SEED + 1)
    X_test_s, Y_test_s = sample_arrays(X_test, Y_test, MAX_EVAL_SAMPLES, SEED + 2)

    input_dim = X_train_s.shape[1]
    output_dim = Y_train_s.shape[1]

    model = DigitalTwinMLP(input_dim=input_dim, output_dim=output_dim).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.SmoothL1Loss()

    train_loader = make_loader(X_train_s, Y_train_s, BATCH_SIZE, shuffle=True)
    val_loader = make_loader(X_val_s, Y_val_s, BATCH_SIZE, shuffle=False)

    history = []

    best_val_loss = float("inf")
    best_path = MODEL_DIR / "digital_twin_mlp.pt"

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                val_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        print(f"Epoch {epoch:02d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "input_dim": input_dim,
                "output_dim": output_dim,
                "target_names": TARGET_NAMES,
            }, best_path)

    # Load best model.
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    pred_val = predict(model, X_val_s, device)
    pred_test = predict(model, X_test_s, device)

    val_metrics = regression_metrics(Y_val_s, pred_val, TARGET_NAMES)
    test_metrics = regression_metrics(Y_test_s, pred_test, TARGET_NAMES)

    save_prediction_csv(MODEL_DIR / "pred_validation.csv", Y_val_s, pred_val, TARGET_NAMES)
    save_prediction_csv(MODEL_DIR / "pred_test.csv", Y_test_s, pred_test, TARGET_NAMES)

    config = {
        "seed": SEED,
        "max_train_samples": MAX_TRAIN_SAMPLES,
        "max_eval_samples": MAX_EVAL_SAMPLES,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "device": device,
        "note": "Sanity digital twin uses current KPI-derived features; final state-action model will be stricter.",
    }

    metrics = {
        "config": config,
        "history": history,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "outputs": {
            "model": str(best_path),
            "pred_validation": str(MODEL_DIR / "pred_validation.csv"),
            "pred_test": str(MODEL_DIR / "pred_test.csv"),
        }
    }

    with open(MODEL_DIR / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    with open(MODEL_DIR / "step0_6_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(RESULT_DIR / "step0_6_report.md", "w", encoding="utf-8") as f:
        f.write("# Step 0.6 Digital Twin Sanity Model Report\n\n")
        f.write("## Training setup\n\n")
        for k, v in config.items():
            f.write(f"- {k}: {v}\n")

        f.write("\n## Validation metrics\n\n")
        for name, m in val_metrics.items():
            f.write(f"### {name}\n")
            for k, v in m.items():
                f.write(f"- {k}: {v}\n")

        f.write("\n## Test metrics\n\n")
        for name, m in test_metrics.items():
            f.write(f"### {name}\n")
            for k, v in m.items():
                f.write(f"- {k}: {v}\n")

        f.write("\n## Important interpretation\n\n")
        f.write("- This is a sanity digital twin, not the final paper model.\n")
        f.write("- Current KPI-derived fields are included, so performance may be optimistic.\n")
        f.write("- The purpose is to verify training, prediction, residual, and calibration readiness.\n")

    print("Step 0.6 completed.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()