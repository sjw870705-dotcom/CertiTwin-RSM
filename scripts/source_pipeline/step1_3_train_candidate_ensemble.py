import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "commercial_twin"
    / "state_action_candidate_dataset"
)

MODEL_DIR = PROJECT_ROOT / "models" / "digital_twin_candidate_ensemble"
RESULT_DIR = PROJECT_ROOT / "results" / "step1_3"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_NAMES = ["V_embb", "V_urllc", "V_total", "management_utility"]

ENSEMBLE_SIZE = 3
BASE_SEED = 20260604

MAX_TRAIN_SAMPLES = 1_000_000
MAX_EVAL_SAMPLES = 300_000

BATCH_SIZE = 4096
EPOCHS = 10
LR = 1e-3
WEIGHT_DECAY = 1e-5


class CandidateDigitalTwinMLP(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 192),
            nn.ReLU(),
            nn.LayerNorm(192),
            nn.Dropout(dropout),

            nn.Linear(192, 192),
            nn.ReLU(),
            nn.LayerNorm(192),
            nn.Dropout(dropout),

            nn.Linear(192, 96),
            nn.ReLU(),
            nn.LayerNorm(96),
            nn.Dropout(dropout / 2),

            nn.Linear(96, 64),
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


def train_one_model(model_idx, X_train, Y_train, X_val, Y_val, input_dim, output_dim, device):
    seed = BASE_SEED + model_idx * 17
    set_seed(seed)

    X_train_s, Y_train_s = sample_arrays(X_train, Y_train, MAX_TRAIN_SAMPLES, seed)
    X_val_s, Y_val_s = sample_arrays(X_val, Y_val, MAX_EVAL_SAMPLES, seed + 1)

    model = CandidateDigitalTwinMLP(input_dim=input_dim, output_dim=output_dim, dropout=0.10).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.SmoothL1Loss()

    train_loader = make_loader(X_train_s, Y_train_s, BATCH_SIZE, shuffle=True)
    val_loader = make_loader(X_val_s, Y_val_s, BATCH_SIZE, shuffle=False)

    history = []
    best_val_loss = float("inf")
    best_path = MODEL_DIR / f"candidate_dt_ensemble_seed{model_idx}.pt"

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

        print(
            f"[Model {model_idx}] Epoch {epoch:02d} | "
            f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "input_dim": input_dim,
                "output_dim": output_dim,
                "target_names": TARGET_NAMES,
                "seed": seed,
                "model_idx": model_idx,
            }, best_path)

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    return model, {
        "model_idx": model_idx,
        "seed": seed,
        "best_val_loss": best_val_loss,
        "model_path": str(best_path),
        "history": history,
    }


def ensemble_predict(models, X, device):
    preds = []
    for model in models:
        preds.append(predict(model, X, device))
    stacked = np.stack(preds, axis=0)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    return mean, std, stacked


def regression_metrics(y_true, y_pred_mean, y_pred_std, target_names):
    metrics = {}
    eps = 1e-9

    for j, name in enumerate(target_names):
        yt = y_true[:, j]
        yp = y_pred_mean[:, j]
        ys = y_pred_std[:, j]

        err = yp - yt
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        bias = float(np.mean(err))

        ss_res = float(np.sum((yt - yp) ** 2))
        ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
        r2 = float(1.0 - ss_res / (ss_tot + eps))

        # Correlation between absolute error and ensemble std.
        abs_err = np.abs(err)
        if np.std(abs_err) > 1e-12 and np.std(ys) > 1e-12:
            uncertainty_error_corr = float(np.corrcoef(abs_err, ys)[0, 1])
        else:
            uncertainty_error_corr = 0.0

        metrics[name] = {
            "mae": mae,
            "rmse": rmse,
            "bias": bias,
            "r2": r2,
            "true_mean": float(np.mean(yt)),
            "pred_mean": float(np.mean(yp)),
            "true_std": float(np.std(yt)),
            "pred_std": float(np.std(yp)),
            "ensemble_sigma_mean": float(np.mean(ys)),
            "ensemble_sigma_p95": float(np.quantile(ys, 0.95)),
            "uncertainty_error_corr": uncertainty_error_corr,
        }

    return metrics


def save_ensemble_prediction_csv(path, y_true, pred_mean, pred_std, target_names, max_rows=300000):
    n = min(len(y_true), max_rows)
    data = {}
    for j, name in enumerate(target_names):
        data[f"true_{name}"] = y_true[:n, j]
        data[f"pred_mean_{name}"] = pred_mean[:n, j]
        data[f"pred_std_{name}"] = pred_std[:n, j]
        data[f"abs_err_{name}"] = np.abs(pred_mean[:n, j] - y_true[:n, j])
    pd.DataFrame(data).to_csv(path, index=False, encoding="utf-8-sig")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    X_train, Y_train = load_npy("train")
    X_val, Y_val = load_npy("validation")
    X_test, Y_test = load_npy("test")

    # Fixed eval sample for fair ensemble comparison.
    X_val_s, Y_val_s = sample_arrays(X_val, Y_val, MAX_EVAL_SAMPLES, BASE_SEED + 100)
    X_test_s, Y_test_s = sample_arrays(X_test, Y_test, MAX_EVAL_SAMPLES, BASE_SEED + 200)

    input_dim = X_train.shape[1]
    output_dim = Y_train.shape[1]

    print(f"Input dim: {input_dim}, output dim: {output_dim}")
    print(f"Eval validation shape: {X_val_s.shape}")
    print(f"Eval test shape: {X_test_s.shape}")

    models = []
    model_reports = []

    for model_idx in range(ENSEMBLE_SIZE):
        model, model_report = train_one_model(
            model_idx,
            X_train,
            Y_train,
            X_val,
            Y_val,
            input_dim,
            output_dim,
            device,
        )
        models.append(model)
        model_reports.append(model_report)

    pred_val_mean, pred_val_std, _ = ensemble_predict(models, X_val_s, device)
    pred_test_mean, pred_test_std, _ = ensemble_predict(models, X_test_s, device)

    val_metrics = regression_metrics(Y_val_s, pred_val_mean, pred_val_std, TARGET_NAMES)
    test_metrics = regression_metrics(Y_test_s, pred_test_mean, pred_test_std, TARGET_NAMES)

    pred_val_path = MODEL_DIR / "pred_validation_ensemble.csv"
    pred_test_path = MODEL_DIR / "pred_test_ensemble.csv"

    save_ensemble_prediction_csv(pred_val_path, Y_val_s, pred_val_mean, pred_val_std, TARGET_NAMES)
    save_ensemble_prediction_csv(pred_test_path, Y_test_s, pred_test_mean, pred_test_std, TARGET_NAMES)

    config = {
        "ensemble_size": ENSEMBLE_SIZE,
        "base_seed": BASE_SEED,
        "max_train_samples": MAX_TRAIN_SAMPLES,
        "max_eval_samples": MAX_EVAL_SAMPLES,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "device": device,
        "target_names": TARGET_NAMES,
        "note": "Candidate-aware ensemble digital twin for uncertainty estimation.",
    }

    summary = {
        "config": config,
        "model_reports": model_reports,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "outputs": {
            "pred_validation_ensemble": str(pred_val_path),
            "pred_test_ensemble": str(pred_test_path),
        }
    }

    with open(MODEL_DIR / "step1_3_ensemble_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(RESULT_DIR / "step1_3_report.md", "w", encoding="utf-8") as f:
        f.write("# Step 1.3 Candidate-Aware Ensemble Digital Twin Report\n\n")

        f.write("## Training setup\n\n")
        for k, v in config.items():
            f.write(f"- {k}: {v}\n")

        f.write("\n## Model reports\n\n")
        for m in model_reports:
            f.write(f"### Model {m['model_idx']}\n")
            f.write(f"- seed: {m['seed']}\n")
            f.write(f"- best_val_loss: {m['best_val_loss']}\n")
            f.write(f"- model_path: `{m['model_path']}`\n")

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
        f.write("- This ensemble provides both mean prediction and uncertainty sigma.\n")
        f.write("- The next step will compute calibrated risk upper bounds using calibration residuals.\n")
        f.write("- For the final paper, ensemble uncertainty should reduce false-safe actions compared with raw prediction.\n")

    print("Step 1.3 completed.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()