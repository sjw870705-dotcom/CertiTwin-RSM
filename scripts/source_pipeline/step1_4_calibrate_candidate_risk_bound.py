import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "commercial_twin"
    / "state_action_candidate_dataset"
)

MODEL_DIR = PROJECT_ROOT / "models" / "digital_twin_candidate_ensemble"

OUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "commercial_twin"
    / "candidate_calibration"
)

RESULT_DIR = PROJECT_ROOT / "results" / "step1_4"

OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_NAMES = ["V_embb", "V_urllc", "V_total", "management_utility"]
RISK_TARGETS = ["V_embb", "V_urllc", "V_total"]

MODEL_PATHS = [
    MODEL_DIR / "candidate_dt_ensemble_seed0.pt",
    MODEL_DIR / "candidate_dt_ensemble_seed1.pt",
    MODEL_DIR / "candidate_dt_ensemble_seed2.pt",
]

DELTA_LIST = [0.20, 0.10, 0.05]

SIGMA_FLOOR = 1e-4
PRED_BATCH_SIZE = 8192


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


def load_split(split):
    X = np.load(DATA_DIR / f"X_{split}.npy").astype(np.float32)
    Y = np.load(DATA_DIR / f"Y_{split}.npy").astype(np.float32)
    return X, Y


def load_models(device):
    models = []

    for p in MODEL_PATHS:
        if not p.exists():
            raise FileNotFoundError(f"Missing model file: {p}")

        ckpt = torch.load(p, map_location=device)
        model = CandidateDigitalTwinMLP(
            input_dim=ckpt["input_dim"],
            output_dim=ckpt["output_dim"],
            dropout=0.10,
        ).to(device)

        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        models.append(model)

    return models


@torch.no_grad()
def predict_one_model(model, X, device, batch_size=PRED_BATCH_SIZE):
    preds = []

    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).to(device)
        pred = model(xb).cpu().numpy()
        preds.append(pred)

    return np.vstack(preds)


def ensemble_predict(models, X, device):
    preds = []

    for model in models:
        preds.append(predict_one_model(model, X, device))

    stacked = np.stack(preds, axis=0)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)

    return mean, std


def make_prediction_df(Y, pred_mean, pred_std, split):
    data = {"split": [split] * len(Y)}

    for j, name in enumerate(TARGET_NAMES):
        data[f"true_{name}"] = Y[:, j]
        data[f"pred_mean_{name}"] = pred_mean[:, j]
        data[f"pred_std_{name}"] = pred_std[:, j]
        data[f"abs_err_{name}"] = np.abs(pred_mean[:, j] - Y[:, j])

    return pd.DataFrame(data)


def compute_quantiles(calib_df):
    q_table = {}

    for target in RISK_TARGETS:
        true = calib_df[f"true_{target}"].to_numpy()
        pred = calib_df[f"pred_mean_{target}"].to_numpy()
        sigma = calib_df[f"pred_std_{target}"].to_numpy()

        pos_residual = np.maximum(true - pred, 0.0)
        norm_residual = pos_residual / np.maximum(sigma, SIGMA_FLOOR)

        q_table[target] = {}

        for delta in DELTA_LIST:
            level = int((1.0 - delta) * 100)
            q = float(np.quantile(norm_residual, 1.0 - delta))
            q_table[target][f"q_{level}"] = q

    return q_table


def add_upper_bounds(df, q_table):
    out = df.copy()

    for target in RISK_TARGETS:
        pred = out[f"pred_mean_{target}"].to_numpy()
        sigma = out[f"pred_std_{target}"].to_numpy()

        for delta in DELTA_LIST:
            level = int((1.0 - delta) * 100)
            q = q_table[target][f"q_{level}"]
            out[f"upper_{target}_q{level}"] = pred + q * np.maximum(sigma, SIGMA_FLOOR)

        out[f"pos_residual_{target}"] = np.maximum(
            out[f"true_{target}"] - out[f"pred_mean_{target}"],
            0.0
        )

    return out


def evaluate_coverage(df, q_table, split_name):
    rows = []

    for target in RISK_TARGETS:
        true = df[f"true_{target}"].to_numpy()

        for delta in DELTA_LIST:
            level = int((1.0 - delta) * 100)
            upper = df[f"upper_{target}_q{level}"].to_numpy()

            coverage = float(np.mean(true <= upper))
            mean_upper_gap = float(np.mean(upper - true))
            p05_upper_gap = float(np.quantile(upper - true, 0.05))
            p95_upper_gap = float(np.quantile(upper - true, 0.95))

            # Sanity risk threshold for false-safe analysis.
            threshold = float(np.mean(true) + 0.5 * np.std(true))

            pred_safe = upper <= threshold
            true_safe = true <= threshold

            if pred_safe.sum() > 0:
                false_safe_rate = float((pred_safe & (~true_safe)).sum() / pred_safe.sum())
            else:
                false_safe_rate = 0.0

            unsafe = ~true_safe
            if unsafe.sum() > 0:
                unsafe_interception_rate = float(((~pred_safe) & unsafe).sum() / unsafe.sum())
            else:
                unsafe_interception_rate = 0.0

            safe_ratio = float(pred_safe.mean())

            rows.append({
                "split": split_name,
                "target": target,
                "delta": delta,
                "target_coverage": 1.0 - delta,
                "q_level": level,
                "q": q_table[target][f"q_{level}"],
                "empirical_coverage": coverage,
                "mean_upper_gap": mean_upper_gap,
                "p05_upper_gap": p05_upper_gap,
                "p95_upper_gap": p95_upper_gap,
                "risk_threshold_sanity": threshold,
                "safe_ratio_sanity": safe_ratio,
                "false_safe_rate_sanity": false_safe_rate,
                "unsafe_interception_rate_sanity": unsafe_interception_rate,
            })

    return pd.DataFrame(rows)


def save_capped_csv(df, path, max_rows=300000):
    if len(df) > max_rows:
        df.head(max_rows).to_csv(path, index=False, encoding="utf-8-sig")
    else:
        df.to_csv(path, index=False, encoding="utf-8-sig")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    models = load_models(device)

    prediction_dfs = {}
    upper_dfs = {}

    for split in ["calibration", "validation", "test"]:
        print(f"Predicting split: {split}")
        X, Y = load_split(split)

        pred_mean, pred_std = ensemble_predict(models, X, device)
        pred_df = make_prediction_df(Y, pred_mean, pred_std, split)
        prediction_dfs[split] = pred_df

    calib_df = prediction_dfs["calibration"]
    q_table = compute_quantiles(calib_df)

    coverage_tables = []

    outputs = {}

    for split in ["calibration", "validation", "test"]:
        upper_df = add_upper_bounds(prediction_dfs[split], q_table)
        upper_dfs[split] = upper_df

        coverage_tables.append(evaluate_coverage(upper_df, q_table, split))

        out_path = OUT_DIR / f"step1_4_{split}_predictions_with_upper.csv"
        save_capped_csv(upper_df, out_path, max_rows=300000)
        outputs[f"{split}_predictions_with_upper"] = str(out_path)

    coverage_df = pd.concat(coverage_tables, ignore_index=True)

    q_path = OUT_DIR / "step1_4_calibration_quantiles.json"
    coverage_path = OUT_DIR / "step1_4_coverage_table.csv"

    with open(q_path, "w", encoding="utf-8") as f:
        json.dump(q_table, f, indent=2)

    coverage_df.to_csv(coverage_path, index=False, encoding="utf-8-sig")

    outputs["quantiles"] = str(q_path)
    outputs["coverage_table"] = str(coverage_path)

    # Summary for quick inspection.
    summary = {
        "risk_targets": RISK_TARGETS,
        "target_names": TARGET_NAMES,
        "delta_list": DELTA_LIST,
        "sigma_floor": SIGMA_FLOOR,
        "quantiles": q_table,
        "outputs": outputs,
        "important_note": (
            "Step 1.4 uses ensemble mean and ensemble std to construct calibrated risk upper bounds. "
            "Quantiles are computed using the calibration split only."
        )
    }

    summary_path = OUT_DIR / "step1_4_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    report_path = RESULT_DIR / "step1_4_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Step 1.4 Candidate-Aware Calibrated Risk Bound Report\n\n")

        f.write("## Calibration quantiles\n\n")
        for target, qs in q_table.items():
            f.write(f"### {target}\n")
            for k, v in qs.items():
                f.write(f"- {k}: {v}\n")

        f.write("\n## Output files\n\n")
        for k, v in outputs.items():
            f.write(f"- {k}: `{v}`\n")

        f.write("\n## Important interpretation\n\n")
        f.write("- Ensemble mean is used as predicted risk.\n")
        f.write("- Ensemble std is used as uncertainty sigma.\n")
        f.write("- Calibration quantiles are estimated using calibration split only.\n")
        f.write("- The next step will construct the certified safe candidate set.\n")

    print("Step 1.4 completed.")
    print(json.dumps(summary, indent=2))
    print("\nCoverage table preview:")
    print(coverage_df.to_string(index=False))


if __name__ == "__main__":
    main()