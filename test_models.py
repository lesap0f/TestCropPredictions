import os
from pathlib import Path
import pandas as pd
import geopandas as gpd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, accuracy_score
from baselines import Baselines
import numpy as np
from scipy import stats
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate crop yield predictions against ground truth.")
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("ground_truth/pub_yipeeo_yield_fl.geojson"),
        help="Path to the ground truth GeoJSON file.",
    )
    parser.add_argument(
        "--preds-dir",
        type=Path,
        default=Path("preds"),
        help="Directory containing prediction CSV files.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory to write output CSVs.",
    )
    return parser.parse_args()

args = parse_args()
GROUND_TRUTH_FILE_NAME = args.ground_truth
PREDS_DIR = args.preds_dir
RESULTS_DIR = args.results_dir


CROP_ALIAS = {
    "barley": "barley", "canola": "canola", "corn_bu": "corn_bu",
    "corn_tons": "corn_tons", "cotton": "cotton", "hay": "hay",
    "oats": "oats", "peanuts": "peanuts", "rice": "rice",
    "sorghum": "sorghum", "soybeans": "soybeans", "wheat": "wheat",
    "common winter wheat": "wheat", "common spring wheat": "wheat", "durum wheat": "wheat",
    "spring barley": "barley", "winter barley": "barley",
    "grain maize and corn-cob-mix": "corn_bu", "grain maize": "corn_bu",
    "grain corn": "corn_bu", "maize": "corn_bu", "corn": "corn_bu",
    "green maize": "corn_tons", "soya": "soybeans",
    "field peas": "field peas", "onions": "onions",
    "other oil seed crops n.e.c.": "other oil seed crops n.e.c.",
    "potatoes": "potatoes", "sugar beet": "sugar beet",
    "winter rape and turnip rape seeds": "winter rape and turnip rape seeds",
}

CROP_NAMES = [
    "corn_bu", "corn_tons", "cotton",
    "oats", "peanuts", "rice", "soybeans", "wheat",
]

THRESHOLD = 0.55
TILE_AREA = 23.04  # ha


BASELINE_NAMES = [
    "dominant_class_normalised_average_baseline",
    "right_class_field_adjusted_normalised_class_average",
    "random_class_normalised_class_average",
]


def get_preds_from_csv():
    preds = {}
    directory = Path(os.getcwd()) / PREDS_DIR
    for csv_path in directory.glob("*.csv"):
        df = pd.read_csv(csv_path)
        df["field_id"] = df["field_id"].astype(str)
        preds[csv_path.stem] = df
    return preds


def evaluate(y_true, y_pred, gt_labels, pred_labels):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    residuals = np.abs(y_pred - y_true)
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "class_acc": accuracy_score(gt_labels, pred_labels),
        "n": len(y_true),
        "residuals": residuals,
    }


def mean_improvement(x, y, axis=0):
    return np.mean(x - y, axis=axis)# Check if this is the absolute difference or just greater
                                    # The difference is absolute, look above at how the residuals are calculated
                                    # The residuals are absolute differences so I just look at what absolute difference is larger
                                    # This however does not tells me if I am biased in any direction


def compute_baseline_predictions(baseline_name, gt_indexed, baseline_obj):
    """Field-level baseline: one row per field."""
    rows = []
    for fid, gt_row in gt_indexed.iterrows():
        gt_prod = gt_row["production"]
        area_ha = min(gt_row["area_ha"], TILE_AREA)
        gt_crop = gt_row["crop_alias"]

        if baseline_name == "dominant_class_normalised_average_baseline":
            pred_label, pred_prod = baseline_obj.dominant_class_normalised_average_baseline()
        elif baseline_name == "dominant_class_field_adjusted_normalised_average_baseline":
            pred_label, pred_prod = baseline_obj.dominant_class_field_adjusted_normalised_average_baseline(area_ha)
        elif baseline_name == "right_class_normalised_class_average":
            pred_label, pred_prod = baseline_obj.right_class_normalised_class_average(gt_crop)
        elif baseline_name == "right_class_field_adjusted_normalised_class_average":
            pred_label, pred_prod = baseline_obj.right_class_field_adjusted_normalised_class_average(gt_crop, area_ha)
        elif baseline_name == "random_class_normalised_dataset_average":
            pred_label, pred_prod = baseline_obj.random_class_normalised_dataset_average()
        elif baseline_name == "random_class_normalised_class_average":
            pred_label, pred_prod = baseline_obj.random_class_normalised_class_average()
        elif baseline_name == "random_class_field_adjusted_normalised_class_average":
            pred_label, pred_prod = baseline_obj.random_class_field_adjusted_normalised_class_average(area_ha)
        else:
            raise ValueError(f"Unknown baseline: {baseline_name}")

        rows.append({
            "field_id": fid,
            "y_true": gt_prod,
            "y_pred": pred_prod,
            "gt_label": gt_crop,
            "pred_label": pred_label,
            "abs_res": abs(pred_prod - gt_prod),
        })

    return pd.DataFrame(rows)


def compute_baseline_predictions_fm(baseline_name, gt_indexed, baseline_obj, field_month_index):
    """
    field_month_index: DataFrame with columns ['field_id', 'month']
    representing the exact (field_id, month) pairs observed in the model predictions.
    One baseline row is produced per pair — same multiplicity as the model.
    """
    # First compute field-level predictions
    field_preds = {}
    for fid, gt_row in gt_indexed.iterrows():
        gt_prod = gt_row["production"]
        area_ha = min(gt_row["area_ha"], TILE_AREA)
        gt_crop = gt_row["crop_alias"]

        if baseline_name == "dominant_class_normalised_average_baseline":
            pred_label, pred_prod = baseline_obj.dominant_class_normalised_average_baseline()
        elif baseline_name == "dominant_class_field_adjusted_normalised_average_baseline":
            pred_label, pred_prod = baseline_obj.dominant_class_field_adjusted_normalised_average_baseline(area_ha)
        elif baseline_name == "right_class_normalised_class_average":
            pred_label, pred_prod = baseline_obj.right_class_normalised_class_average(gt_crop)
        elif baseline_name == "right_class_field_adjusted_normalised_class_average":
            pred_label, pred_prod = baseline_obj.right_class_field_adjusted_normalised_class_average(gt_crop, area_ha)
        elif baseline_name == "random_class_normalised_dataset_average":
            pred_label, pred_prod = baseline_obj.random_class_normalised_dataset_average()
        elif baseline_name == "random_class_normalised_class_average":
            pred_label, pred_prod = baseline_obj.random_class_normalised_class_average()
        elif baseline_name == "random_class_field_adjusted_normalised_class_average":
            pred_label, pred_prod = baseline_obj.random_class_field_adjusted_normalised_class_average(area_ha)
        else:
            raise ValueError(f"Unknown baseline: {baseline_name}")

        field_preds[fid] = {
            "y_true": gt_prod,
            "y_pred": pred_prod,
            "gt_label": gt_crop,
            "pred_label": pred_label,
        }

    # Broadcast to exact (field_id, month) pairs from the model
    rows = []
    for _, row in field_month_index.iterrows():
        fid = row["field_id"]
        month = row["month"]
        if fid not in field_preds:
            continue
        fp = field_preds[fid]
        rows.append({
            "field_id": fid,
            "month": month,
            "y_true": fp["y_true"],
            "y_pred": fp["y_pred"],
            "gt_label": fp["gt_label"],
            "pred_label": fp["pred_label"],
            "abs_res": abs(fp["y_pred"] - fp["y_true"]),
        })

    return pd.DataFrame(rows)


# ── Load GT ──────────────────────────────────────────────────────────────────
ground_truth = gpd.read_file(GROUND_TRUTH_FILE_NAME)
# opens the file second time to extract the "id"
with open(GROUND_TRUTH_FILE_NAME, "r", encoding="utf-8") as f:
    raw = json.load(f)

ground_truth["field_id"] = [feat["id"] for feat in raw["features"]]
ground_truth["field_id"] = ground_truth["field_id"].astype(int)
gt_indexed = ground_truth.set_index("field_id")

ground_truth["crop_alias"] = ground_truth["crop_type"].replace(CROP_ALIAS)

# Filter ground truth to insample crops with existing yield values
ground_truth = ground_truth[
    ground_truth["crop_alias"].isin(CROP_NAMES) &
    ground_truth["yield"].notna()
].copy()

ground_truth_proj = ground_truth.to_crs("EPSG:6933")
ground_truth["area_ha"] = (ground_truth_proj.geometry.area / 10_000).clip(upper=TILE_AREA)
ground_truth["production"] = ground_truth["yield"] * ground_truth["area_ha"]

gt_indexed = ground_truth.set_index("field_id")
all_preds_usda = get_preds_from_csv()

# ── Model evaluation + store per-field-month rows ────────────────────────────
per_field_month_rows = []
perm_test_rows = []
all_results = []

for model_name, df in all_preds_usda.items():

    df["field_id"] = df["field_id"].astype(int)
    shared_ids = gt_indexed.index.intersection(df["field_id"].astype("int32").unique())
    print(f"{model_name}: {len(shared_ids)} / {len(gt_indexed)} fields matched")

    months = sorted(df["month"].unique())
    monthly_buffers = {
        m: {"y_true": [], "y_pred": [], "gt_labels": [], "pred_labels": []}
        for m in months
    }
    month_overall = {
        "y_true": [], "y_pred": [], "gt_labels": [], "pred_labels": [],
        "field_id": [], "month": [], "abs_res": []
    }

    for fid in shared_ids:
        gt_row = gt_indexed.loc[fid]
        gt_crop = gt_row["crop_alias"]
        gt_prod = gt_row["production"]

        field_df = df[df["field_id"].astype("int32") == fid]

        for month, month_df in field_df.groupby("month"):
            month_df_sorted = month_df.sort_values("pred_crop_prob", ascending=False)

            gt_match = month_df_sorted[
                (month_df_sorted["pred_crop_type"] == gt_crop) &
                (month_df_sorted["pred_crop_prob"] >= THRESHOLD)
            ]

            if not gt_match.empty:
                pred_prod = gt_match.iloc[0]["predicted_production"]
                pred_label = gt_match.iloc[0]["pred_crop_type"]
            else:
                top1 = month_df_sorted.iloc[0]
                pred_prod = top1["predicted_production"]
                pred_label = "other"


            monthly_buffers[month]["y_true"].append(gt_prod)
            monthly_buffers[month]["y_pred"].append(pred_prod)
            monthly_buffers[month]["gt_labels"].append(gt_crop)
            monthly_buffers[month]["pred_labels"].append(pred_label)

            month_overall["y_true"].append(gt_prod)
            month_overall["y_pred"].append(pred_prod)
            month_overall["gt_labels"].append(gt_crop)
            month_overall["pred_labels"].append(pred_label)
            month_overall["field_id"].append(fid)
            month_overall["month"].append(month)
            month_overall["abs_res"].append(abs(pred_prod - gt_prod))

            per_field_month_rows.append({
                "model": model_name,
                "field_id": fid,
                "month": month,
                "y_true": gt_prod,
                "y_pred": pred_prod,
                "gt_label": gt_crop,
                "pred_label": pred_label,
                "abs_res": abs(pred_prod - gt_prod),
            })

    for month, buf in monthly_buffers.items():
        if len(buf["y_true"]) < 2:
            continue

        metrics = evaluate(
            buf["y_true"], buf["y_pred"], buf["gt_labels"], buf["pred_labels"]
        )

        all_results.append({
            "model": model_name,
            "month": month,
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "r2": metrics["r2"],
            "class_acc": metrics["class_acc"],
            "n": metrics["n"],
        })

    if len(month_overall["y_true"]) >= 2:
        metrics = evaluate(
            month_overall["y_true"], month_overall["y_pred"],
            month_overall["gt_labels"], month_overall["pred_labels"]
        )

        all_results.append({
            "model": model_name,
            "month": "month_agg_overall",
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "r2": metrics["r2"],
            "class_acc": metrics["class_acc"],
            "n": metrics["n"],
        })


per_field_month_df = pd.DataFrame(per_field_month_rows)

# Exact (field_id, month) pairs seen in the model data — no all_months needed
field_month_index = (
    per_field_month_df[["field_id", "month"]]
    .drop_duplicates()
    .reset_index(drop=True)
)

b = Baselines()

# ── Compute field-level baselines and choose best ────────────────────────────
baseline_dfs = {}        # field-level
baseline_dfs_fm = {}     # field-month-level

for baseline_name in BASELINE_NAMES:
    baseline_df = compute_baseline_predictions(baseline_name, gt_indexed, b)
    baseline_dfs[baseline_name] = baseline_df

    # field_agg_overall — one row per field, unweighted
    metrics_field = evaluate(
        baseline_df["y_true"].values,
        baseline_df["y_pred"].values,
        baseline_df["gt_label"].values,
        baseline_df["pred_label"].values,
    )
    all_results.append({
        "model": baseline_name,
        "month": "overall",          # for best-baseline selection only
        "mae": metrics_field["mae"],
        "rmse": metrics_field["rmse"],
        "r2": metrics_field["r2"],
        "class_acc": metrics_field["class_acc"],
        "n": metrics_field["n"],
    })
    all_results.append({
        "model": baseline_name,
        "month": "field_agg_overall",
        "mae": metrics_field["mae"],
        "rmse": metrics_field["rmse"],
        "r2": metrics_field["r2"],
        "class_acc": metrics_field["class_acc"],
        "n": metrics_field["n"],
    })

    # field-month expanded — weighted by months per field
    baseline_df_fm = compute_baseline_predictions_fm(baseline_name, gt_indexed, b, field_month_index)
    baseline_dfs_fm[baseline_name] = baseline_df_fm

    metrics_fm = evaluate(
        baseline_df_fm["y_true"].values,
        baseline_df_fm["y_pred"].values,
        baseline_df_fm["gt_label"].values,
        baseline_df_fm["pred_label"].values,
    )
    all_results.append({
        "model": baseline_name,
        "month": "month_agg_overall",   # comparable to model's month_agg_overall
        "mae": metrics_fm["mae"],
        "rmse": metrics_fm["rmse"],
        "r2": metrics_fm["r2"],
        "class_acc": metrics_fm["class_acc"],
        "n": metrics_fm["n"],
    })

baseline_results_df = pd.DataFrame(all_results)
baseline_overall_df = baseline_results_df[
    (baseline_results_df["month"] == "overall") &
    (baseline_results_df["model"].isin(BASELINE_NAMES))
].copy()

best_baseline_row = baseline_overall_df.loc[baseline_overall_df["r2"].idxmax()]
best_baseline_name = best_baseline_row["model"]
best_baseline_df = baseline_dfs[best_baseline_name]
best_baseline_df_fm = baseline_dfs_fm[best_baseline_name]   # <-- field-month version

print("\nBest baseline chosen automatically:")
print(best_baseline_row.to_string())




field_overall_agg = (
    per_field_month_df
    .groupby(["model", "field_id"], as_index=False)
    .agg(
        y_true_mean=("y_true", "mean"),
        y_pred_mean=("y_pred", "mean"),
        gt_label_dominant=("gt_label", lambda x: x.value_counts().index[0]),
        pred_label_dominant=("pred_label", lambda x: x.value_counts().index[0]),
        abs_res_mean=("abs_res", "mean"),
    )
)

for model_name, sub in field_overall_agg.groupby("model"):
    metrics = evaluate(
        sub["y_true_mean"].to_numpy(),
        sub["y_pred_mean"].to_numpy(),
        sub["gt_label_dominant"].to_numpy(),
        sub["pred_label_dominant"].to_numpy(),
    )

    all_results.append({
        "model": model_name,
        "month": "field_agg_overall",
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "r2": metrics["r2"],
        "class_acc": metrics["class_acc"],
        "n": metrics["n"],
    })


# ── Per-month permutation tests (vs field-month baseline) ─────────────────────
for model_name in per_field_month_df["model"].unique():
    model_months = sorted(per_field_month_df.loc[
        per_field_month_df["model"] == model_name, "month"
    ].unique())

    for month in model_months:
        model_month_df = per_field_month_df[
            (per_field_month_df["model"] == model_name) &
            (per_field_month_df["month"] == month)
        ][["field_id", "abs_res"]].copy()

        # Use field-month baseline: filter to same month
        baseline_month_df = best_baseline_df_fm[
            best_baseline_df_fm["month"] == month
        ][["field_id", "abs_res"]].copy()

        merged_month = pd.merge(
            model_month_df,
            baseline_month_df,
            on="field_id",
            suffixes=("_model", "_baseline")
        )

        if len(merged_month) < 2:
            continue

        err_model = merged_month["abs_res_model"].to_numpy()
        err_baseline = merged_month["abs_res_baseline"].to_numpy()

        perm_res = stats.permutation_test(
            data=(err_baseline, err_model),
            statistic=mean_improvement,
            permutation_type="samples",
            alternative="greater",
            n_resamples=100_000,
            random_state=0,
            vectorized=True,
        )

        perm_test_rows.append({
            "model": model_name,
            "month": month,
            "test_type": "month_level",
            "baseline": best_baseline_name,
            "n": len(merged_month),
            "mean_improvement": float(mean_improvement(err_baseline, err_model)),
            "p_value": float(perm_res.pvalue),
        })


# ── Field-aggregated permutation tests (vs field-level baseline) ──────────────
for model_name in per_field_month_df["model"].unique():
    model_field_agg = (
        per_field_month_df[per_field_month_df["model"] == model_name]
        .groupby("field_id", as_index=False)
        .agg(
            mean_abs_res=("abs_res", "mean"),
            n_months=("month", "nunique"),
        )
    )

    merged_field = pd.merge(
        model_field_agg,
        best_baseline_df[["field_id", "abs_res"]],
        on="field_id",
        suffixes=("_model", "_baseline")
    )

    if len(merged_field) < 2:
        continue

    err_model = merged_field["mean_abs_res"].to_numpy()
    err_baseline = merged_field["abs_res"].to_numpy()

    perm_res = stats.permutation_test(
        data=(err_baseline, err_model),
        statistic=mean_improvement,
        permutation_type="samples",
        alternative="greater",
        n_resamples=100_000,
        random_state=0,
        vectorized=True,
    )

    perm_test_rows.append({
        "model": model_name,
        "month": "overall",
        "test_type": "field_aggregated",
        "baseline": best_baseline_name,
        "n": len(merged_field),
        "mean_improvement": float(mean_improvement(err_baseline, err_model)),
        "p_value": float(perm_res.pvalue),
    })

# ── Month-aggregated-overall permutation tests (vs field-month baseline) ─────
for model_name in per_field_month_df["model"].unique():
    # All field-month rows for this model
    model_all_fm = per_field_month_df[
        per_field_month_df["model"] == model_name
    ][["field_id", "month", "abs_res"]].copy()

    # Match to baseline at the same (field_id, month) pairs
    baseline_all_fm = best_baseline_df_fm[["field_id", "month", "abs_res"]].copy()

    merged_all = pd.merge(
        model_all_fm,
        baseline_all_fm,
        on=["field_id", "month"],
        suffixes=("_model", "_baseline"),
    )

    if len(merged_all) < 2:
        continue

    err_model = merged_all["abs_res_model"].to_numpy()
    err_baseline = merged_all["abs_res_baseline"].to_numpy()

    perm_res = stats.permutation_test(
        data=(err_baseline, err_model),
        statistic=mean_improvement,
        permutation_type="samples",
        alternative="greater",
        n_resamples=100_000,
        random_state=0,
        vectorized=True,
    )

    perm_test_rows.append({
        "model": model_name,
        "month": "month_agg_overall",
        "test_type": "month_aggregated",
        "baseline": best_baseline_name,
        "n": len(merged_all),
        "mean_improvement": float(mean_improvement(err_baseline, err_model)),
        "p_value": float(perm_res.pvalue),
    })

# ── Save outputs ──────────────────────────────────────────────────────────────
results_df = pd.DataFrame(all_results)
perm_tests_df = pd.DataFrame(perm_test_rows)

monthly_df = results_df[
    ~results_df["month"].isin(["overall", "month_agg_overall", "field_agg_overall"])
].sort_values(["model", "month"])

month_overall_df = results_df[
    results_df["month"] == "month_agg_overall"
].sort_values(["model"])

field_overall_df_out = results_df[
    results_df["month"] == "field_agg_overall"
].sort_values(["model"])

results_df_final = pd.concat(
    [monthly_df, month_overall_df, field_overall_df_out],
    ignore_index=True
)

results_df_final.to_csv("results/test_results.csv", index=False)
perm_tests_df.to_csv("results/permutation_results.csv", index=False)

print("\n=== Metrics ===")
print(results_df_final.to_string(index=False))

print("\n=== Permutation tests ===")
if len(perm_tests_df):
    print(perm_tests_df.to_string(index=False))
else:
    print("No permutation test results produced.")

print("\nThank you very much for your help!")