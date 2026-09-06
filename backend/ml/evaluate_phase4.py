"""
Phase 4 Evaluation Suite: Real Dataset Testing & Performance Profiling

Evaluates the complete Network Traffic Classification pipeline against authentic
CIC-IDS2017 flow data from `dataset/processed/training_dataset.csv`.

Measures:
1. In-memory validation & feature extraction latency
2. Vectorized XGBoost multiclass model inference throughput (flows/sec)
3. End-to-end FastAPI HTTP batch API performance (/predict/batch)
4. Memory consumption (RSS)
5. Model prediction distributions & confidence metrics
6. Rigorous ground-truth evaluation (Accuracy, Precision, Recall, Macro/Weighted F1, Confusion Matrix)
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Ensure stdout handles utf-8 cleanly on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.ml.predictor import NetworkTrafficPredictor, normalize_display_label

DATASET_PATH = PROJECT_ROOT / "dataset" / "processed" / "training_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "models" / "phase4_evaluation"
API_BATCH_URL = "http://127.0.0.1:8000/predict/batch"


def get_current_rss_mb() -> float:
    """Returns the current process resident set size in Megabytes."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def load_stratified_real_dataset(target_size: int = 50_000, random_state: int = 42) -> pd.DataFrame:
    """
    Streams and samples authentic CIC-IDS2017 records across the entire `training_dataset.csv`
    ensuring representation across all 15 classes.
    """
    print(f"Streaming authentic data from {DATASET_PATH}...")
    chunk_size = 200_000
    class_pools: Dict[str, List[pd.DataFrame]] = {}

    for chunk in pd.read_csv(DATASET_PATH, chunksize=chunk_size):
        for label, group in chunk.groupby("Label"):
            nl = normalize_display_label(str(label))
            if nl not in class_pools:
                class_pools[nl] = []
            if len(group) > 3000:
                class_pools[nl].append(group.sample(n=3000, random_state=random_state))
            else:
                class_pools[nl].append(group)

    print("Collating pool of authentic samples...")
    combined_groups: Dict[str, pd.DataFrame] = {}
    total_available = 0
    for nl, dfs in class_pools.items():
        combined_groups[nl] = pd.concat(dfs, ignore_index=True)
        cnt = len(combined_groups[nl])
        total_available += cnt
        print(f"  {nl:<26}: {cnt:>6,} available")

    print(f"Total pool collected: {total_available:,} rows across {len(combined_groups)} classes.")

    # Target allocations per class to reach exactly target_size
    sampled_dfs: List[pd.DataFrame] = []
    # Rare classes: keep all available
    rare_classes = {
        "Heartbleed", "Web Attack - Sql Injection", "Infiltration",
        "Web Attack - XSS", "Web Attack - Brute Force", "Bot"
    }

    allocated_count = 0
    for cls in rare_classes:
        if cls in combined_groups:
            sampled_dfs.append(combined_groups[cls])
            allocated_count += len(combined_groups[cls])

    remaining_needed = target_size - allocated_count

    # Major classes: allocate proportionally
    major_classes = [c for c in combined_groups.keys() if c not in rare_classes]
    weights = {
        "BENIGN": 0.40,
        "DoS Hulk": 0.15,
        "DDoS": 0.15,
        "PortScan": 0.15,
        "DoS GoldenEye": 0.05,
        "FTP-Patator": 0.04,
        "SSH-Patator": 0.03,
        "DoS slowloris": 0.02,
        "DoS Slowhttptest": 0.01,
    }

    for cls in major_classes:
        df_cls = combined_groups[cls]
        w = weights.get(cls, 0.02)
        n = min(len(df_cls), max(10, int(remaining_needed * w)))
        sampled_dfs.append(df_cls.sample(n=n, random_state=random_state))

    full_sample = pd.concat(sampled_dfs, ignore_index=True)
    if len(full_sample) < target_size and "BENIGN" in combined_groups:
        diff = target_size - len(full_sample)
        add_benign = combined_groups["BENIGN"].sample(n=diff, random_state=random_state)
        full_sample = pd.concat([full_sample, add_benign], ignore_index=True)

    full_sample = full_sample.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    if len(full_sample) > target_size:
        full_sample = full_sample.iloc[:target_size].copy()

    # Normalize GroundTruth column
    full_sample["GroundTruth"] = full_sample["Label"].apply(
        lambda l: normalize_display_label(str(l))
    )

    print(f"Constructed stratified test dataset: {len(full_sample):,} flows across {full_sample['GroundTruth'].nunique()} classes.")
    return full_sample


def evaluate_batch_sizes(
    df: pd.DataFrame,
    predictor: NetworkTrafficPredictor,
    batch_sizes: List[int]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Evaluates end-to-end performance metrics across multiple batch sizes.
    """
    results: List[Dict[str, Any]] = []
    all_batch_predictions: Dict[int, List[Dict[str, Any]]] = {}

    expected_features = predictor.feature_names
    X_all = df[expected_features]

    for batch_size in batch_sizes:
        if batch_size > len(df):
            continue

        print(f"\n--- Benchmarking Batch Size: {batch_size:,} flows ---")
        sub_df = df.iloc[:batch_size].copy()
        sub_X = sub_df[expected_features]

        # 1. Measure Ingestion & CSV Parsing Simulation
        csv_text = sub_X.to_csv(index=False)
        t0 = time.perf_counter()
        parsed_df = pd.read_csv(pd.io.common.StringIO(csv_text))
        t_parse = time.perf_counter() - t0

        # 2. Measure Validation Time (verifying 61 columns and numeric types)
        t0 = time.perf_counter()
        is_valid = set(expected_features).issubset(set(parsed_df.columns)) and not parsed_df.isna().any().any()
        t_validate = time.perf_counter() - t0

        # 3. Measure In-Memory Predictor Model Inference (Pure XGBoost core)
        mem_before = get_current_rss_mb()
        t0 = time.perf_counter()
        preds_in_memory = predictor.predict_batch(sub_X)
        t_model = time.perf_counter() - t0
        mem_after = get_current_rss_mb()

        throughput_model = batch_size / t_model if t_model > 0 else 0.0

        # 4. Measure FastAPI HTTP API batch performance (if server is up and batch size <= 25,000)
        t_api = None
        throughput_api = None
        api_status = "Skipped"

        if batch_size <= 25_000:
            try:
                # Prepare JSON payload
                samples_list = sub_X.to_dict(orient="records")
                payload = json.dumps({"samples": samples_list}).encode("utf-8")
                
                req = urllib.request.Request(
                    API_BATCH_URL,
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST"
                )
                t0 = time.perf_counter()
                with urllib.request.urlopen(req, timeout=120) as resp:
                    api_resp = json.loads(resp.read().decode("utf-8"))
                    assert api_resp["total_samples"] == batch_size
                t_api = time.perf_counter() - t0
                throughput_api = batch_size / t_api
                api_status = "Success"
            except Exception as exc:
                api_status = f"Failed: {exc}"
                t_api = None

        total_elapsed = t_parse + t_validate + t_model
        overall_throughput = batch_size / total_elapsed if total_elapsed > 0 else 0.0

        res_row = {
            "batch_size": batch_size,
            "csv_parsing_sec": round(t_parse, 4),
            "validation_sec": round(t_validate, 5),
            "model_prediction_sec": round(t_model, 4),
            "model_throughput_flows_sec": round(throughput_model, 1),
            "api_http_sec": round(t_api, 4) if t_api is not None else "N/A",
            "api_throughput_flows_sec": round(throughput_api, 1) if throughput_api is not None else "N/A",
            "api_status": api_status,
            "total_processing_sec": round(total_elapsed, 4),
            "overall_throughput_flows_sec": round(overall_throughput, 1),
            "rss_memory_mb": round(mem_after, 2),
            "rss_memory_delta_mb": round(mem_after - mem_before, 2),
        }
        results.append(res_row)
        all_batch_predictions[batch_size] = preds_in_memory

        print(f"  Model Latency:  {t_model*1000:.2f} ms ({throughput_model:,.1f} flows/sec)")
        if t_api is not None:
            print(f"  HTTP API Batch: {t_api*1000:.2f} ms ({throughput_api:,.1f} flows/sec)")
        print(f"  Total Ingest+Infer: {total_elapsed:.4f} s (Overall: {overall_throughput:,.1f} flows/sec)")
        print(f"  Process Memory: {mem_after:.1f} MB (Delta: {mem_after - mem_before:+.2f} MB)")

    perf_df = pd.DataFrame(results)
    return perf_df, all_batch_predictions


def analyze_predictions(predictions: List[Dict[str, Any]], all_classes: List[str]) -> Dict[str, Any]:
    """
    Computes statistical telemetry based on model predictions.
    """
    total_flows = len(predictions)
    benign_count = sum(1 for p in predictions if not p["is_attack"])
    attack_count = sum(1 for p in predictions if p["is_attack"])
    attack_percentage = (attack_count / total_flows) * 100 if total_flows > 0 else 0.0

    confidences = [p["confidence"] for p in predictions]
    conf_mean = float(np.mean(confidences))
    conf_median = float(np.median(confidences))
    conf_min = float(np.min(confidences))
    conf_max = float(np.max(confidences))
    conf_std = float(np.std(confidences))

    class_counts: Dict[str, int] = {cls: 0 for cls in all_classes}
    for p in predictions:
        pred_name = p["prediction"]
        class_counts[pred_name] = class_counts.get(pred_name, 0) + 1

    class_distribution = [
        {
            "class": cls,
            "count": class_counts[cls],
            "percentage": round((class_counts[cls] / total_flows) * 100, 2) if total_flows > 0 else 0.0,
            "is_attack": cls != "BENIGN",
        }
        for cls in sorted(class_counts.keys(), key=lambda c: class_counts[c], reverse=True)
    ]

    return {
        "total_flows": total_flows,
        "benign_flows": benign_count,
        "attack_flows": attack_count,
        "attack_percentage": round(attack_percentage, 2),
        "confidence_metrics": {
            "mean": round(conf_mean, 4),
            "median": round(conf_median, 4),
            "min": round(conf_min, 4),
            "max": round(conf_max, 4),
            "std": round(conf_std, 4),
        },
        "class_distribution": class_distribution,
    }


def evaluate_ground_truth(
    y_true: List[str],
    y_pred: List[str],
    all_classes: List[str]
) -> Dict[str, Any]:
    """
    Performs rigorous ground-truth machine learning evaluation.
    """
    acc = accuracy_score(y_true, y_pred)
    macro_p = precision_score(y_true, y_pred, labels=all_classes, average="macro", zero_division=0)
    macro_r = recall_score(y_true, y_pred, labels=all_classes, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, labels=all_classes, average="macro", zero_division=0)
    
    weighted_p = precision_score(y_true, y_pred, labels=all_classes, average="weighted", zero_division=0)
    weighted_r = recall_score(y_true, y_pred, labels=all_classes, average="weighted", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=all_classes, average="weighted", zero_division=0)

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=all_classes,
        output_dict=True,
        zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=all_classes)
    cm_list = cm.tolist()

    return {
        "accuracy": round(float(acc), 4),
        "macro_metrics": {
            "precision": round(float(macro_p), 4),
            "recall": round(float(macro_r), 4),
            "f1": round(float(macro_f1), 4),
        },
        "weighted_metrics": {
            "precision": round(float(weighted_p), 4),
            "recall": round(float(weighted_r), 4),
            "f1": round(float(weighted_f1), 4),
        },
        "per_class_metrics": report_dict,
        "confusion_matrix": {
            "classes": all_classes,
            "matrix": cm_list,
        },
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("PHASE 4: REAL DATASET TESTING & PERFORMANCE EVALUATION")
    print("=" * 70)

    # 1. Initialize Predictor
    predictor = NetworkTrafficPredictor()
    all_classes = predictor.classes
    print(f"Predictor initialized: {predictor.num_classes} classes, {predictor.num_features} features.")

    # 2. Stream and Sample Real Dataset
    df_eval = load_stratified_real_dataset(target_size=50_000, random_state=42)

    # 3. Test Batch Sizes
    batch_sizes = [100, 1_000, 5_000, 10_000, 25_000, 50_000]
    perf_df, predictions_dict = evaluate_batch_sizes(df_eval, predictor, batch_sizes)

    # Save performance results CSV
    perf_csv_path = OUTPUT_DIR / "performance_results.csv"
    perf_df.to_csv(perf_csv_path, index=False)
    print(f"\nSaved performance results to: {perf_csv_path}")

    # 4. Analyze Predictions on the Evaluated Dataset
    max_batch = max(predictions_dict.keys())
    full_preds = predictions_dict[max_batch]
    pred_analysis = analyze_predictions(full_preds, all_classes)

    pred_stat_rows = []
    for item in pred_analysis["class_distribution"]:
        pred_stat_rows.append({
            "class": item["class"],
            "category": "Attack" if item["is_attack"] else "Benign",
            "predicted_count": item["count"],
            "percentage_of_traffic": item["percentage"],
        })
    pred_stat_df = pd.DataFrame(pred_stat_rows)
    pred_stat_csv_path = OUTPUT_DIR / "prediction_statistics.csv"
    pred_stat_df.to_csv(pred_stat_csv_path, index=False)
    print(f"Saved prediction statistics to: {pred_stat_csv_path}")

    # 5. Ground-Truth Model Evaluation
    y_true = df_eval.iloc[:max_batch]["GroundTruth"].tolist()
    y_pred = [p["prediction"] for p in full_preds]
    gt_eval = evaluate_ground_truth(y_true, y_pred, all_classes)

    gt_eval_path = OUTPUT_DIR / "ground_truth_evaluation.json"
    with open(gt_eval_path, "w", encoding="utf-8") as f:
        json.dump(gt_eval, f, indent=2)
    print(f"Saved ground truth evaluation to: {gt_eval_path}")

    # 6. Save Overall Summary JSON
    summary_data = {
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_type": "XGBoost Multiclass Classifier (hist)",
        "features_evaluated": predictor.num_features,
        "classes_evaluated": predictor.num_classes,
        "total_test_samples": len(df_eval),
        "performance_scaling": perf_df.to_dict(orient="records"),
        "prediction_statistics": pred_analysis,
        "ground_truth_metrics": {
            "accuracy": gt_eval["accuracy"],
            "macro_f1": gt_eval["macro_metrics"]["f1"],
            "weighted_f1": gt_eval["weighted_metrics"]["f1"],
            "macro_precision": gt_eval["macro_metrics"]["precision"],
            "macro_recall": gt_eval["macro_metrics"]["recall"],
        },
    }

    summary_json_path = OUTPUT_DIR / "phase4_performance_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved performance summary to: {summary_json_path}")

    # Print Key Results Summary
    print("\n" + "=" * 70)
    print("PHASE 4 EVALUATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"Overall Accuracy:       {gt_eval['accuracy']*100:.2f}%")
    print(f"Macro F1-Score:         {gt_eval['macro_metrics']['f1']:.4f}")
    print(f"Weighted F1-Score:      {gt_eval['weighted_metrics']['f1']:.4f}")
    print(f"Mean Confidence:        {pred_analysis['confidence_metrics']['mean']*100:.2f}%")
    print(f"Median Confidence:      {pred_analysis['confidence_metrics']['median']*100:.2f}%")
    print(f"Attack Percentage:      {pred_analysis['attack_percentage']:.2f}%")
    print("\nThroughput Scaling (Flows / Second):")
    for _, row in perf_df.iterrows():
        b = int(row['batch_size'])
        tp_model = row['model_throughput_flows_sec']
        tp_api = row['api_throughput_flows_sec']
        print(f"  Batch {b:>6,}: Model Core = {tp_model:>9,.1f} flows/sec | HTTP API = {str(tp_api):>9} flows/sec")
    print("=" * 70)


if __name__ == "__main__":
    main()
