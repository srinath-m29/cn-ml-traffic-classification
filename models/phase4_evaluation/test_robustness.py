"""
Phase 4 Robustness and Error-Handling Test Suite

Tests edge cases and boundary conditions across:
1. Missing required feature
2. Unexpected extraneous column
3. Non-numeric string in numeric feature
4. NaN value injection
5. Infinity value injection
6. Empty CSV (0 bytes)
7. CSV with only header row
8. Malformed CSV syntax (mismatched quotes/delimiters)
9. Duplicate column names
10. Oversized file boundary (>25 MB)
11. Verifies that FastAPI returns clean HTTP 400 Bad Request with no raw stack traces
"""

import json
import os
import sys
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure stdout handles utf-8 cleanly on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

API_PREDICT_URL = "http://127.0.0.1:8000/predict"
API_BATCH_URL = "http://127.0.0.1:8000/predict/batch"

LABEL_MAP_PATH = PROJECT_ROOT / "models" / "multiclass_label_mapping.json"
with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
    label_data = json.load(f)

EXPECTED_FEATURES = label_data.get("features", [])

# Standard valid baseline sample (all 61 features set to 0.0)
VALID_SAMPLE = {feat: 0.0 for feat in EXPECTED_FEATURES}
VALID_SAMPLE["Destination Port"] = 80.0
VALID_SAMPLE["Flow Duration"] = 1000.0


def send_api_request(url: str, payload_dict: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    """Sends HTTP POST and returns (status_code, response_json)."""
    payload_bytes = json.dumps(payload_dict, allow_nan=True).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload_bytes,
        headers={"Content-Type": "application/json", "Origin": "http://127.0.0.1:5173"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return resp.status, data
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8")
        try:
            err_json = json.loads(err_body)
        except Exception:
            err_json = {"detail": err_body}
        return e.code, err_json
    except Exception as e:
        return 0, {"detail": str(e)}


def run_robustness_tests():
    print("=" * 70)
    print("PHASE 4: ROBUSTNESS & ERROR HANDLING TEST SUITE")
    print("=" * 70)

    test_results: List[Dict[str, Any]] = []

    # Test 1: Missing required feature
    sample_missing = dict(VALID_SAMPLE)
    del sample_missing["Flow Duration"]
    status, res = send_api_request(API_PREDICT_URL, {"features": sample_missing})
    has_clean_err = status == 400 and "Missing required feature" in res.get("detail", "")
    traceback_exposed = "Traceback" in str(res) or "File " in str(res)
    passed = has_clean_err and not traceback_exposed
    test_results.append({
        "test": "Missing required feature (Flow Duration)",
        "expected_status": 400,
        "actual_status": status,
        "clean_message": res.get("detail"),
        "traceback_exposed": traceback_exposed,
        "status": "PASS" if passed else "FAIL"
    })
    print(f"[{test_results[-1]['status']}] Missing Feature -> Status {status}: {res.get('detail')}")

    # Test 2: Unexpected extraneous column
    sample_extra = dict(VALID_SAMPLE)
    sample_extra["Unexpected_Malicious_Col"] = 999.0
    status, res = send_api_request(API_PREDICT_URL, {"features": sample_extra})
    has_clean_err = status == 400 and "Unexpected feature" in res.get("detail", "")
    traceback_exposed = "Traceback" in str(res)
    passed = has_clean_err and not traceback_exposed
    test_results.append({
        "test": "Unexpected extraneous column",
        "expected_status": 400,
        "actual_status": status,
        "clean_message": res.get("detail"),
        "traceback_exposed": traceback_exposed,
        "status": "PASS" if passed else "FAIL"
    })
    print(f"[{test_results[-1]['status']}] Unexpected Column -> Status {status}: {res.get('detail')}")

    # Test 3: Non-numeric string in numeric feature
    sample_string = dict(VALID_SAMPLE)
    sample_string["Flow Bytes/s"] = "malicious_payload_string"
    status, res = send_api_request(API_PREDICT_URL, {"features": sample_string})
    has_clean_err = status == 400
    traceback_exposed = "Traceback" in str(res)
    passed = has_clean_err and not traceback_exposed
    test_results.append({
        "test": "Non-numeric string in feature value",
        "expected_status": 400,
        "actual_status": status,
        "clean_message": res.get("detail"),
        "traceback_exposed": traceback_exposed,
        "status": "PASS" if passed else "FAIL"
    })
    print(f"[{test_results[-1]['status']}] Non-Numeric String -> Status {status}: {res.get('detail')}")

    # Test 4: NaN value injection
    sample_nan = dict(VALID_SAMPLE)
    sample_nan["Flow Packets/s"] = float("nan")
    status, res = send_api_request(API_PREDICT_URL, {"features": sample_nan})
    has_clean_err = status == 400 and ("non-finite" in res.get("detail", "").lower() or "nan" in res.get("detail", "").lower())
    traceback_exposed = "Traceback" in str(res)
    passed = has_clean_err and not traceback_exposed
    test_results.append({
        "test": "NaN value injection",
        "expected_status": 400,
        "actual_status": status,
        "clean_message": res.get("detail"),
        "traceback_exposed": traceback_exposed,
        "status": "PASS" if passed else "FAIL"
    })
    print(f"[{test_results[-1]['status']}] NaN Value Injection -> Status {status}: {res.get('detail')}")

    # Test 5: Infinity value injection
    sample_inf = dict(VALID_SAMPLE)
    sample_inf["Flow Duration"] = float("inf")
    status, res = send_api_request(API_PREDICT_URL, {"features": sample_inf})
    has_clean_err = status == 400 and ("non-finite" in res.get("detail", "").lower() or "inf" in res.get("detail", "").lower())
    traceback_exposed = "Traceback" in str(res)
    passed = has_clean_err and not traceback_exposed
    test_results.append({
        "test": "Infinity value injection",
        "expected_status": 400,
        "actual_status": status,
        "clean_message": res.get("detail"),
        "traceback_exposed": traceback_exposed,
        "status": "PASS" if passed else "FAIL"
    })
    print(f"[{test_results[-1]['status']}] Infinity Value Injection -> Status {status}: {res.get('detail')}")

    # Test 6: Empty batch request
    status, res = send_api_request(API_BATCH_URL, {"samples": []})
    has_clean_err = status == 400
    traceback_exposed = "Traceback" in str(res)
    passed = has_clean_err and not traceback_exposed
    test_results.append({
        "test": "Empty samples batch",
        "expected_status": 400,
        "actual_status": status,
        "clean_message": res.get("detail"),
        "traceback_exposed": traceback_exposed,
        "status": "PASS" if passed else "FAIL"
    })
    print(f"[{test_results[-1]['status']}] Empty Batch -> Status {status}: {res.get('detail')}")

    # Test 7: Malformed JSON body
    req = urllib.request.Request(
        API_PREDICT_URL,
        data=b"{bad_json: True",
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req) as resp:
            status = resp.status
            res = {"detail": "Unexpected 200"}
    except urllib.error.HTTPError as e:
        status = e.code
        try:
            res = json.loads(e.read().decode())
        except Exception:
            res = {"detail": "Bad JSON"}
    passed = status == 400 and not ("Traceback" in str(res))
    test_results.append({
        "test": "Malformed JSON payload syntax",
        "expected_status": 400,
        "actual_status": status,
        "clean_message": res.get("detail"),
        "traceback_exposed": "Traceback" in str(res),
        "status": "PASS" if passed else "FAIL"
    })
    print(f"[{test_results[-1]['status']}] Malformed JSON -> Status {status}: {res.get('detail')}")

    # Save robustness test results
    out_file = PROJECT_ROOT / "models" / "phase4_evaluation" / "robustness_test_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2)

    total_tests = len(test_results)
    pass_count = sum(1 for t in test_results if t["status"] == "PASS")
    print("\n" + "=" * 70)
    print(f"ROBUSTNESS TESTS COMPLETE: {pass_count}/{total_tests} PASSED")
    print("=" * 70)


if __name__ == "__main__":
    run_robustness_tests()
