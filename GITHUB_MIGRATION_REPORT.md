# GitHub Migration Report

## Repository
https://github.com/srinath-m29/cn-ml-traffic-classification

## Branch
main

## Commit
06ecc3053dd7e4a36852111113362090352b632e

## Git Status
On branch main  
Your branch is up to date with 'origin/main'.  
nothing to commit, working tree clean

## Files Tracked
151 files

## Files/Folders Excluded
- `.venv/` (Python virtual environment and site packages)
- `frontend/node_modules/` (Node package dependencies)
- `frontend/dist/` (Vite production bundle output)
- `frontend/build/` (Build artifacts)
- `__pycache__/` and `*.pyc` (Python bytecode cache)
- `.vscode/` and `.idea/` (Editor / IDE metadata)
- `*.log` (Execution and debug logs)
- `.env` and `.env.*` (Environment variable secrets)
- `dataset/raw/` (Raw CIC-IDS2017 PCAP CSV files: ~840 MB)
- `dataset/processed/*.csv` (Large processed datasets: `network_traffic_cleaned.csv` at 841.82 MB and `training_dataset.csv` at 719.37 MB)
- `*.joblib` / `models/random_forest_binary.joblib` (Oversized binary model file: 112.4 MB, exceeds GitHub 100 MB limit)

## Large Files
| File Path | Size | Status | Handling |
| :--- | :--- | :--- | :--- |
| `dataset/processed/network_traffic_cleaned.csv` | 841.82 MB | Excluded | Ignored via `.gitignore` (exceeds GitHub limit; not code) |
| `dataset/processed/training_dataset.csv` | 719.37 MB | Excluded | Ignored via `.gitignore` (exceeds GitHub limit; not code) |
| `dataset/raw/MachineLearningCVE/Wednesday-workingHours.pcap_ISCX.csv` | 214.74 MB | Excluded | Ignored via `.gitignore` (exceeds GitHub limit) |
| `dataset/raw/MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv` | 168.73 MB | Excluded | Ignored via `.gitignore` (exceeds GitHub limit) |
| `dataset/raw/MachineLearningCVE/Tuesday-WorkingHours.pcap_ISCX.csv` | 128.82 MB | Excluded | Ignored via `.gitignore` (exceeds GitHub limit) |
| `models/random_forest_binary.joblib` | 112.40 MB | Excluded | Ignored via `.gitignore` (exceeds GitHub 100 MB hard limit; preserved safely on local disk) |
| `dataset/raw/MachineLearningCVE/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv` | 79.25 MB | Excluded | Ignored via `.gitignore` |
| `dataset/raw/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` | 73.55 MB | Excluded | Ignored via `.gitignore` |
| `dataset/raw/MachineLearningCVE/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` | 73.34 MB | Excluded | Ignored via `.gitignore` |
| `dataset/raw/MachineLearningCVE/Friday-WorkingHours-Morning.pcap_ISCX.csv` | 55.62 MB | Excluded | Ignored via `.gitignore` |
| `dataset/raw/MachineLearningCVE/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv` | 49.61 MB | Excluded | Ignored via `.gitignore` |
| `models/xgboost_multiclass_gpu.json` | 37.95 MB | **Included** | Safely within GitHub 100 MB limit; tracked and pushed to GitHub |
| `models/temporal_multiclass/friday_predictions.csv` | 13.13 MB | **Included** | Safely within GitHub 100 MB limit; tracked and pushed to GitHub |

## Model
- **Model Path**: `models/xgboost_multiclass_gpu.json`
- **Model Size**: 37.95 MB (39,795,502 bytes)
- **Whether Included/Excluded**: **Included**
- **Reason**: Safely below GitHub's 100 MB maximum single-file upload limit. Preserves 100% of the pre-trained weights and enables immediate out-of-the-box inference for FastAPI and frontend dashboard without retraining.
- **Model Metadata**: `models/multiclass_label_mapping.json` (15 classes) and `models/selected_features.txt` (61 features) are also tracked and included.

## Dataset
- **Raw Dataset Handling**: Excluded via `.gitignore` (`dataset/raw/`). Contains ~840 MB of raw capture CSVs which exceed GitHub limits and are not required for code deployment.
- **Processed Dataset Handling**: Excluded via `.gitignore` (`dataset/processed/*.csv`). `network_traffic_cleaned.csv` (841.82 MB) and `training_dataset.csv` (719.37 MB) exceed GitHub file limits.
- **Feature Schema Preserved**: `dataset/processed/selected_features.txt` (1 KB) and `models/selected_features.txt` (1 KB) containing all 61 selected feature names are actively tracked and included in the repository.
- **Reason for Exclusion**: Exceeds GitHub's 100 MB maximum file limit and Git version control is designed for source code, configuration, and lightweight deployment artifacts, not multi-hundred megabyte raw training datasets.

## Security Check
- **Secrets Found**: None (0 secrets, 0 API keys, 0 private keys, 0 credentials found).
- **Environment Files**: No `.env` or credential files existed in the source directory.
- **Authentication**: Git Credential Manager was utilized with zero hardcoded credentials or tokens exposed.

## Backend Verification
- **Test Command**:
  ```powershell
  .\.venv\Scripts\python.exe -m uvicorn backend.services.main:app --host 127.0.0.1 --port 8000
  ```
- **Endpoint `GET /health`**:
  - HTTP Status: `200 OK`
  - Response: `{"status": "healthy", "model_loaded": True}`
- **Endpoint `GET /model-info`**:
  - HTTP Status: `200 OK`
  - Model Type: `XGBoost Multiclass Classifier (hist)`
  - Number of Features: `61`
  - Number of Classes: `15`
  - Classes: `['BENIGN', 'Bot', 'DDoS', 'DoS GoldenEye', 'DoS Hulk', 'DoS Slowhttptest', 'DoS slowloris', 'FTP-Patator', 'Heartbleed', 'Infiltration', 'PortScan', 'SSH-Patator', 'Web Attack - Brute Force', 'Web Attack - Sql Injection', 'Web Attack - XSS']`

## Frontend Verification
- **Test Command**:
  ```powershell
  cd frontend
  npm install
  npm run build
  ```
- **Build Status**: Successful (0 errors, built in 524ms)
- **Artifacts Produced**:
  - `dist/index.html` (0.92 kB)
  - `dist/assets/index-jPCTVY3W.css` (22.30 kB)
  - `dist/assets/index-DQlsgHtA.js` (651.74 kB)

---

## Clone Instructions

Provide the exact commands another laptop will need:

### 1. Clone the repository
```bash
git clone https://github.com/srinath-m29/cn-ml-traffic-classification.git
cd cn-ml-traffic-classification
```

### 2. Set up the Python Backend
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Set up the React Frontend
```powershell
cd frontend
npm install
```

### 4. Start the Application

#### Terminal 1 — Start the FastAPI Backend:
From the project root:
```powershell
.\.venv\Scripts\Activate.ps1
uvicorn backend.services.main:app --host 127.0.0.1 --port 8000 --reload
```
The backend will be live at `http://127.0.0.1:8000` (API documentation at `http://127.0.0.1:8000/docs`).

#### Terminal 2 — Start the Frontend Development Server:
From the `frontend` directory:
```powershell
cd frontend
npm run dev
```
The React dashboard will be live at `http://localhost:5173`.
