# Network Traffic Classification and Intrusion Detection

A local web application for classifying CIC-IDS2017 network flows. The React dashboard sends validated flow features to a FastAPI service, which serves the checked-in XGBoost multiclass model.

## What is included

- **Frontend:** React, TypeScript, and Vite dashboard for CSV validation, batch predictions, charts, and single-flow examples.
- **Backend:** FastAPI endpoints for health, model metadata, single-flow predictions, and batch predictions.
- **Inference artifacts:** the multiclass model, label mapping, and selected feature schema in `models/`.
- **Training and dataset analysis:** scripts and data are present in the repository, but training is separate from running the frontend/backend application.

The inference API uses the existing model; starting the app does not train a model or read the training datasets.

## Requirements

- Python 3.12 recommended
- Node.js and npm
- The repository's tracked `models/xgboost_multiclass_gpu.json` and `models/multiclass_label_mapping.json` files

## Start the backend

From the repository root in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn backend.services.main:app --host 127.0.0.1 --port 8000
```

The API is available at `http://127.0.0.1:8000`; interactive API documentation is at `http://127.0.0.1:8000/docs`.

## Start the frontend

In a second PowerShell terminal:

```powershell
cd frontend
npm install
Copy-Item .env.example .env.local
npm run dev
```

Open the Vite URL shown in the terminal (normally `http://localhost:5173`). The frontend uses `VITE_API_BASE_URL` from `frontend/.env.local`; when unset, it uses `http://127.0.0.1:8000`.

## Configuration

The backend allows the local Vite origins by default. For another frontend host, set `CORS_ORIGINS` in the backend environment to a comma-separated list of full origins, for example:

```text
CORS_ORIGINS=https://your-frontend.example,https://www.your-frontend.example
```

Use the frontend host's origin only (scheme and host, with an optional port; no path). `CORS_ORIGINS` is read from the process environment. `.env.example` documents the local default; the application does not load `.env` files automatically.

## API endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Reports API and model readiness |
| `GET` | `/model-info` | Returns the expected feature names and class labels |
| `POST` | `/predict` | Classifies one flow |
| `POST` | `/predict/batch` | Classifies a list of flows |

CSV uploads are validated in the browser against the feature schema returned by `/model-info`. Each batch is limited to 10,000 rows and 25 MB; split larger files into smaller batches.

## GitHub and data files

Raw and processed training datasets are excluded from Git because of their size. The inference model and its small metadata files are tracked so the frontend/backend can serve predictions after dependencies are installed. Keep secrets in environment variables; do not commit `.env` files.
