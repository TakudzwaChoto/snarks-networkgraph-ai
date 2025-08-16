# River Water Quality Management System

A Flask-based web application for river network management that integrates a Neo4j graph database with interactive dashboards and machine learning predictions for water-quality trading (WEC). The system includes:

- Graph-backed modeling of river topology and monitoring points
- An interactive UI (index + analytics dashboard)
- A unified prediction API with a deep model (PyTorch/Sklearn) and optional baseline models (LightGBM/XGBoost)
- Utilities to initialize/test the Neo4j database
- Optional zkSNARK proof verification endpoints

---

## Features

- Neo4j graph database integration for river topology and monitoring points
- Interactive network and analytics dashboards (Plotly, custom JS/CSS)
- Unified prediction API
  - Deep predictor (PyTorch if available, otherwise Sklearn MLP)
  - Optional baselines: LightGBM and XGBoost (install separately)
- Database utilities: initialize dataset, add sample data, health checks
- Optional zkSNARK verification endpoints (requires `snarkjs` and a verifier key)

---

## Prerequisites

- Python 3.9+
- Neo4j running locally or via Docker
- Required data files in project root (or configure paths via env vars):
  - `河流拓扑结构.xlsx` (river topology)
  - `河道氨氮统计数据--环境容量.xlsx` (ammonia nitrogen stats)
- Optional for baselines/graph features:
  - `lightgbm`, `xgboost`, `networkx`
- Optional for zk verification:
  - Node.js tool `snarkjs` available on PATH and `zk/verifier_key.json`

---

## Installation (Local)

```bash
# Clone
git clone <repository-url>
cd <repo>

# (Recommended) Create venv
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate

# Install core dependencies
pip install -r requirements.txt

# Optional extras, if you plan to use baseline models and graph features
pip install lightgbm xgboost networkx
```

Place data files at the project root or set env vars (see Configuration):
- `河流拓扑结构.xlsx`
- `河道氨氮统计数据--环境容量.xlsx`
- `train_tradedata.csv` (for ML training; included in repo)

---

## Run (Local)

```bash
# Preferred: uses configuration from config.py
python run.py

# Alternate dev mode
python app.py

# Production example (if desired)
# gunicorn -w 2 -b 0.0.0.0:5000 app:app
```

App will be available at `http://localhost:5000`.

---

## Docker & Compose

This repo includes a `Dockerfile` and `docker-compose.yml` with services for Neo4j, the Flask app, and optional Nginx.

```bash
# Build & start all services
docker compose up -d --build

# View logs
docker compose logs -f river_app

# Stop
docker compose down
```

Ports:
- App: 5000 (mapped from container)
- Neo4j: 7474 (HTTP), 7687 (Bolt)
- Nginx (optional): 80/443

Note: Compose uses defaults for data file paths. If you place data under `./data`, set env vars like `TOPOLOGY_FILE=/app/data/河流拓扑结构.xlsx` and `WATER_QUALITY_FILE=/app/data/河道氨氮统计数据--环境容量.xlsx` in `docker-compose.yml`.

---

## Configuration

This app reads configuration from environment variables (see `config.py`) and supports `.env` files.

- SECRET_KEY: Flask secret key (default: `your-secret-key-here`)
- NEO4J_URI: `bolt://localhost:7687` (Compose sets `bolt://neo4j:7687`)
- NEO4J_USER: `neo4j`
- NEO4J_PASSWORD: `12345678`
- NEO4J_DATABASE: `neo4j`
- FLASK_CONFIG: `development` | `production` | `testing` | `default` (maps to development)
- FLASK_HOST: `0.0.0.0` (bind address)
- FLASK_PORT: `5000`
- DEFAULT_ALERT_THRESHOLD: `1.0`
- TOPOLOGY_FILE: `河流拓扑结构.xlsx`
- WATER_QUALITY_FILE: `河道氨氮统计数据--环境容量.xlsx`
- TRAIN_DATA_FILE: `train_tradedata.csv`
- TEST_DATA_FILE: `test_tradedata.csv`
- ML_MODEL_PATH: `river_quality_model.pkl`
- ML_SCALER_PATH: `river_quality_scaler.pkl`
- LOG_FILE: `river_system.log`

Example `.env`:

```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=12345678
FLASK_CONFIG=development
FLASK_PORT=5000
TOPOLOGY_FILE=河流拓扑结构.xlsx
WATER_QUALITY_FILE=河道氨氮统计数据--环境容量.xlsx
```

---

## Data Files

- River topology (`河流拓扑结构.xlsx`)
  - Expected columns: `Subbasin`, `FROM_NODE`, `TO_NODE`, `FLOW_OUTcms`, `AreaC`, `Len2`, `Slo2`, `Wid2`, `Dep2`
- Water quality stats (`河道氨氮统计数据--环境容量.xlsx`)
  - Expected columns: `RCH`, `Cs`, `K`
- ML training data: `train_tradedata.csv` with `Buyer`, `Seller`, `Sold_wec`

Additional example files in repo (optional):
- `河道总氮统计数据--环境容量.xlsx`, `河道总磷统计数据--环境容量.xlsx`

---

## Initialize the Database

1) Start the app and Neo4j
2) Load data into Neo4j:
   - Open `http://localhost:5000/initialize-database`
   - Or use the main page button (Initialize Database)
3) Verify: `http://localhost:5000/test-database`

You can add demo data via `http://localhost:5000/add-test-data`.

---

## UI

- `/` main dashboard (interactive graph, actions)
- `/dashboard` analytics dashboard (charts, trends)
- `/debug-graph` debug page for visualization

---

## REST API

- GET `/health` — service and dependency status
- GET `/api/graph-data` — graph nodes and relationships (subset)
- GET `/api/water-quality-stats` — aggregate stats from monitoring points
- GET `/api/water-quality-alerts?threshold=1.0` — alerts above threshold
- GET `/initialize-database` — reads Excel files and writes to Neo4j
- GET `/add-test-data` — inserts sample nodes and relations
- GET `/test-database` — simple database counters

Prediction and training:

- POST `/api/predict` — unified prediction endpoint
  - Body (deep model, default):
    ```json
    {"buyer": 1, "seller": 2, "model": "deep"}
    ```
  - Body (baselines):
    ```json
    {"buyer": 1, "seller": 2, "model": "lgbm"}
    ```
    Train baseline first (see below). If the model does not exist or dependency is missing, an error is returned.

- POST `/api/train-ml` — train a baseline model and persist under `models/`
  - Body: `{"model":"lgbm"}` or `{"model":"xgb"}`

- POST `/api/train-deep` — train/retrain the deep predictor; persists under `models/`
- GET `/api/evaluate-deep` — evaluate deep predictor on training set
- GET `/api/feature-importance` — permutation-style importance (deep model features)

zkSNARK verification (optional):

- POST `/api/zk/verify`
  - Body: `{ "proof": {...}, "publicSignals": [...] }`
  - Requires `snarkjs` on PATH and `zk/verifier_key.json`
- GET `/api/zk/metrics` — basic verification stats and recent series

Examples:

```bash
# Health
curl -s http://localhost:5000/health | jq

# Alerts at threshold 1.2
curl -s "http://localhost:5000/api/water-quality-alerts?threshold=1.2" | jq

# Deep prediction
curl -s -X POST http://localhost:5000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"buyer": 10, "seller": 25, "model": "deep"}' | jq

# Train LightGBM (requires lightgbm, networkx if using topology features)
curl -s -X POST http://localhost:5000/api/train-ml \
  -H 'Content-Type: application/json' \
  -d '{"model":"lgbm"}' | jq
```

---

## Machine Learning

- Deep predictor (`deep_predictor.py`)
  - Uses PyTorch if available; otherwise falls back to Sklearn `MLPRegressor`
  - Artifacts: `models/deep_model.pt` or `models/mlp_sklearn.pkl`, plus `models/scaler.pkl`, `models/meta.json`
  - Trains automatically on first use if no artifacts are present
  - Features: buyer, seller, |buyer − seller|, buyer % 10, seller % 10, (buyer + seller) % 20

- Baselines (`ml_baselines.py`) — optional
  - LightGBM or XGBoost (install extras)
  - Can incorporate additional graph-based features derived from topology (requires `networkx` and presence of topology file)
  - Train via `/api/train-ml` and predict via `/api/predict` with `model` set to `lgbm` or `xgb`

- Classic fallback in `app.py`
  - A simple RandomForestRegressor is kept as a fallback for prediction if the deep model is unavailable

---

## Nginx (optional)

`nginx.conf` is provided for reverse proxying to the Flask app (used by Compose service `nginx`). Adjust SSL settings if you enable HTTPS.

---

## Troubleshooting

- Neo4j connection failed
  - Ensure Neo4j is running and accessible at `NEO4J_URI`
  - Default auth: `neo4j/12345678`
  - For Compose, the app uses `bolt://neo4j:7687`

- Missing data files
  - Confirm file names and locations
  - Override paths via `TOPOLOGY_FILE` and `WATER_QUALITY_FILE`

- Baseline training error: dependency not installed
  - Install `lightgbm`, `xgboost`, and `networkx` (for graph features)

- zk verifier unavailable
  - Install `snarkjs` and add `zk/verifier_key.json`

- Graph visualization empty
  - Ensure `/initialize-database` was run and Neo4j is populated

---

## Repository Structure (selected)

```
app.py                 # Flask app and routes
run.py                 # Startup wrapper using config.py
config.py              # Configuration with env-var defaults
ml_baselines.py        # Optional baseline models (LightGBM/XGBoost)
deep_predictor.py      # Deep predictor (Torch/Sklearn)
zk_verifier.py         # Optional zkSNARK verification helpers
graph_features.py      # Topology-based feature engineering
models/                # Saved model artifacts
static/                # CSS/JS assets
templates/             # HTML templates (index, dashboard, debug)
Dockerfile             # Container build
docker-compose.yml     # Neo4j + app (+ optional nginx)
nginx.conf             # Reverse proxy config
```
