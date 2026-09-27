# SatQuery: Earth Observation & Geospatial Intelligence Platform

SatQuery is a monorepo architecture for satellite imagery question answering, object grounding, bi-temporal change detection, and multi-sensor SAR/optical fusion.

## Repository Layout

```
satquery/
├── docker-compose.yml       # Orchestrates frontend and backend containers
├── backend/                 # FastAPI (Python 3.11+) Backend Service
│   ├── Dockerfile
│   ├── main.py              # Exposes GET /health, /agents/status, API router
│   ├── requirements.txt
│   ├── agents/              # Router, Planner, Registry
│   ├── models/              # VQA, Grounding, Change Detection, Sensor Fusion
│   ├── geospatial/          # Validator, Metadata, Preprocessing
│   ├── evidence/            # Confidence Scorer, Evidence Verifier
│   └── tests/               # Pytest suite
└── frontend/                # Next.js (TypeScript) Web Dashboard
    ├── Dockerfile
    ├── src/
    │   └── app/             # App Router, UI Components, Styling
    └── package.json
```

## Quickstart with Docker Compose

Run both the FastAPI backend and Next.js frontend together:

```bash
cd satquery
docker compose up --build
```

- **Frontend Dashboard**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **Interactive OpenAPI Docs**: http://localhost:8000/docs
- **Health Check Endpoint**: http://localhost:8000/health

## Running Locally

### Backend
```bash
cd satquery/backend
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Run tests:
```bash
pytest tests/ -v
```

### Frontend
```bash
cd satquery/frontend
npm install
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) in your browser.
