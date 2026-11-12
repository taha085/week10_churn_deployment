# Week 10 — Streamlit UI, Docker & Deployment

A two-service churn prediction demo. Streamlit calls the Week 9 FastAPI `/predict` endpoint over HTTP. The backend loads the frozen Week 7 pipeline; the UI never loads the model.

## Project structure

```text
week10_churn_deployment/
├── backend/
│   ├── main.py
│   ├── models/churn_pipeline.joblib
│   └── requirements.txt
├── frontend/
│   ├── app.py
│   └── requirements.txt
├── logs/
│   └── predictions.log       # generated on first successful prediction
├── Dockerfile                # FastAPI backend
├── Dockerfile.frontend       # Streamlit UI
├── docker-compose.yml
└── MONITORING_NOTES.md
```

## Run locally without Docker

Terminal 1 (backend):
```bash
cd backend
python -m pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Terminal 2 (frontend):
```bash
cd frontend
python -m pip install -r requirements.txt
# Windows PowerShell:
$env:API_URL="http://localhost:8000"
# macOS/Linux: export API_URL=http://localhost:8000
streamlit run app.py
```
Open `http://localhost:8501`.

## Run both services with Docker

Install Docker Desktop, then run from the repository root:
```bash
docker compose up --build
```
- Streamlit: `http://localhost:8501`
- FastAPI docs: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`

The Compose network routes Streamlit to `http://api:8000`. The host `logs/` directory is mounted into the backend container.

## Logging

Each successful prediction writes one JSONL event to `logs/predictions.log`: UTC timestamp, a limited feature summary, probability, and predicted label. No customer ID is accepted or stored.

## Public deployment

A public live URL requires the owner's hosting account and deployment authorization; it cannot be created just by generating these files. For Hugging Face Spaces, create a Docker Space for the backend and another Streamlit Space for the frontend. Upload the backend files (`Dockerfile`, `backend/main.py`, `backend/models/`, and backend requirements) to the backend Space. Set the frontend `API_URL` secret/variable to the backend Space's reachable API URL, then upload `frontend/app.py` and frontend requirements to a Streamlit Space. Confirm `/health`, `/docs`, and a sample prediction before sharing the frontend Space URL.

Render can also host each service separately, but free-service sleep and ephemeral filesystem behavior may affect demo availability and log persistence. For durable logs, attach persistent storage or send logs to a managed logging service.

## Important deployment and privacy notes

- Do not commit credentials, API keys, or private customer records.
- Use synthetic/demo inputs for a public demonstration.
- `joblib` artifacts must come from a trusted source; unpickling untrusted artifacts can execute code.
- This is a demo, not a fully monitored production service. See `MONITORING_NOTES.md`.


## Verification

A local FastAPI TestClient smoke test was run against the included frozen pipeline:
- `GET /health` returned HTTP 200.
- The sample `POST /predict` returned probability `0.6626099944`, label `Yes`, and threshold `0.3251224160`.
- A JSON Lines sample is included at `logs/predictions.log`.
- Automated tests are in `tests/test_api.py`; run with `pytest -q` after installing backend requirements plus `pytest` and `httpx`.
