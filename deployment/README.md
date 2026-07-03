CrisisLens Deployment README

Streamlit dashboard (PoC)

Run the FastAPI server (inference API) first, e.g.:

```bash
# from project root
uvicorn disaster_nlp.server:app --host 0.0.0.0 --port 8000
```

Then run the Streamlit dashboard (recommended inside a virtualenv):

```bash
pip install -r requirements.txt
streamlit run deployment/streamlit_dashboard.py
```

Configuration
- Use the `PREDICTOR_URL` env var or the dashboard sidebar to point to the running `/predict` endpoint.

Notes
- The dashboard simulates a social feed and posts simulated tweets to the inference API when no real Twitter credentials are provided.
- Explanations are rendered from the HTML payload returned by the server; ensure `shap` is installed and that the server was started with explanation support available.
- The dashboard displays drift status values returned by the server in prediction responses.

Docker / Compose
- For a local demo that includes Postgres, API, and Streamlit: use `docker-compose.yml` and `.env` (copy `.env.example`).
- Start with: `make build && make up` or `docker-compose up --build`.
