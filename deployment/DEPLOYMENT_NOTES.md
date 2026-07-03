Deployment Notes

Environment variables
- MODEL_PATH: path to saved model (sklearn joblib or transformer dir)
- DB_URL: database URL (default: sqlite:///crisislens.db)
- ALERT_SMTP_HOST, ALERT_SMTP_PORT, ALERT_FROM, ALERT_TO: SMTP settings for email alerts
- ALERT_SMTP_USER, ALERT_SMTP_PASS: optional SMTP credentials
- ALERT_SLACK_WEBHOOK: Slack incoming webhook URL for alerts
- TWITTER_BEARER_TOKEN: Bearer token for Twitter/X API v2 streaming
- PREDICTOR_URL: URL of running predictor API (default http://localhost:8000/predict)
- ALERT_COOLDOWN: seconds between repeated alerts (default 3600)

Example dummy env file (.env.example)
```
# Use DUMMY values to avoid sending real credentials during testing
MODEL_PATH=/path/to/model
DB_URL=sqlite:///crisislens.db
ALERT_SMTP_HOST=DUMMY
ALERT_SMTP_PORT=25
ALERT_FROM=DUMMY
ALERT_TO=DUMMY
ALERT_SMTP_USER=DUMMY
ALERT_SMTP_PASS=DUMMY
ALERT_SLACK_WEBHOOK=DUMMY
TWITTER_BEARER_TOKEN=DUMMY
PREDICTOR_URL=http://localhost:8000/predict
ALERT_COOLDOWN=3600
```

Files added
- `disaster_nlp/db.py` - SQLAlchemy persistence of posts and drift metrics
- `disaster_nlp/alerting.py` - Email/Slack alert helpers
- `disaster_nlp/server.py` - FastAPI service (updated to persist and alert)
- `deployment/streamlit_dashboard.py` - Streamlit PoC dashboard (simulated stream)
- `deployment/twitter_streamer.py` - Twitter streaming client (env-driven)

Running
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the API server:
   ```bash
   uvicorn disaster_nlp.server:app --host 0.0.0.0 --port 8000
   ```
3. Start the dashboard (optional):
   ```bash
   streamlit run deployment/streamlit_dashboard.py
   ```
4. (Optional) Run the Twitter streamer once env vars are set:
   ```bash
   python deployment/twitter_streamer.py
   ```

Security and privacy
- Store credentials in secure secret storage; do not commit them to Git.
- The demo persists incoming text to local DB; review retention policies before deploying.

Scaling notes
- For production, use a robust message queue (Kafka/RabbitMQ) and async workers for inference.
- Use managed DB (Postgres) rather than SQLite for concurrent writes.
- Harden alerting with retry/backoff and rate-limiting.
