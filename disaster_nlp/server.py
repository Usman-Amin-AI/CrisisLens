"""FastAPI server exposing prediction endpoints with explanations and drift monitoring."""
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional, Any, Dict
import logging
import os
import joblib

from disaster_nlp.drift import DriftMonitor
from disaster_nlp.explain import PredictionExplainer, explain_prediction
from disaster_nlp import db as storage
from disaster_nlp import alerting
import time

app = FastAPI(title="CrisisLens Prediction Service")

# Setup logger
LOG_DIR = os.environ.get('CRISISLOG_DIR', 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
logger = logging.getLogger('crisislens')
logger.setLevel(logging.INFO)
fh = logging.FileHandler(os.path.join(LOG_DIR, 'api.log'))
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

# Drift monitor instance (global)
DRIFT = DriftMonitor()

# Initialize DB
try:
    storage.init_db()
except Exception:
    logger.exception('DB init failed')

# Alert cooldown (seconds)
LAST_ALERT_TS = 0
ALERT_COOLDOWN = int(os.environ.get('ALERT_COOLDOWN', '3600'))


class PredictRequest(BaseModel):
    text: str
    model_type: Optional[str] = 'classical'
    model_path: Optional[str] = None


class BatchPredictRequest(BaseModel):
    texts: List[str]
    model_type: Optional[str] = 'classical'
    model_path: Optional[str] = None


def load_model(path: Optional[str], model_type: str) -> Any:
    """Attempt to load a model from path. If path is None, raise an informative error."""
    if path is None:
        raise ValueError("Model path must be provided via 'model_path' or set MODEL_PATH env var.")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    # Try joblib for sklearn pipelines
    try:
        model = joblib.load(path)
        return model
    except Exception:
        # Fallback: try to let model implement its own load API
        # For transformer models we expect a directory with saved HF files; return path
        return path


@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request: {request.method} {request.url}")
    try:
        body = await request.json()
    except Exception:
        body = None
    logger.info(f"Body: {str(body)}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response


@app.post('/predict')
async def predict(req: PredictRequest):
    try:
        model_path = req.model_path or os.environ.get('MODEL_PATH')
        model = load_model(model_path, req.model_type)
    except Exception as e:
        logger.exception('Model loading failed')
        raise HTTPException(status_code=500, detail=str(e))

    # Compute probability and label
    try:
        # If model is a path (transformer dir), caller should load transformer class
        if isinstance(model, str):
            # attempt to use TransformerClassifier load if available
            from disaster_nlp.model import TransformerClassifier
            tfm = TransformerClassifier()
            tfm.load(model)
            model = tfm

        probs = None
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba([req.text])
            if probs.ndim > 1:
                prob = float(probs[0][1])
            else:
                prob = float(probs[0])
        else:
            # try predict_proba on pipeline
            if hasattr(model, 'pipeline') and hasattr(model.pipeline, 'predict_proba'):
                prob = float(model.pipeline.predict_proba([req.text])[0, 1])
            else:
                raise ValueError('Model does not expose predict_proba')

        label = 'disaster' if prob >= 0.5 else 'non-disaster'

        # Explanation
        try:
            expl = PredictionExplainer()
            explanation = expl.explain_text(model, req.text, model_type='transformer' if req.model_type=='transformer' else 'classical')
        except Exception as ex:
            logger.warning(f'Explanation failed: {ex}')
            explanation = {'error': str(ex)}

        # Persist post
        try:
            storage.save_post(req.text, label, prob, explanation, raw_response=resp if 'resp' in locals() else None)
        except Exception:
            logger.exception('Failed to persist post')

        # Update drift monitor
        DRIFT.add_batch([1 if label=='disaster' else 0], [prob])

        drift_status = DRIFT.check_drift()

        # Persist drift metric
        try:
            storage.save_drift_metric(drift_status.get('disaster_rate'), drift_status.get('avg_confidence'), drift_status.get('n_samples'), drift_status.get('reasons'))
        except Exception:
            logger.exception('Failed to persist drift metric')

        # Alerting (simple cooldown)
        global LAST_ALERT_TS
        if drift_status.get('drift_detected'):
            now = time.time()
            if now - LAST_ALERT_TS > ALERT_COOLDOWN:
                LAST_ALERT_TS = now
                subject = 'CrisisLens Drift Alert'
                body = f"Drift detected: {drift_status}"
                alerting.send_email_alert(subject, body)
                alerting.send_slack_alert(body)

        resp = {
            'label': label,
            'confidence': prob,
            'explanation': explanation,
            'drift': drift_status
        }

        return resp

    except Exception as e:
        logger.exception('Prediction failed')
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/batch_predict')
async def batch_predict(req: BatchPredictRequest):
    try:
        model_path = req.model_path or os.environ.get('MODEL_PATH')
        model = load_model(model_path, req.model_type)
    except Exception as e:
        logger.exception('Model loading failed')
        raise HTTPException(status_code=500, detail=str(e))

    results = []
    labels = []
    confs = []

    for text in req.texts:
        try:
            # ensure model loaded as object if path
            if isinstance(model, str):
                from disaster_nlp.model import TransformerClassifier
                tfm = TransformerClassifier()
                tfm.load(model)
                _model = tfm
            else:
                _model = model

            if hasattr(_model, 'predict_proba'):
                probs = _model.predict_proba([text])
                prob = float(probs[0][1]) if probs.ndim > 1 else float(probs[0])
            else:
                if hasattr(_model, 'pipeline') and hasattr(_model.pipeline, 'predict_proba'):
                    prob = float(_model.pipeline.predict_proba([text])[0,1])
                else:
                    raise ValueError('Model does not expose predict_proba')

            label = 'disaster' if prob >= 0.5 else 'non-disaster'
            labels.append(1 if label=='disaster' else 0)
            confs.append(prob)

            # Explanation (best-effort)
            try:
                expl = PredictionExplainer()
                explanation = expl.explain_text(_model, text, model_type='transformer' if req.model_type=='transformer' else 'classical')
            except Exception as ex:
                explanation = {'error': str(ex)}

            results.append({'text': text, 'label': label, 'confidence': prob, 'explanation': explanation})

        except Exception as e:
            logger.exception('Single prediction failed')
            results.append({'text': text, 'error': str(e)})

    # Update drift monitor with entire batch
    DRIFT.add_batch(labels, confs)
    drift_status = DRIFT.check_drift()

    return {'results': results, 'drift': drift_status}
