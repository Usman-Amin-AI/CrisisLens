# Deployment Guide - CrisisLens

This document describes simple deployment options for the CrisisLens project: Docker Compose for local demo and Kubernetes manifests for cluster deployment.

Prerequisites
- Docker & Docker Compose (for local)
- kubectl and a Kubernetes cluster (for k8s manifests)
- Python 3.10+ (for local development)

Quick local demo (docker-compose)
1. Copy `.env.example` to `.env` and edit values as needed.
2. Build and start services:

```bash
make build
make up
```

3. Check logs:

```bash
make logs
```

4. Open the Streamlit dashboard at http://localhost:8501 and API at http://localhost:8000/docs

Kubernetes (cluster)
1. Ensure `kubectl` is configured to use your cluster.
2. Apply configmap and postgres PVC/deployment:

```bash
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/postgres-deployment.yaml
```

3. Deploy API and Streamlit:

```bash
kubectl apply -f k8s/api-deployment.yaml
kubectl apply -f k8s/streamlit-deployment.yaml
```

Notes
- For production use, replace DUMMY credentials in the `.env` or `ConfigMap` with real secrets (use Kubernetes Secrets or a secrets manager).
- Tune resource requests/limits in the k8s manifests for your cluster.
- Consider using an image registry (Docker Hub, ECR, GCR) and update image references in k8s manifests.

Troubleshooting
- If the database service fails, inspect `kubectl describe pod` and `kubectl logs` for details.
- Make sure Postgres PVC bindings have sufficient storage class available.

Contact
- For questions, open an issue in the project repository.
