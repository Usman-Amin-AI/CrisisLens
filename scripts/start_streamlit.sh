#!/usr/bin/env bash
# Start the Streamlit dashboard (development)
set -e
streamlit run deployment/streamlit_dashboard.py --server.port 8501 --server.address 0.0.0.0
