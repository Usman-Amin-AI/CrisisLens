web: uvicorn disaster_nlp.server:app --host 0.0.0.0 --port ${PORT:-8000}
streamlit: streamlit run deployment/streamlit_dashboard.py --server.port ${STREAMLIT_PORT:-8501}
