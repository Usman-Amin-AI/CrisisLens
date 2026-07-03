"""Streamlit proof-of-concept dashboard for real-time inference and drift monitoring.

This app simulates a streaming social feed (or can be connected to a real source)
and forwards posts to the Phase 5 inference API (`/predict`). It displays a rolling
map/list of detected disaster posts, confidence scores, explanation highlights,
and drift-monitoring status returned by the server.

Run with:
    streamlit run deployment/streamlit_dashboard.py

Important: set `PREDICTOR_URL` env var or enter the URL in the sidebar (default http://localhost:8000/predict)
"""
import streamlit as st
import threading
import time
import requests
import random
from collections import deque
import pandas as pd
import datetime
import os


DEFAULT_SERVER = os.environ.get('PREDICTOR_URL', 'http://localhost:8000/predict')


SAMPLE_TWEETS = [
    ("Massive earthquake hits downtown, buildings collapsed", 34.0522, -118.2437),
    ("Wildfire approaching homes, evacuate now", 37.7749, -122.4194),
    ("Flooding reported after heavy rains, stay safe", 29.7604, -95.3698),
    ("Traffic jam today, sunny day", 40.7128, -74.0060),
    ("Power outage in several neighborhoods", 41.8781, -87.6298),
    ("Great concert last night!", 51.5074, -0.1278),
    ("Tornado warning issued for counties nearby", 35.4676, -97.5164),
    ("Small earthquake felt, no damage reported", 34.0522, -118.2437),
]


def init_state():
    if 'feed' not in st.session_state:
        st.session_state['feed'] = deque(maxlen=500)
    if 'running' not in st.session_state:
        st.session_state['running'] = False
    if 'server' not in st.session_state:
        st.session_state['server'] = DEFAULT_SERVER


def simulator_loop(server_url: str, rate: float = 2.0):
    """Simple loop that posts simulated tweets to the server and stores results in session_state."""
    while st.session_state['running']:
        text, lat, lon = random.choice(SAMPLE_TWEETS)
        payload = {'text': text, 'model_type': 'classical'}
        try:
            r = requests.post(server_url, json=payload, timeout=10)
            if r.status_code == 200:
                data = r.json()
                item = {
                    'timestamp': datetime.datetime.utcnow().isoformat() + 'Z',
                    'text': text,
                    'lat': lat,
                    'lon': lon,
                    'label': data.get('label'),
                    'confidence': data.get('confidence'),
                    'explanation_html': data.get('explanation', {}).get('html') if isinstance(data.get('explanation'), dict) else None,
                    'drift': data.get('drift')
                }
            else:
                item = {'timestamp': datetime.datetime.utcnow().isoformat() + 'Z', 'text': text, 'error': r.text}
        except Exception as e:
            item = {'timestamp': datetime.datetime.utcnow().isoformat() + 'Z', 'text': text, 'error': str(e)}

        st.session_state['feed'].appendleft(item)
        time.sleep(1.0 / max(0.1, rate))


def start_simulator(server_url: str, rate: float):
    if st.session_state['running']:
        return
    st.session_state['running'] = True
    thread = threading.Thread(target=simulator_loop, args=(server_url, rate), daemon=True)
    thread.start()


def stop_simulator():
    st.session_state['running'] = False


def main():
    st.set_page_config(page_title='CrisisLens Live Dashboard', layout='wide')
    init_state()

    st.sidebar.title('Configuration')
    server_url = st.sidebar.text_input('Predictor URL', value=st.session_state['server'])
    rate = st.sidebar.slider('Events per second', 0.1, 5.0, 1.0)
    start = st.sidebar.button('Start Simulation')
    stop = st.sidebar.button('Stop Simulation')

    if start:
        st.session_state['server'] = server_url
        start_simulator(server_url, rate)
    if stop:
        stop_simulator()

    st.title('CrisisLens - Real-time Triage Dashboard (PoC)')

    # Top metrics
    cols = st.columns(3)
    with cols[0]:
        total = len(st.session_state['feed'])
        st.metric('Recent posts', total)
    with cols[1]:
        disasters = sum(1 for it in st.session_state['feed'] if it.get('label') == 'disaster')
        rate_disp = f"{(disasters / total * 100):.1f}%" if total>0 else 'N/A'
        st.metric('Predicted disaster rate', rate_disp)
    with cols[2]:
        confidences = [it.get('confidence', 0) for it in st.session_state['feed'] if it.get('confidence') is not None]
        avg_conf = (sum(confidences)/len(confidences)) if confidences else 0
        st.metric('Avg confidence', f"{avg_conf:.3f}")

    # Map and list
    left, right = st.columns([1,1])
    with left:
        st.subheader('Map of recent posts')
        df_map = pd.DataFrame([{'lat': it['lat'], 'lon': it['lon'], 'label': it.get('label','unknown')} for it in st.session_state['feed'] if it.get('lat') is not None])
        if not df_map.empty:
            st.map(df_map.rename(columns={'lat':'lat','lon':'lon'}))
        else:
            st.info('No geolocated posts yet')

    with right:
        st.subheader('Recent detected disaster posts')
        for it in list(st.session_state['feed'])[:50]:
            if it.get('label') == 'disaster':
                st.markdown(f"**{it.get('timestamp')} - {it.get('label').upper()} ({it.get('confidence'):.2f})**")
                st.write(it.get('text'))
                if it.get('explanation_html'):
                    st.components.v1.html(it.get('explanation_html'), height=80)
                if it.get('drift'):
                    st.write('Drift:', it.get('drift'))

    st.subheader('Full recent feed (most recent first)')
    st.write(pd.DataFrame(list(st.session_state['feed'])))


if __name__ == '__main__':
    main()
