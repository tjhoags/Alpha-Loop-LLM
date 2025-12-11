import os
import requests
import streamlit as st

API_URL = os.getenv("DEEP_PULSE_API", "http://localhost:8001")

st.set_page_config(page_title="DeepPulse", page_icon="📈", layout="wide")
st.title("DeepPulse")
st.caption("Real-time sentiment pulse for any topic (MVP heuristic).")

topic = st.text_input("Topic / Ticker / Phrase", value="NVDA earnings strong beat")

if st.button("Analyze"):
    try:
        resp = requests.get(f"{API_URL}/sentiment", params={"topic": topic}, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        st.success(f"Score: {data['scores']['score']:.3f}")
        st.json(data)
    except Exception as exc:  # pragma: no cover
        st.error(f"Failed to analyze: {exc}")


