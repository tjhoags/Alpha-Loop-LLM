import os
import requests
import streamlit as st

API_URL = os.getenv("PROMPT_FORGE_API", "http://localhost:8000")

st.set_page_config(page_title="Prompt Forge", page_icon="💡", layout="wide")
st.title("Prompt Forge")
st.caption("Buy & sell optimized prompts. Minimal MVP.")


def fetch_prompts():
    try:
        resp = requests.get(f"{API_URL}/prompts", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return []


prompts = fetch_prompts()

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Marketplace")
    if prompts:
        for p in prompts:
            st.markdown(f"**{p['title']}** - ${p['price']:.2f}  \n{p['content']}")
    else:
        st.info("No prompts yet. Add one on the right.")

with col2:
    st.subheader("List a prompt")
    title = st.text_input("Title")
    content = st.text_area("Content")
    price = st.number_input("Price", min_value=0.0, value=0.0, step=0.5)
    author = st.text_input("Author", value="anon")

    if st.button("Publish"):
        payload = {
            "id": title.lower().replace(" ", "-")[:32] or "prompt",
            "title": title or "Untitled",
            "content": content or "Example prompt",
            "price": price,
            "author": author,
        }
        try:
            resp = requests.post(f"{API_URL}/prompts", json=payload, timeout=5)
            if resp.status_code == 200:
                st.success("Published! Refresh to see it live.")
            else:
                st.error(resp.text)
        except Exception as exc:  # pragma: no cover
            st.error(f"Failed to publish: {exc}")


