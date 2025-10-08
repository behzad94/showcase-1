# src/ui/app.py
# goal: a tiny UI to query API and show answer + evidences + audits

import os, json, requests, streamlit as st

API = os.getenv("RAG_API", "http://127.0.0.1:8000")

st.set_page_config(page_title="RAG Showcase", page_icon="📚", layout="wide")

with st.sidebar:
    k = st.slider("Top-K results", 1, 5, 2, 1)
    use_sum = st.checkbox("Use LLM summarization (Ollama)", value=False)
    if st.button("Build/Rebuild Index"):
        try:
            r = requests.post(f"{API}/rebuild_index", json={"corpus_path": "corpus"})
            st.success(r.json())
            st.cache_data.clear()
            st.rerun()
        except Exception as e:
            st.error(str(e))

st.title("RAG Showcase")

st.markdown("""
🛡️ **Transparency & GDPR Notice**
- Answers come **only** from local documents you uploaded.
- No hidden web calls.
- May be incomplete or inaccurate.
- **GDPR Article 22:** Do not use this for impactful automated decisions without human review.
""")

q = st.text_input("❓ Ask a question about the documents", placeholder="Who wrote the paper?")
if st.button("Search"):
    payload = {"question": q, "top_k": k, "summarize": use_sum}
    try:
        r = requests.post(f"{API}/ask", json=payload)
        if r.status_code != 200:
            st.error(f"Failed: {r.text}")
        else:
            ans = r.json()

            if ans.get("needs_clarification"):
                st.warning(ans.get("clarify_hint", "Low confidence. Please clarify your question."))

            ca = ans.get("citation_audit", {})
            if ca.get("warning"):
                st.info(f"Citation coverage low (coverage={ca.get('coverage')}, threshold={ca.get('threshold')}).")

            st.subheader("📖 Answer")
            st.write(ans.get("text", ""))
            st.caption(f"LLM: {os.getenv('OLLAMA_MODEL','phi3:latest')} via Ollama")

            evs = ans.get("evidences", [])
            if evs:
                with st.expander("🔎 Evidence"):
                    for ev in evs:
                        ps, pe = ev.get("page_start"), ev.get("page_end")
                        page_info = (f", p.{ps}" if ps == pe else f", p.{ps}–{pe}") if ps and pe else ""
                        st.markdown(f"**{ev['file']}** (score={ev['score']:.3f}{page_info})")
                        st.write(ev.get("snippet", ""))
                        st.divider()
    except Exception as e:
        st.error(str(e))

