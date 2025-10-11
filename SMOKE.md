# Smoke Test
1) Put at least 1-2 PDFs into ./corpus
2) Start API: uvicorn src.api.server:app --reload
3) POST /rebuild_index  (or click in Streamlit)
4) Start UI: streamlit run src/ui/app.py
5) Ask: "Who is the author?" → see evidence + citation audit.
