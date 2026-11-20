# fin_rag_advanced
Financial report RAG: Docling + FAISS/BM25 hybrid search + cross-encoder rerank + local HF LLM.

## Setup
1. python -m venv venv ; venv\Scripts\activate
2. pip install -r requirements.txt
3. Put 10-K PDFs in data/pdfs/ named Company_Year_10-K.pdf (e.g. Apple_2025_10-K.pdf)
4. Add new companies to TICKERS in config.py (for yfinance verification)

## Run
- python ingest.py
- python pipeline.py "Compare Apple and Microsoft over last 3 years"
- streamlit run app.py

## Evaluation (for the paper)
- python evaluate.py --build
- python evaluate.py
- python evaluate.py --llm

## Features
Multi-company comparison, table-aware extraction, hybrid search + rerank, extra ratios
(FCF, interest coverage, Altman Z), forecast, peer benchmark, risk summary, yfinance verification,
hallucination guard, chat memory, PDF export, evaluation module.
