# FinRAG Advanced

**An AI-powered Financial Report Analysis and Question Answering System using Retrieval-Augmented Generation (RAG).**

FinRAG Advanced is a financial document intelligence system designed to extract, retrieve, analyze, and compare information from company annual reports and SEC filings.

The system combines **Docling-based document processing, hybrid retrieval (FAISS + BM25), Reciprocal Rank Fusion (RRF), Cross-Encoder reranking, and locally hosted Hugging Face LLMs** to generate context-grounded financial insights.

---

## 1. Key Features

- **Financial Document Processing:** Extracts structured text and tables from financial PDF reports using Docling.
- **Hybrid Retrieval:** Combines dense vector search using FAISS with sparse keyword retrieval using BM25.
- **Reciprocal Rank Fusion (RRF):** Combines rankings from multiple retrieval methods.
- **Cross-Encoder Reranking:** Re-ranks retrieved chunks based on query-document relevance.
- **Multi-Company Analysis:** Supports financial comparisons between multiple companies.
- **Table-Aware Extraction:** Retrieves relevant financial statement tables for numerical analysis.
- **Financial Metrics:** Calculates additional indicators, including:
  - Free Cash Flow (FCF)
  - Interest Coverage Ratio
  - Altman Z-Score
  - Return on Equity (ROE)
  - Operating and Net Profit Margins
- **Financial Forecasting:** Generates simple trend-based financial forecasts.
- **Peer Benchmarking:** Compares financial metrics across selected companies.
- **Risk Analysis:** Summarizes relevant risk factors from financial reports.
- **Financial Verification:** Supports cross-checking selected financial values using Yahoo Finance.
- **Hallucination Guard:** Validates generated numerical claims against extracted financial data.
- **Conversational Memory:** Maintains context across follow-up questions.
- **PDF Report Export:** Generates downloadable financial analysis reports.
- **Evaluation Framework:** Supports retrieval and extraction evaluation.

---

## 2. System Architecture

FinRAG follows a hybrid Retrieval-Augmented Generation architecture.

```text
             Financial PDF Reports
                       |
                       v
             Document Processing
                   (Docling)
                       |
                       v
              Text & Table Extraction
                       |
                       v
                 Chunking
                       |
             +---------+---------+
             |                   |
             v                   v
        Dense Retrieval     Sparse Retrieval
             |                   |
          FAISS                 BM25
             |                   |
             +---------+---------+
                       |
                       v
          Reciprocal Rank Fusion (RRF)
                       |
                       v
             Cross-Encoder Reranking
                       |
                       v
             Relevant Context Chunks
                       |
                       v
               Local Hugging Face LLM
                       |
                       v
           Grounded Financial Response
                       |
             +---------+----------+
             |                    |
             v                    v
       Financial Analysis    Verification
             |                    |
             +---------+----------+
                       |
                       v
             Final Financial Report
```

### Retrieval Components

| Component | Technology |
|---|---|
| Document processing | Docling |
| Vector database / index | FAISS |
| Embedding model | BAAI/bge-small-en-v1.5 |
| Sparse retrieval | BM25 |
| Rank fusion | Reciprocal Rank Fusion (RRF) |
| Reranking | Cross-Encoder |
| Language model | Hugging Face Transformers |

---

## 3. Technology Stack

- **Programming Language:** Python
- **Frontend:** Streamlit
- **Document Processing:** Docling
- **Vector Search:** FAISS
- **Keyword Search:** BM25
- **Embeddings:** Sentence Transformers
- **Reranking:** Cross-Encoder
- **LLM Framework:** Hugging Face Transformers
- **Data Processing:** Pandas, NumPy
- **Financial Data Verification:** yfinance
- **Visualization:** Altair

---

## 4. Project Setup

### Prerequisites

- Python 3.10 or compatible version supported by the dependencies.
- pip package manager.
- Git.
- Sufficient RAM and storage for local model execution.
- CUDA-compatible GPU for GPU-accelerated execution, if supported by the installed PyTorch configuration.

### Step 1: Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd fin_rag_advanced
```

### Step 2: Create a Virtual Environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Ensure that the appropriate PyTorch version is installed for your CPU or CUDA environment.

---

## 5. Add Financial Documents

Place the financial PDF reports inside:

```text
data/
└── pdfs/
    ├── Apple_2025_10-K.pdf
    ├── Microsoft_2025_10-K.pdf
    ├── Tesla_2025_10-K.pdf
    └── Infosys_2025_20-F.pdf
```

Use the following naming convention:

```text
Company_Year_ReportType.pdf
```

For example:

```text
Apple_2025_10-K.pdf
```

Add the corresponding company ticker in `config.py` if Yahoo Finance verification is required.

---

## 6. Running the Project

### Step 1: Process and Index Documents

Run the ingestion pipeline:

```bash
python ingest.py
```

This processes the financial reports, extracts relevant content, and builds the retrieval index.

### Step 2: Run Financial Analysis from the Command Line

Example:

```bash
python pipeline.py "Compare Apple and Microsoft over last 3 years"
```

Other example queries:

```text
Give financial report of Apple in last 3 years

Compare Tesla and Microsoft over last 3 years

What is the profitability trend of Infosys?

Compare the balance sheet strength of Apple and Microsoft

Analyze the risk factors of Tesla
```

### Step 3: Launch the Streamlit Application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal to access the FinRAG interface.

---

## 7. Evaluation

FinRAG includes an evaluation module to assess retrieval and financial information extraction.

### Build the Evaluation Dataset

```bash
python evaluate.py --build
```

### Run Evaluation

```bash
python evaluate.py
```

### Run LLM-Based Evaluation

```bash
python evaluate.py --llm
```

The evaluation framework supports comparison of retrieval configurations and extraction approaches.

---

## 8. Financial Analysis Capabilities

FinRAG supports analysis of several financial indicators:

| Metric | Description |
|---|---|
| Revenue | Total income generated from business operations |
| Net Income | Profit after expenses and taxes |
| EBITDA | Earnings before interest, taxes, depreciation and amortization |
| Operating Margin | Operating income relative to revenue |
| Net Profit Margin | Net income relative to revenue |
| ROE | Return on shareholders' equity |
| Free Cash Flow | Operating cash flow minus capital expenditure |
| Interest Coverage | Ability to meet interest obligations |
| Altman Z-Score | Financial distress risk indicator |

Financial calculations are performed using extracted financial data and the project's analysis pipeline.

---

## 9. Repository Structure

```text
fin_rag_advanced/
│
├── app.py                 # Streamlit application
├── pipeline.py            # Main RAG and analysis pipeline
├── ingest.py              # Document ingestion and indexing
├── retriever.py           # Hybrid retrieval and reranking
├── table_parser.py        # Financial table processing
├── config.py              # Project configuration
├── llm.py                 # Local LLM integration
├── metrics.py             # Financial metric calculations
├── analysis.py            # Financial and risk analysis
├── verify.py              # Financial value verification
├── evaluate.py            # Evaluation framework
├── export.py               # PDF report generation
├── seed_cache.py           # Cache initialization
│
├── data/
│   ├── pdfs/               # Source financial reports
│   └── index/              # Generated retrieval index
│
├── eval/                   # Evaluation data and results
│
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## 10. Important Notes

- The retrieval index must be generated before running the application.
- Adding new financial reports requires running the ingestion pipeline again.
- Yahoo Finance verification requires an internet connection and supported ticker mappings.
- Local LLM performance depends on available system memory and hardware.
- Financial forecasts are illustrative trend-based estimates and should not be treated as investment recommendations.
- Extracted financial values should be verified against the original company filings before making important decisions.

---

## 11. Future Improvements

- Improve financial table extraction for complex PDF layouts.
- Expand evaluation across additional companies and fiscal years.
- Improve numerical consistency and multi-step financial reasoning.
- Add support for additional financial document formats.
- Optimize retrieval latency and model inference.
- Extend the system with more advanced financial analysis capabilities.

---

## 12. Disclaimer

FinRAG is an academic and experimental financial document analysis project. Its outputs are intended for research and informational purposes only and do not constitute financial, investment, or legal advice.

Always verify important financial information against the original company filings and authoritative sources.

---

**FinRAG Advanced — Bringing Retrieval-Augmented Generation to Financial Intelligence.**
