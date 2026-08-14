# 📊 AI-Powered Data Analyst Copilot

An AI copilot that lets you upload a CSV/Excel file and analyze it through natural-language conversation — with every number grounded in real Pandas/SQL execution, never LLM guesswork.

> **Flagship project #1** in an ongoing collection of production-oriented GenAI applications.

## 🧩 Problem Statement

Analysts and business users spend significant time writing repetitive code to explore datasets, and non-technical stakeholders often can't self-serve at all. Meanwhile, naive "chat with your CSV" tools frequently let the LLM hallucinate numbers. This project solves both: a conversational interface backed by a strict **tool-calling architecture** where the AI can only report numbers that came from real, executed Pandas/SQL code.

## 🎯 Objective

Provide a Streamlit application where a user uploads a dataset and can:
- get an instant, deterministic profile and cleaning suggestions,
- ask analytical questions in plain English,
- receive answers, charts, and SQL results computed directly from their real data,
- and export a full analysis report.

## ✨ Key Features

- CSV & multi-sheet Excel upload
- Deterministic dataset profiling (shape, dtypes, missing %, stats, correlations)
- Automated, actionable data-cleaning suggestions (duplicates, missing values, outliers, whitespace, casing)
- Conversational natural-language data analysis (Groq LLM + tool calling)
- **Grounded numeric answers**: the LLM writes Pandas code or SQL, which executes in a sandbox against the real dataframe — the LLM only narrates the actual result
- EDA & statistical analysis, correlation heatmaps
- Automatic chart generation across **30 chart types** — trend, comparison, relationship, distribution, part-to-whole, hierarchical, flow, KPI, 3D, geographic, and financial charts (line, area, bar, bar_horizontal, stacked_bar, grouped_bar, polar_bar, scatter, bubble, scatter_3d, density_heatmap, contour, pie, donut, treemap, sunburst, histogram, box, violin, strip, ecdf, heatmap, funnel, waterfall, sankey, radar, gauge, bullet, candlestick, choropleth) via Plotly — charts always render on a white, dark-text background for readability, independent of the app's own theme
- **Dashboard tab**: every chart generated during the chat session is automatically collected into a Power BI-style grid dashboard
- Deterministic trend & anomaly detection (IQR-based outliers, period-over-period % change)
- SQL query generation & execution via DuckDB, directly on the in-memory dataframe
- Multi-turn conversational memory
- Downloadable Markdown analysis report

## 🏗️ Architecture

The core design principle: **the LLM never states a number from memory.**

```
┌─────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│  Streamlit  │───▶│  Orchestrator     │───▶│  Groq LLM (tool      │
│  UI (chat + │    │  (tool-calling    │    │  calling / router)   │
│  profiling) │◀───│  loop)            │◀───│                      │
└─────────────┘    └──────┬───────────┘    └─────────────────────┘
                           │ dispatches to
        ┌──────────────────┼───────────────────────┐
        ▼                  ▼                        ▼
 ┌───────────────┐  ┌───────────────┐      ┌──────────────────┐
 │ Pandas Query   │  │ Chart Builder  │      │ SQL (DuckDB) exec │
 │ Sandbox (exec) │  │ (Plotly)       │      │ on real dataframe │
 └───────────────┘  └───────────────┘      └──────────────────┘
        │                  │                        │
        └────────────┬─────┴────────────────────────┘
                      ▼
             Real computed result
                      │
                      ▼
         LLM narrates ONLY the real result
```

**Why not LangGraph for v1?** A single tool-calling loop (see `app/core/llm_engine.py`) is simpler, easier to debug, and fully sufficient for this scope. A natural v2 extension — autonomous multi-step report generation, or a planning agent that chains several analyses — would genuinely benefit from LangGraph's state graph; noted as a roadmap item below rather than added speculatively.

## 🛠️ Tech Stack

| Layer | Choice |
|---|---|
| UI | Streamlit |
| LLM | Groq (`openai/gpt-oss-120b`), native tool/function calling |
| Data | Pandas, NumPy, openpyxl |
| SQL | DuckDB (zero-copy query on the real dataframe) |
| Charts | Plotly |
| Sandbox | Restricted `exec()` in a subprocess with AST safety checks + timeout |
| Config | `python-dotenv` |
| Testing | `pytest` |

## 📁 Project Structure

```
ai-data-analyst-copilot/
├── app/
│   ├── main.py                 # Streamlit entrypoint
│   ├── core/
│   │   ├── config.py           # Env/config loading
│   │   ├── data_handler.py     # CSV/Excel upload + parsing
│   │   ├── profiling.py        # Deterministic profiling
│   │   ├── cleaning.py         # Cleaning suggestion engine
│   │   ├── sandbox.py          # Restricted pandas code executor
│   │   ├── sql_engine.py       # DuckDB SQL execution
│   │   ├── charts.py           # Plotly chart builder
│   │   ├── anomaly.py          # Outlier & trend detection
│   │   ├── llm_engine.py       # Groq tool-calling orchestration
│   │   ├── tools.py            # Tool schemas + dispatcher
│   │   ├── prompts.py          # System / grounding prompts
│   │   └── report.py           # Markdown report builder
│   └── ui/
│       ├── sidebar.py          # Upload, profile, cleaning UI
│       ├── chat.py             # Chat interface
│       └── profiling_view.py   # Profile & correlations tab
├── data/sample/sample_sales.csv
├── tests/                      # pytest unit tests
├── .streamlit/config.toml
├── .env.example
├── requirements.txt
├── Dockerfile
└── README.md
```

## 🚀 Setup

```bash
git clone <your-repo-url>
cd ai-data-analyst-copilot
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# then edit .env and add your GROQ_API_KEY (get one at https://console.groq.com)
```

## ▶️ Run

```bash
streamlit run app/main.py
```

Try it with the included sample dataset: `data/sample/sample_sales.csv`.

Sample questions to try in the chat:
- "What's the total revenue by region?"
- "Show me a monthly revenue trend chart."
- "Are there any outliers in the revenue column?"
- "Write SQL to get the top 5 highest-revenue orders."
- "What percentage of rows have missing units_sold?"

## 🧪 Testing

```bash
pytest tests/ -v
```

Covers: sandbox safety (rejects imports, dunder access, missing `result` var, syntax errors), profiling correctness against raw Pandas, and anomaly-detection accuracy against known outliers.

## 🐳 Deployment

**Option A — Streamlit Community Cloud**: point it at this repo, set `GROQ_API_KEY` in the app's Secrets, done.

**Option B — Docker:**
```bash
docker build -t data-analyst-copilot .
docker run -p 8501:8501 --env-file .env data-analyst-copilot
```

**Notes:**
- Secrets are always read from environment variables — never hard-coded.
- Large files are capped (`MAX_UPLOAD_MB`, default 200MB); very large row counts trigger a UI warning.
- The Pandas sandbox runs in a separate subprocess with a timeout (`SANDBOX_TIMEOUT_SECONDS`) to prevent runaway or expensive generated code from hanging the app.

## 🗺️ Roadmap / Possible v2 Extensions

- LangGraph-based multi-step autonomous report agent (plan → analyze → chart → summarize) for open-ended "give me a full report" requests
- Vector-store-backed RAG over data dictionaries / column metadata for larger, less self-describing datasets
- PDF export of reports (in addition to Markdown)
- Persistent session storage (multi-file, multi-session history)

## 📄 License

MIT — free to use as a portfolio/demo project.

