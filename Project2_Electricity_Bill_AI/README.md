# ⚡ AI-Powered Electricity Bill Calculator & Comparison Chatbot

Calculate what the SAME electricity consumption would cost under the **old tariff policy** (previous government) versus the **new tariff policy** (current government) — with a deterministic, slab-wise breakdown and an AI chatbot to explain the results in plain language.

> **Flagship project #2** in an ongoing collection of production-oriented GenAI applications.

## 🧩 Problem Statement

Electricity tariff policy changes (e.g. a revised free-unit subsidy) are hard to reason about manually, since telescopic slab billing means a small policy change can have a disproportionate effect depending on your usage level. Naive "AI bill calculators" risk having the LLM hallucinate numbers. This project solves both: a deterministic Python calculation engine (the only source of billing truth) paired with a conversational layer that can only explain and compare — never invent — those numbers.

## 🎯 Objective

A Streamlit app where a user enters their consumption (in units) **once** and gets:
- the exact bill under the **old tariff policy** (single-tier: only 100 units ever free),
- the exact bill under the **new tariff policy** (two-tier: 200 free units if total ≤500, else 100),
- a full old-vs-new comparison (bill, % change, slab-by-slab),
- descriptive charts,
- and a chatbot that can answer natural-language questions (including Tanglish) grounded entirely in the already-computed numbers.

## ⚠️ Important: Tariff Rates — Verify Before Real Use

`app/core/tariff.py` models two policies for TANGEDCO's domestic (LT-IA) category:
- **Old policy**: single-tier — only the first 100 units are ever free, regardless of total consumption.
- **New policy**: two-tier — total consumption ≤500 units gets 200 free units; above 500, only 100 units are free (same steeper ladder as the old policy beyond that point).

These are sourced from TNERC Tariff Order No. 6 of 2024 (effective 1 July 2024) via third-party cross-referencing at the time this was written — **not fetched live from an official source**. Tamil Nadu's tariff is revised periodically. **Verify current rates at [tnerc.tn.gov.in](https://www.tnerc.tn.gov.in) before relying on this for real billing decisions**, and update `tariff.py` if they've changed.

**Why the slab-wise comparison can show extra rows:** the old and new policies can use entirely different slab boundaries for the same units (e.g. 450 units: old policy's ladder starts 1-100, new policy's Tier 1 ladder starts 1-200). The comparison logic aligns rows by their actual slab boundary rather than by position, so a boundary that exists on only one side shows as zero on the other — this is correct, not a bug (see `comparison.structure_changed`).

This calculator covers **energy (slab) charges only** — fixed/service charges, the quarterly Fuel Cost Adjustment (FCA), and electricity duty are not modeled, matching the slab-wise breakdown originally scoped for this tool.

## ✨ Key Features

- Deterministic, Python-only bill calculation — **the LLM never computes a number**
- Slab-wise breakdown for both old-policy and new-policy bills, for the same consumption
- Full previous-vs-current comparison: units, bill, difference, % change
- Slab-by-slab comparison table
- 4 descriptive charts (consumption comparison, bill comparison, slab-wise consumption, slab-wise bill)
- AI chatbot (Groq) that explains/compares using only the already-calculated data — supports Tanglish questions
- Automatic model fallback if a Groq model hits a rate/quota limit
- Clean, editable tariff configuration separated from calculation logic

## 🏗️ Architecture

```
┌──────────────┐     ┌─────────────────────┐     ┌───────────────────┐
│  Streamlit   │────▶│  calculator.py /     │────▶│  Deterministic      │
│  UI (inputs) │     │  comparison.py       │     │  BillResult /       │
│              │◀────│  (pure Python)        │◀────│  ComparisonResult   │
└──────────────┘     └─────────────────────┘     └───────────────────┘
                                │
                                ▼ (serialized as JSON context)
                      ┌─────────────────────┐
                      │  Groq LLM chatbot    │
                      │  — narrates ONLY     │
                      │  the given numbers   │
                      └─────────────────────┘
```

The tariff/slab configuration (`tariff.py`) is fully separated from the calculation engine (`calculator.py`), so rates can be updated without touching any logic.

## 🛠️ Tech Stack

Streamlit · Python · Pandas · Plotly · Groq (LLM) · python-dotenv

## 📁 Project Structure

```
Project2_Electricity_Bill_AI/
├── app/
│   ├── main.py
│   ├── core/
│   │   ├── config.py       # Env/Groq settings
│   │   ├── tariff.py       # EDIT THIS for official rates — slab config
│   │   ├── calculator.py   # Deterministic bill calculation
│   │   ├── comparison.py   # Previous vs current comparison logic
│   │   ├── chatbot.py      # Groq chatbot + model fallback
│   │   └── charts.py       # Plotly chart builders
│   └── ui/
│       ├── sidebar.py
│       ├── dashboard.py
│       └── chat.py
├── tests/
│   ├── test_calculator.py
│   ├── test_comparison.py
│   └── test_tariff.py
├── .streamlit/config.toml
├── .env.example
├── requirements.txt
└── README.md
```

## 🚀 Setup (Local)

```bash
git clone <your-repo-url>
cd Project2_Electricity_Bill_AI
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and add your GROQ_API_KEY (get one at https://console.groq.com)
```

## ▶️ Run Locally

```bash
streamlit run app/main.py
```

Enter your consumption (units) in the sidebar, click **Calculate**, then explore the dashboard and chat tab. Try:
- "Old policy bill evlo?"
- "New policy la evlo save aagum?"
- "Which slab contributes most to my bill?"
- "450 units na epadi old vs new policy la difference erukum?"

## 🧪 Testing

```bash
pytest tests/ -v
```

Covers: 0 units, exactly 100/200 units, values crossing slab boundaries, high consumption reaching the open-ended slab, negative/invalid input rejection, old-vs-new policy comparison at various consumption levels (including the exact 450-unit case), the zero-bill percentage edge case, cross-structure slab alignment when old/new policies use different boundaries, and tariff config validation (contiguity, open-ended last slab, non-negative rates).

## ☁️ Deploy on Streamlit Cloud

1. Push this project to a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and click **New app**.
3. Select your repo, branch, and set the main file path to `app/main.py`.
4. Under **Advanced settings → Secrets**, add:
   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   GROQ_MODEL = "llama-3.3-70b-versatile"
   ```
5. Click **Deploy**. Streamlit Cloud reads secrets the same way `python-dotenv` reads `.env` locally — no code changes needed.

## 📄 License

MIT — free to use as a portfolio/demo project.
