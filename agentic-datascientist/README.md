# Agentic Data Scientist 🚀

A multi‑agent GenAI system that autonomously explores, analyzes, models, and reports on any CSV dataset — like a junior Data Scientist that never sleeps.

## Highlights
- **Dataset Understanding Agent**: infers schema, leaks/missing/outliers, cleaning plan.
- **EDA Agent**: generates plots and stats + markdown narrative.
- **Model Agent**: AutoML sweep across XGBoost / LightGBM / RandomForest with cross‑validation.
- **Report Agent**: compiles a human‑readable report (Markdown) and structured JSON.
- **Planner**: coordinates agents; optional LLM (Ollama / OpenAI) to propose next steps.
- **Serving**: FastAPI endpoints and a Gradio UI.
- **MLOps**: run metadata + metrics saved to `examples/output/`.

> Works **without any LLM keys** (rule‑based fallback). For GenAI features, set an LLM in `.env`.

## Quickstart

### 1) Install
```bash
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2) (Optional) Configure LLM
Create a `.env` with one of:
```
# For OpenAI
OPENAI_API_KEY=sk-...

# Or for Ollama (local)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3
```

### 3) Run end‑to‑end on the demo dataset
```bash
python -m src.orchestration.run_pipeline --data examples/data/demo.csv --target churned
```
Artifacts will be written to `examples/output/`:
- `report.md` – human narrative of insights
- `metrics.json` – model + validation metrics
- `model.joblib` – trained model
- `schema.json` – inferred types and basic stats

### 4) Serve API + UI
```bash
# FastAPI
uvicorn src.api.app:app --reload

# Gradio UI
python ui/app.py
```

Open http://127.0.0.1:8000/docs for the API and the Gradio link for the UI.

## Repo Structure
```
agentic-datascientist/
├── src/
│   ├── agents/                # dataset_agent.py, eda_agent.py, model_agent.py, report_agent.py, planner.py
│   ├── orchestration/         # run_pipeline.py
│   ├── utils/                 # io.py, plotting.py, metrics.py
│   └── api/                   # app.py (FastAPI)
├── ui/                        # app.py (Gradio)
├── examples/
│   ├── data/demo.csv
│   └── output/
├── configs/                   # future configs
├── tests/
├── requirements.txt
└── README.md
```

## Notes
- Default run uses **rule‑based planning** (no LLM). If an LLM is configured, the planner will auto‑ask it to critique and suggest next steps.
- The UI supports natural‑language prompts; when no LLM, it maps a small set of intents to actions.

---

*Made for showcasing senior‑level DS + GenAI system design.* 
