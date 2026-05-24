# DiaTrace.AI: Type 2 Diabetes Mellitus (T2DM) Care & Complication Predictor

DiaTrace.AI is a clinical intelligence engine designed to solve the problem of **Clinical Inertia** and fragmented Electronic Health Records (EHR). By leveraging a fine-tuned Medical LLM and a multi-agent system coordinated via the CrewAI framework, DiaTrace.AI ingests years of unstructured physician progress notes and structured laboratory telemetry to map patient trajectories, flag hidden microvascular complications, and recommend guideline-aligned therapeutic adjustments.

---

## 1. The Core Problem

In busy hospital environments, clinicians often only have time to review the most recent lab tests. Longitudinal patient details—such as mild symptoms noted three years ago or medication non-adherence documented two years ago—remain buried in massive walls of unstructured text. This leads to:
1. **Missed Early Complications:** Sub-clinical organ damage goes unnoticed (e.g., dropping eGFR rates coupled with sporadic microalbuminuria).
2. **Suboptimal Pharmacotherapy:** Medication adjustments are made without cross-referencing documented side effects or official American Diabetes Association (ADA) guidelines.

**DiaTrace.AI solves this by building an auditable, explainable, and multi-agent temporal analysis engine.**

---

## 2. Multi-Agent Architecture

The core of DiaTrace.AI is built on the **CrewAI** framework, routing tasks through four specialized agent nodes:

```
                  ┌──────────────────────────────┐
                  │    Chronological EHR Text    │
                  └──────────────┬───────────────┘
                                 ▼
                   ┌────────────────────────────┐
                   │ 1. Temporal Mining Agent   │
                   └─────────────┬──────────────┘
                                 ▼ Extract Structured Timeline
                   ┌─────────────┴──────────────┐
                   ▼                            ▼
      ┌─────────────────────────┐  ┌─────────────────────────┐
      │ 2. Complication Agent   │  │ 3. Medication Agent     │
      └────────────┬────────────┘  └────────────┬────────────┘
       [Scans for kidney/nerve      [Checks drug lines against 
        damage trends over years]    ADA guidelines & failures]
                   │                            │
                   └─────────────┬──────────────┘
                                 ▼ Combined Analysis
                   ┌─────────────▼──────────────┐
                   │ 4. XAI Clinical Reasoner   │
                   └────────────────────────────┘
```

* **Temporal Trajectory Mining Agent (Agent 1):** Ingests raw EHR texts, extracts structured JSON timelines detailing medications, biomarkers, symptoms, and side effects.
* **Microvascular Complication Agent (Agent 2):** Scans the timeline for early indicators of kidney, nerve, or eye damage (e.g., declining eGFR, positive microalbuminuria).
* **Pharmacotherapy Optimizer Agent (Agent 3):** Looks up active medication lines against a Vector Database of the latest ADA (American Diabetes Association) Standards of Care.
* **XAI Clinical Reasoner Agent (Agent 4):** Resolves clinical conflicts, generates the consolidated clinical recommendation, and compiles the **Causal Evidence Chain** for human verification.

---

## 3. Directory Layout

The repository is structured to separate data pipelines, agent nodes, configuration, and interface code:

```
DiaTrace.AI/
├── config/                  # App configurations, LLM system prompts, and DB schemas
├── data/                    # Local sandbox directory for datasets
│   ├── raw/                 # Unprocessed Kaggle & MIMIC-IV demo data
│   └── processed/           # Cleaned cohort files and telemetry databases
├── src/                     # Core application source code
│   ├── agents/              # CrewAI multi-agent implementation
│   │   ├── orchestrator_agent.py
│   │   ├── temporal_miner.py
│   │   ├── complication_agent.py
│   │   ├── pharmacotherapy_optimizer.py
│   │   └── clinical_reasoner.py
│   ├── database/            # Mock EHR SQLite telemetry database handlers
│   ├── pipelines/           # Raw data ingestion, ETL, and formatting pipelines
│   ├── vector_db/           # ChromaDB/FAISS vector store for ADA guidelines
│   └── web_app/             # Streamlit/FastAPI frontend & developer server
├── tests/                   # Pytest test suite for agents and pipelines
├── notebooks/               # Jupyter notebooks for model fine-tuning experiments
├── requirements.txt         # Python project dependencies
├── .gitignore               # Excludes large models and build directories
├── RUNNING.md               # Step-by-step setup and execution guide
└── README.md                # Project documentation overview
```

---

## 4. Setup, Running & Verification

All detailed instructions for setting up the local sandboxes, extracting local GGUF models, running vLLM inside WSL2, starting the FastAPI and React servers, and triggering E2E integration tests are documented in the operational guide:

👉 **[Go to RUNNING.md (Operational Guide)](file:///p:/AIML%20Projects/Diabetes%20Temporal%20Trace%20&%20Analysis%20Engine/RUNNING.md)**
