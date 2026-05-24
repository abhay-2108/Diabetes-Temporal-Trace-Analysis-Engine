# DiaTrace.AI — Operational Setup & Execution Guide

This document contains step-by-step instructions for initializing environments, extracting model weights, starting servers, and verifying the end-to-end integration of the DiaTrace.AI clinical decision engine.

---

## 🛠️ Step 1: Initialize Backend Environment & Data

To prepare your local Python environment and load the patient SQLite databases, run these commands inside your project root directory:

```powershell
# 1. Create the main virtual environment using uv
uv venv .venv

# 2. Activate on Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# 3. Install core dependencies
uv pip install -r requirements.txt

# 4. Generate the database and ingest patient SOAP notes
python src/pipelines/ingest_notes.py

# 5. Index the ADA Guidelines into ChromaDB Vector Store
python src/pipelines/ingest_ada.py
```

---

## 💾 Step 2: Extract your Local Ollama GGUF Base Model

Ollama stores model weights as SHA-256 blobs in an internal registry directory. To use it in vLLM without downloading heavy models, extract your local `qwen2.5:7b` file:

1. **Locate the Hash:** Open the file:
   `C:\Users\<YourUsername>\.ollama\models\manifests\registry.ollama.ai\library\qwen2.5\7b`
   Look for the digest field of the model layer (e.g. `"digest": "sha256:2bada8a7450677000f678be90653b85d364de7db25eb5ea54136ada5f3933730"`).
2. **Copy & Rename:** Go to your blobs folder `C:\Users\<YourUsername>\.ollama\models\blobs\`. Find the file matching your hash, replacing the colon with a hyphen (e.g. `sha256-2bada8a7450677000f678be90653b85d364de7db25eb5ea54136ada5f3933730`).
3. **Move to Project:** Copy this file into your workspace project root folder and rename it to:
   `qwen2.5-7b.gguf` (size: ~4.68 GB)

---

## 🚀 Step 3: Set up and Launch vLLM Server (inside WSL2)

> [!IMPORTANT]
> **Windows vLLM Compatibility Warning:**
> vLLM depends on compiled C++ extension bindings (`vllm._C`), which do not compile natively on Windows. To run the vLLM server with full GPU acceleration, **you must execute it inside WSL2 (Windows Subsystem for Linux Ubuntu)**.

Inside your WSL2 terminal, navigate to the project directory:

```bash
# 1. Create a virtual environment inside WSL2
uv venv vllm_env

# 2. Activate the environment
source vllm_env/bin/activate

# 3. Install vLLM (standard Linux wheel)
uv pip install vllm

# 4. Launch the Multi-LoRA vLLM Server on port 8000
python -m vllm.entrypoints.openai.api_server \
    --model ./qwen2.5-7b.gguf \
    --served-model-name qwen2.5:7b \
    --enable-lora \
    --lora-modules diatrace-lora=raj0120/diatrace-qwen2.5-lora \
    --max-lora-rank 64 \
    --port 8000 \
    --gpu-memory-utilization 0.80
```

*Note: Your Hugging Face fine-tuned adapter (`raj0120/diatrace-qwen2.5-lora`) will be downloaded and overlaid onto the GGUF base model dynamically.*

---

## 💻 Step 4: Launch the Live Multi-Agent Backend

Open a new PowerShell terminal on Windows, activate the main environment, and start the FastAPI web server.

To toggle between live multi-agent execution and fast local mockup rendering, set the `USE_MOCK_LLM` variable:

```powershell
# 1. Activate main environment
.\.venv\Scripts\Activate.ps1

# 2. Enable Live Inference Mode (detects local vLLM or falls back to legacy Ollama)
$env:USE_MOCK_LLM="false"

# 3. Start the FastAPI API Server
python src/web_app/api.py
```
*Successfully listening at `http://127.0.0.1:8000`.*

---

## 🌐 Step 5: Launch the React Frontend

To bypass space/ampersand path parsing bugs on Windows, boot up the Vite frontend developer server by executing node directly inside the frontend folder:

```powershell
# 1. Navigate to frontend folder
cd src/web_app/frontend

# 2. Boot Vite server directly
node "node_modules/vite/bin/vite.js"
```
*Successfully hosting the developer server at `http://localhost:5173/`.*

---

## 📊 Step 6: End-to-End Integration Testing

To verify the database connections, local LLM detection, and agent orchestration, run our pre-configured script from the root directory:

```powershell
.\.venv\Scripts\python.exe src/web_app/test_api_run.py
```

This verifies the live sequential execution of the 4 agents and returns structured guideline auditing results in your console.
