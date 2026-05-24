# DiaTrace.AI — Fast Launch & Command Cheatsheet

Since your virtual environments, library packages, and model file copying are **already done**, use these direct commands to run the project.

---

### 🖥️ Terminal 1: Launch the vLLM Server (WSL2 Linux)
Open your WSL2 terminal, navigate to the project directory, and run:

```bash
# 1. Activate the environment
source vllm_env/bin/activate

# 2. Start the Multi-LoRA Server
python -m vllm.entrypoints.openai.api_server \
    --model ./qwen2.5-7b.gguf \
    --served-model-name qwen2.5:7b \
    --enable-lora \
    --lora-modules diatrace-lora=raj0120/diatrace-qwen2.5-lora \
    --max-lora-rank 64 \
    --port 8000 \
    --gpu-memory-utilization 0.80
```

---

### 🖥️ Terminal 2: Launch the FastAPI Backend (Windows PowerShell)
Open a new PowerShell terminal at the project root and run:

```powershell
# 1. Activate the environment
.\.venv\Scripts\Activate.ps1

# 2. Enable Live Inference & Start API Server
$env:USE_MOCK_LLM="false"
python src/web_app/api.py
```
*Port: `http://localhost:8000`*

---

### 🖥️ Terminal 3: Launch the React Frontend (Windows PowerShell)
Open another PowerShell terminal and run:

```powershell
# 1. Navigate to the frontend directory
cd src/web_app/frontend

# 2. Launch the developer server directly
node "node_modules/vite/bin/vite.js"
```
*Port: `http://localhost:5173/`*

---

### 🖥️ Terminal 4: Trigger E2E Integration Test (Windows PowerShell)
To test the active multi-agent pipeline from the command line, run:

```powershell
.\.venv\Scripts\python.exe src/web_app/test_api_run.py
```
