# Quick Start Guide

## Prerequisites

- Python 3.8+
- Node.js 16+ and npm
- (Optional) CUDA-capable GPU for faster LLM inference

## 5-Minute Setup

### 1. Backend Setup (2 minutes)

```bash
cd Finance_projects/black_scholes

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Start Backend (30 seconds)

```bash
# Windows
python backend_api.py

# Or use the script
run_backend.bat
```

Backend runs on `http://localhost:8000`

### 3. Frontend Setup (2 minutes)

```bash
# In a new terminal
cd Finance_projects/black_scholes/frontend

# Install dependencies
npm install

# Start dev server
npm run dev
```

Frontend runs on `http://localhost:3000`

## First Use

1. Open `http://localhost:3000` in your browser
2. Go to "AI Agent Chat" tab
3. Try: "What's the price of a call option with strike 100, spot 105, 30 days to expiration, 5% risk-free rate, and 20% volatility?"
4. Or use the "Option Calculator" tab for precise calculations

## Troubleshooting

### LLM Model Download

On first run, the AI agent will download the LLM model (~500MB). This happens automatically but may take a few minutes.

### Port Already in Use

If port 8000 or 3000 is in use:
- Backend: Edit `backend_api.py`, change `port=8000` to another port
- Frontend: Edit `vite.config.ts`, change `port: 3000` to another port

### GPU Not Detected

The system works on CPU. For GPU:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Then in `ai_agent.py`, set `device="cuda"`.

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check API docs at `http://localhost:8000/docs`
- Run tests: `pytest tests/`

