# Black-Scholes AI Agent
## Renaissance Technologies - Option Pricing System

A full-stack AI agent system for Black-Scholes option pricing with natural language interaction capabilities.

## Features

- **Black-Scholes Model**: Complete implementation with Greeks, Monte Carlo simulation, and implied volatility
- **AI Agent**: Natural language interface powered by private LLM (HuggingFace Transformers)
- **FastAPI Backend**: RESTful API with comprehensive endpoints
- **React Frontend**: Modern UI with chat interface and option calculator
- **Risk Analytics**: Comprehensive risk metrics and Greeks calculations

## Architecture

```
Finance_projects/black_scholes/
├── black_scholes_model.py    # Core Black-Scholes implementation
├── ai_agent.py               # AI agent with LLM integration
├── backend_api.py            # FastAPI backend server
├── requirements.txt          # Python dependencies
├── frontend/                 # React frontend application
│   ├── src/
│   │   ├── components/      # React components
│   │   ├── api/             # API client
│   │   └── App.tsx          # Main app component
│   └── package.json
└── README.md
```

## Installation

### Backend Setup

1. Create a virtual environment:
```bash
cd Finance_projects/black_scholes
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) For GPU support with PyTorch:
```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Frontend Setup

1. Navigate to frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

## Running the Application

### Start Backend Server

```bash
# From Finance_projects/black_scholes directory
python backend_api.py
```

The API will be available at `http://localhost:8000`

API documentation: `http://localhost:8000/docs`

### Start Frontend Development Server

```bash
# From Finance_projects/black_scholes/frontend directory
npm run dev
```

The frontend will be available at `http://localhost:3000`

## Usage

### AI Agent Chat Interface

The AI agent understands natural language queries about option pricing:

- "What's the price of a call option with strike 100, spot 105, 30 days to expiration, 5% risk-free rate, and 20% volatility?"
- "Calculate the Greeks for a put option: S=100, K=95, T=0.25, r=0.05, sigma=0.25"
- "What's the implied volatility if the market price is $5.50 for a call option with S=100, K=100, T=0.25, r=0.05?"

### Option Calculator

Use the calculator tab for precise calculations with all parameters:

1. Enter option parameters (Spot, Strike, Time, Rate, Volatility)
2. Select option type (Call/Put)
3. Click "Calculate Price", "Calculate Greeks", or "Risk Metrics"

### API Endpoints

#### Calculate Option Price
```bash
POST /api/v1/price
{
  "S": 100,
  "K": 100,
  "T": 0.25,
  "r": 0.05,
  "sigma": 0.2,
  "option_type": "call",
  "dividend_yield": 0.0
}
```

#### Calculate Greeks
```bash
POST /api/v1/greeks
# Same request body as /api/v1/price
```

#### Agent Query
```bash
POST /api/v1/agent/query
{
  "query": "What's the price of a call option with strike 100, spot 105, 30 days to expiration?",
  "context": []
}
```

## Model Configuration

The AI agent uses HuggingFace Transformers for local LLM inference. By default, it uses `microsoft/DialoGPT-medium`. You can change this in `ai_agent.py`:

```python
agent = BlackScholesAgent(
    model_name="your-model-name",
    use_local_llm=True,
    device="cuda"  # or "cpu"
)
```

### Recommended Models

- **Small/CPU**: `microsoft/DialoGPT-small`
- **Medium**: `microsoft/DialoGPT-medium` (default)
- **Large/GPU**: `microsoft/DialoGPT-large` or `gpt2`

For better financial domain understanding, consider fine-tuning on financial text or using models like:
- `EleutherAI/gpt-neo-1.3B`
- `EleutherAI/gpt-neo-2.7B`

## Features in Detail

### Black-Scholes Model

- **Option Pricing**: Standard Black-Scholes formula for calls and puts
- **Greeks**: Delta, Gamma, Theta, Vega, Rho
- **Implied Volatility**: Calculate IV from market prices
- **Monte Carlo Simulation**: Alternative pricing method with confidence intervals
- **Risk Metrics**: Intrinsic value, time value, moneyness, leverage
- **Price Curves**: Generate option price curves for different spot prices

### AI Agent

- **Natural Language Understanding**: Extracts parameters from queries
- **Context Awareness**: Maintains conversation history
- **Tool Calling**: Automatically selects appropriate calculations
- **Fallback Mode**: Works without LLM using rule-based responses

## Development

### Running Tests

```bash
pytest tests/
```

### Code Structure

- `black_scholes_model.py`: Core mathematical models
- `ai_agent.py`: AI agent with LLM integration and parameter extraction
- `backend_api.py`: FastAPI REST API
- `frontend/src/`: React TypeScript frontend

## Performance Notes

- **Monte Carlo**: Default 100,000 simulations. Adjust based on accuracy needs.
- **LLM Inference**: First run downloads model (~500MB for DialoGPT-medium)
- **GPU**: Recommended for LLM inference. CPU works but slower.

## License

Renaissance Technologies - Internal Use

## Contributing

This is a specialized system for Renaissance Technologies. For modifications, please contact the development team.

