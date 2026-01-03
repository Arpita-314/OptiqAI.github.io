# File Structure and Organization

## Complete File Structure

```
Finance_projects/black_scholes/
│
├── Core Implementation Files
│   ├── black_scholes_model.py      # Black-Scholes mathematical model
│   ├── ai_agent.py                  # AI agent with LLM integration
│   ├── file_manager.py              # File handling and data persistence
│   ├── backend_api.py               # FastAPI REST API server
│   └── __init__.py                  # Package initialization
│
├── Frontend Application
│   └── frontend/
│       ├── src/
│       │   ├── api/
│       │   │   └── client.ts        # API client for backend
│       │   ├── components/
│       │   │   ├── ChatInterface.tsx    # AI chat interface
│       │   │   └── OptionCalculator.tsx # Option calculator form
│       │   ├── App.tsx              # Main React component
│       │   ├── App.css              # App styles
│       │   ├── main.tsx             # React entry point
│       │   └── index.css            # Global styles
│       ├── index.html               # HTML template
│       ├── package.json             # Node dependencies
│       ├── tsconfig.json            # TypeScript config
│       ├── vite.config.ts           # Vite build config
│       ├── tailwind.config.js       # Tailwind CSS config
│       └── postcss.config.js        # PostCSS config
│
├── Data Storage (Generated)
│   ├── data/                        # Main data directory
│   │   ├── calculations/            # Saved calculations (JSON)
│   │   ├── history/                 # Conversation history (JSON)
│   │   └── exports/                 # CSV exports
│   └── demo_data/                   # Demo data (from demo.py)
│
├── Tests
│   └── tests/
│       └── test_black_scholes.py    # Unit tests
│
├── Documentation
│   ├── README.md                    # Main documentation
│   ├── QUICKSTART.md                # Quick start guide
│   └── FILE_STRUCTURE.md            # This file
│
├── Configuration Files
│   ├── requirements.txt             # Python dependencies
│   ├── run_backend.sh               # Linux/Mac startup script
│   ├── run_backend.bat              # Windows startup script
│   └── .env.example                 # Environment variables template
│
└── Demo & Examples
    └── demo.py                      # Demonstration script

```

## File Handling Details

### Data Storage Structure

#### Calculations (`data/calculations/`)
- Format: JSON files
- Naming: `{calculation_id}_{timestamp}.json`
- Content:
  ```json
  {
    "calculation_id": "calc_xxx",
    "timestamp": "20260103_162930",
    "datetime": "2026-01-03T16:29:30.827245",
    "data": {
      "parameters": {...},
      "price": 4.61,
      "greeks": {...},
      "risk_metrics": {...}
    }
  }
  ```

#### Conversation History (`data/history/`)
- Format: JSON files
- Naming: `session_{session_id}_{timestamp}.json`
- Content:
  ```json
  {
    "session_id": "session_xxx",
    "timestamp": "20260103_162930",
    "datetime": "2026-01-03T16:29:30.827245",
    "message_count": 5,
    "messages": [...]
  }
  ```

#### Exports (`data/exports/`)
- Format: CSV files
- Naming: `calculations_export_{timestamp}.csv`
- Contains flattened calculation data for analysis

### File Manager API

The `FileManager` class provides:

1. **Save Operations**
   - `save_calculation(calc_id, data)` - Save calculation
   - `save_conversation_history(session_id, messages)` - Save chat history

2. **Load Operations**
   - `load_calculation(calc_id)` - Load by ID
   - `load_conversation_history(session_id)` - Load session

3. **Export Operations**
   - `export_to_csv(calculations, filename)` - Export to CSV

4. **Management Operations**
   - `list_calculations(limit)` - List recent calculations
   - `delete_calculation(calc_id)` - Delete calculation
   - `get_statistics()` - Get storage statistics

### API Endpoints for File Operations

- `POST /api/v1/calculations/save` - Save calculation
- `GET /api/v1/calculations/list` - List calculations
- `GET /api/v1/calculations/{id}` - Get specific calculation
- `POST /api/v1/export/csv` - Export to CSV
- `GET /api/v1/stats` - Get file statistics

## File Naming Conventions

- **Calculations**: `calc_{id}_{timestamp}.json`
- **Sessions**: `session_{id}_{timestamp}.json`
- **Exports**: `calculations_export_{timestamp}.csv`
- **Timestamps**: `YYYYMMDD_HHMMSS` format

## Data Persistence Flow

1. **User makes calculation** → Backend processes
2. **Calculation saved** → `data/calculations/` directory
3. **User queries agent** → Conversation saved to `data/history/`
4. **User exports data** → CSV file in `data/exports/`
5. **User views history** → Files loaded from storage

## Best Practices

1. **Regular Cleanup**: Old calculations can be archived or deleted
2. **Backup**: Regular backups of `data/` directory recommended
3. **Privacy**: Sensitive financial data should be encrypted
4. **Performance**: Large datasets should use database instead of files

