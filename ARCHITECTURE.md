# FOAML Architecture Overview

This document describes the high-level architecture of the FOAML (Fourier Optics AutoML Framework) platform. The system is designed for extensibility, performance, and modularity, inspired by industry-standard simulation tools.


## System Architecture

graph TD
    subgraph User Interface
        A1(PyQt5 Desktop GUI)
        A2(Web Frontend [optional])
    end

    subgraph API Layer
        B1(FastAPI REST API)
    end

    subgraph Orchestration & Workflow
        C1(Workflow Engine)
        C2(Job Scheduler)
        C3(Experiment Manager)
    end

    subgraph Simulation Core
        D1(FDTD Engine [C++/CUDA])
        D2(Ray Tracing Engine [C++])
        D3(MEEP Wrapper)
        D4(Plugin System)
    end

    subgraph Data Management
        E1(SQL/NoSQL Database)
        E2(File Storage)
        E3(Data IO/Converters)
    end

    subgraph ML/Optimization
        F1(PyTorch Models)
        F2(Optuna/Hyperopt)
        F3(Experiment Tracking [MLflow/W&B])
    end

    subgraph Utilities
        G1(Logging & Monitoring)
        G2(Config Management)
        G3(Error Handling)
    end

    %% Connections
    A1 -- "calls" --> B1
    A2 -- "calls" --> B1
    B1 -- "triggers" --> C1
    C1 -- "submits jobs to" --> C2
    C2 -- "runs" --> D1
    C2 -- "runs" --> D2
    C2 -- "runs" --> D3
    C2 -- "loads plugins" --> D4
    D1 -- "reads/writes" --> E2
    D2 -- "reads/writes" --> E2
    D3 -- "reads/writes" --> E2
    D1 -- "stores results" --> E1
    D2 -- "stores results" --> E1
    D3 -- "stores results" --> E1
    C1 -- "logs to" --> G1
    D1 -- "logs to" --> G1
    D2 -- "logs to" --> G1
    D3 -- "logs to" --> G1
    F1 -- "uses data from" --> E1
    F1 -- "uses data from" --> E2
    F2 -- "optimizes" --> D1
    F2 -- "optimizes" --> D2
    F3 -- "tracks" --> F1
    G2 -- "configures" --> B1
    G2 -- "configures" --> C1
    G2 -- "configures" --> D1
    G3 -- "handles errors from" --> B1
    G3 -- "handles errors from" --> C1
    G3 -- "handles errors from" --> D1



## Module Descriptions

- User Interface: PyQt5 desktop GUI for user interaction; optional web frontend for remote access.
- API Layer: FastAPI REST API connects UI to backend services.
- Orchestration & Workflow: Manages simulation jobs, experiment lifecycles, and workflow automation.
- Simulation Core: High-performance C++/CUDA engines for FDTD, ray tracing, and plugin-based extensions.
- Data Management: Handles persistent storage (PostgreSQL, file storage, converters).
- ML/Optimization: Integrates ML models, hyperparameter optimization, and experiment tracking.
- Utilities: Logging, configuration management, and error handling for robustness and maintainability.

---

## Extensibility & Scalability

- Each module is designed to be independently testable and replaceable.
- The simulation core supports plugins for new solvers or physics.
- The architecture supports scaling to distributed/cloud environments.

---

*This architecture enables FOAML to be robust, extensible, and competitive with