from fastapi import FastAPI, HTTPException, Body, Path, Depends, Security
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from typing import Dict, List
from enum import Enum
import torch
import optuna
from abc import ABC, abstractmethod

from experiment_db import SessionLocal, ExperimentResult

app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "ok"}

#fastapi for database systems
@app.get("/results", dependencies=[Depends(verify_api_key)])
def get_results():
    session = SessionLocal()
    results = session.query(ExperimentResult).all()
    session.close()
    return [ {"id": r.id, "name": r.name, "metric": r.metric, "notes": r.notes} for r in results ]

# Add to main.py

@app.post("/results", dependencies=[Depends(verify_api_key)])
def save_result(
    name: str = Body(...),
    metric: float = Body(...),
    notes: str = Body("")
):
    session = SessionLocal()
    result = ExperimentResult(name=name, metric=metric, notes=notes)
    session.add(result)
    session.commit()
    session.close()
    return {"status": "saved"}

#update endpoint
from fastapi import Path

@app.put("/results/{result_id}")
def update_result(
    result_id: int = Path(...),
    name: str = Body(None),
    metric: float = Body(None),
    notes: str = Body(None)
):
    session = SessionLocal()
    result = session.query(ExperimentResult).filter(ExperimentResult.id == result_id).first()
    if not result:
        session.close()
        raise HTTPException(status_code=404, detail="Result not found")
    if name is not None:
        result.name = name
    if metric is not None:
        result.metric = metric
    if notes is not None:
        result.notes = notes
    session.commit()
    session.close()
    return {"status": "updated"}

#delete endpoint
@app.delete("/results/{result_id}")
def delete_result(result_id: int = Path(...)):
    session = SessionLocal()
    result = session.query(ExperimentResult).filter(ExperimentResult.id == result_id).first()
    if not result:
        session.close()
        raise HTTPException(status_code=404, detail="Result not found")
    session.delete(result)
    session.commit()
    session.close()
    return {"status": "deleted"}

# Define data models
class SimulationType(str, Enum):
    FDTD = "FDTD"
    RAY_TRACING = "RAY_TRACING"
    MEEP = "MEEP"

class SimulationRequest(BaseModel):
    simulation_type: SimulationType
    params: Dict

class OptimizationRequest(BaseModel):
    simulation_type: SimulationType
    wavelength_min: float
    wavelength_max: float
    grid_size_min: int
    grid_size_max: int
    n_trials: int

class WorkflowRequest(BaseModel):
    workflow_json: str

# Initialize modules
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
fdtd_simulator = FDTDSimulation(device=device)
ray_tracing_simulator = RayTracingSimulation()
meep_simulator = MEEPSimulation()
data_manager = DataManager()
simulation_engine = {
    "FDTD": fdtd_simulator, 
    "RAY_TRACING": ray_tracing_simulator,
    "MEEP": meep_simulator
}
workflow_automation = WorkflowAutomation(simulation_engine, data_manager)

# Simulation endpoint
@app.post("/simulate")
async def simulate(request: SimulationRequest):
    """
    Endpoint to run a simulation.
    """
    try:
        if request.simulation_type == SimulationType.FDTD:
            simulator = fdtd_simulator
            # Validate required params
            if not all(k in request.params for k in ("wavelength", "grid_size")):
                raise HTTPException(status_code=422, detail="Missing 'wavelength' or 'grid_size' for FDTD simulation.")
        elif request.simulation_type == SimulationType.RAY_TRACING:
            simulator = ray_tracing_simulator
            if not all(k in request.params for k in ("focal_length", "aperture")):
                raise HTTPException(status_code=422, detail="Missing 'focal_length' or 'aperture' for Ray Tracing simulation.")
        elif request.simulation_type == SimulationType.MEEP:
            simulator = meep_simulator
            if not all(k in request.params for k in ("wavelength", "resolution", "size_x", "size_y")):
                raise HTTPException(status_code=422, detail="Missing one or more required parameters for MEEP simulation.")
        else:
            raise HTTPException(status_code=400, detail="Invalid simulation type")

        result = simulator.simulate(request.params)
        return {"result": result}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Optimization endpoint
@app.post("/optimize")
async def optimize(params: OptimizationRequest):
    """
    Endpoint to run AutoML optimization.
    """
    try:
        def objective(trial):
            wavelength = trial.suggest_float("wavelength", params.wavelength_min, params.wavelength_max)
            grid_size = trial.suggest_int("grid_size", params.grid_size_min, params.grid_size_max)
            request_params = {"wavelength": wavelength, "grid_size": grid_size}

            if params.simulation_type == SimulationType.FDTD:
                simulator = fdtd_simulator
            elif params.simulation_type == SimulationType.RAY_TRACING:
                simulator = ray_tracing_simulator
            elif params.simulation_type == SimulationType.MEEP:
                simulator = meep_simulator
            else:
                raise ValueError("Invalid simulation type")

            result = simulator.simulate(request_params)
            return result  # Replace with your actual optimization metric

        study = optuna.create_study(direction="minimize")  # Or "maximize"
        study.optimize(objective, n_trials=params.n_trials)

        best_params = study.best_params
        best_value = study.best_value
        return {"best_params": best_params, "best_value": best_value}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Workflow endpoint
@app.post("/execute_workflow")
async def execute_workflow(request: WorkflowRequest):
    """
    Endpoint to execute a workflow.
    """
    try:
        results = workflow_automation.execute_workflow(request.workflow_json)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "OK"}

class SimulationInterface(ABC):
    @abstractmethod
    def simulate(self, params: Dict) -> object:
        """
        Abstract method for running a simulation.

        Args:
            params (Dict): Simulation parameters.

        Returns:
            object: Simulation results.
        """
        pass

import torch
#from simulation_interface import SimulationInterface

class FDTDSimulation(SimulationInterface):
    def __init__(self, device='cpu'):
        self.device = torch.device(device)

    def simulate(self, params):
        """
        Basic FDTD simulation using PyTorch.
        Replace with your actual FDTD implementation.
        """
        try:
            wavelength = torch.tensor(params['wavelength'], dtype=torch.float32, device=self.device)
            grid_size = torch.tensor(params['grid_size'], dtype=torch.int32, device=self.device)

            # Dummy calculation
            result = wavelength * grid_size
            return result.cpu().numpy()
        except Exception as e:
            raise ValueError(f"FDTD Simulation Failed: {str(e)}")

#from simulation_interface import SimulationInterface

class RayTracingSimulation(SimulationInterface):
    def __init__(self):
        pass

    def simulate(self, params):
        """
        Placeholder for ray tracing simulation.
        Replace with your actual ray tracing implementation.
        """
        try:
            # Dummy calculation
            focal_length = params['focal_length']
            aperture = params['aperture']
            result = focal_length / aperture
            return result
        except Exception as e:
            raise ValueError(f"Ray Tracing Simulation Failed: {str(e)}")

import numpy as np
#from simulation_interface import SimulationInterface

class MEEPSimulation(SimulationInterface):
    def __init__(self):
        pass

    def simulate(self, params):
        """
        MEEP simulation placeholder.
        Replace with your actual MEEP implementation.
        """
        try:
            # Get parameters
            wavelength = params['wavelength']
            resolution = params['resolution']
            size_x = params['size_x']
            size_y = params['size_y']

            # Placeholder for MEEP simulation
            # In a real implementation, you would use:
            # import meep as mp
            # cell = mp.Simulation(...)
            # ...

            # Return dummy data
            return np.random.random((size_x, size_y))

        except Exception as e:
            raise ValueError(f"MEEP Simulation Failed: {str(e)}")

import os
import torch
import numpy as np

class DataManager:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def save_simulation_data(self, data, filename="simulation_data.npy"):
        """
        Save simulation data to a numpy file.
        """
        filepath = os.path.join(self.data_dir, filename)
        np.save(filepath, data)
        return filepath

    def load_simulation_data(self, filename="simulation_data.npy"):
        """
        Load simulation data from a numpy file.
        """
        filepath = os.path.join(self.data_dir, filename)
        try:
            data = np.load(filepath)
            return data
        except FileNotFoundError:
            return None

    def save_model(self, model, filename="model.pth"):
        """
        Save a PyTorch model.
        """
        filepath = os.path.join(self.data_dir, filename)
        torch.save(model.state_dict(), filepath)
        return filepath

    def load_model(self, model, filename="model.pth"):
        """
        Load a PyTorch model.
        """
        filepath = os.path.join(self.data_dir, filename)
        try:
            model.load_state_dict(torch.load(filepath))
            return model
        except FileNotFoundError:
            return None

import json
from typing import List, Dict

class WorkflowAutomation:
    def __init__(self, simulation_engine, data_manager):
        self.simulation_engine = simulation_engine
        self.data_manager = data_manager

    def execute_workflow(self, workflow_json: str) -> List[Dict]:
        """
        Execute a workflow defined in JSON format.
        """
        try:
            workflow = json.loads(workflow_json)
            results = []
            
            for node in workflow["nodes"]:
                node_type = node["type"]
                node_params = node["params"]
                
                if node_type == "simulate":
                    simulation_type = node_params.get("simulation_type", "FDTD")
                    simulator = self.simulation_engine.get(simulation_type)
                    if not simulator:
                        raise ValueError(f"Unknown simulation type: {simulation_type}")
                    
                    simulation_result = simulator.simulate(node_params)
                    results.append({"node_id": node["id"], "result": simulation_result})
                elif node_type == "save_data":
                    data_to_save = next((r["result"] for r in results if r["node_id"] == node_params["input"]), None)
                    if data_to_save is not None:
                        filepath = self.data_manager.save_simulation_data(data_to_save, node_params["filename"])
                        results.append({"node_id": node["id"], "filepath": filepath})
                    else:
                        raise ValueError(f"Input data not found for node: {node['id']}")
                else:
                    raise ValueError(f"Unknown node type: {node_type}")
            
            return results
        except Exception as e:
            raise ValueError(f"Workflow Execution Failed: {str(e)}")

#API key dependency

API_KEY = "your-secret-api-key"
api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")

