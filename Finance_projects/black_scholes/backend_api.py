"""
FastAPI Backend for Black-Scholes AI Agent
Renaissance Technologies - Option Pricing System
"""

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
from enum import Enum
import uvicorn

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from black_scholes_model import (
    BlackScholesModel, OptionParams, OptionType, OptionGreeks
)
from ai_agent import BlackScholesAgent, AgentResponse
from file_manager import FileManager


# Pydantic models for API
class OptionTypeEnum(str, Enum):
    CALL = "call"
    PUT = "put"


class OptionPriceRequest(BaseModel):
    S: float = Field(..., description="Spot price", gt=0)
    K: float = Field(..., description="Strike price", gt=0)
    T: float = Field(..., description="Time to expiration (years)", ge=0)
    r: float = Field(..., description="Risk-free rate (decimal)", ge=0, le=1)
    sigma: float = Field(..., description="Volatility (decimal)", gt=0, le=5)
    option_type: OptionTypeEnum = Field(..., description="Option type: call or put")
    dividend_yield: float = Field(0.0, description="Dividend yield (decimal)", ge=0)


class GreeksRequest(OptionPriceRequest):
    pass


class ImpliedVolatilityRequest(BaseModel):
    market_price: float = Field(..., description="Market price of option", gt=0)
    S: float = Field(..., description="Spot price", gt=0)
    K: float = Field(..., description="Strike price", gt=0)
    T: float = Field(..., description="Time to expiration (years)", ge=0)
    r: float = Field(..., description="Risk-free rate (decimal)", ge=0, le=1)
    option_type: OptionTypeEnum = Field(..., description="Option type: call or put")
    dividend_yield: float = Field(0.0, description="Dividend yield (decimal)", ge=0)


class MonteCarloRequest(OptionPriceRequest):
    num_simulations: int = Field(100000, description="Number of simulations", ge=1000, le=10000000)
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")


class AgentQueryRequest(BaseModel):
    query: str = Field(..., description="Natural language query")
    context: Optional[List[Dict]] = Field(None, description="Conversation context")


class PriceCurveRequest(BaseModel):
    params: OptionPriceRequest
    spot_range: Optional[List[float]] = Field(None, description="Spot price range [min, max]")
    num_points: int = Field(100, description="Number of points in curve", ge=10, le=1000)


# Initialize FastAPI app
app = FastAPI(
    title="Black-Scholes AI Agent API",
    description="Renaissance Technologies - Option Pricing with AI Agent",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models
bs_model = BlackScholesModel()
ai_agent = BlackScholesAgent(use_local_llm=True)
file_manager = FileManager(data_dir="data")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "status": "online",
        "service": "Black-Scholes AI Agent API",
        "version": "1.0.0",
        "endpoints": {
            "price": "/api/v1/price",
            "greeks": "/api/v1/greeks",
            "implied_volatility": "/api/v1/implied_volatility",
            "monte_carlo": "/api/v1/monte_carlo",
            "risk_metrics": "/api/v1/risk_metrics",
            "price_curve": "/api/v1/price_curve",
            "agent_query": "/api/v1/agent/query"
        }
    }


@app.post("/api/v1/price")
async def calculate_price(request: OptionPriceRequest, save: bool = False):
    """
    Calculate option price using Black-Scholes formula
    """
    try:
        params = OptionParams(
            S=request.S,
            K=request.K,
            T=request.T,
            r=request.r,
            sigma=request.sigma,
            option_type=OptionType(request.option_type.value),
            dividend_yield=request.dividend_yield
        )
        
        price = bs_model.price_option(params)
        
        result = {
            "success": True,
            "price": price,
            "parameters": {
                "S": params.S,
                "K": params.K,
                "T": params.T,
                "r": params.r,
                "sigma": params.sigma,
                "option_type": params.option_type.value,
                "dividend_yield": params.dividend_yield
            }
        }
        
        # Optionally save calculation
        if save:
            import uuid
            calc_id = f"price_{uuid.uuid4().hex[:8]}"
            file_manager.save_calculation(calc_id, result)
            result["calculation_id"] = calc_id
        
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/v1/greeks")
async def calculate_greeks(request: GreeksRequest):
    """
    Calculate option Greeks (delta, gamma, theta, vega, rho)
    """
    try:
        params = OptionParams(
            S=request.S,
            K=request.K,
            T=request.T,
            r=request.r,
            sigma=request.sigma,
            option_type=OptionType(request.option_type.value),
            dividend_yield=request.dividend_yield
        )
        
        greeks = bs_model.calculate_greeks(params)
        price = bs_model.price_option(params)
        
        return {
            "success": True,
            "price": price,
            "greeks": {
                "delta": greeks.delta,
                "gamma": greeks.gamma,
                "theta": greeks.theta,
                "vega": greeks.vega,
                "rho": greeks.rho
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/v1/implied_volatility")
async def calculate_implied_volatility(request: ImpliedVolatilityRequest):
    """
    Calculate implied volatility from market price
    """
    try:
        params = OptionParams(
            S=request.S,
            K=request.K,
            T=request.T,
            r=request.r,
            sigma=0.2,  # Dummy value, will be calculated
            option_type=OptionType(request.option_type.value),
            dividend_yield=request.dividend_yield
        )
        
        iv = bs_model.implied_volatility(request.market_price, params)
        
        if iv is None:
            raise HTTPException(
                status_code=400,
                detail="Could not calculate implied volatility. Market price may be inconsistent."
            )
        
        return {
            "success": True,
            "implied_volatility": iv,
            "implied_volatility_percent": iv * 100
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/v1/monte_carlo")
async def monte_carlo_price(request: MonteCarloRequest):
    """
    Price option using Monte Carlo simulation
    """
    try:
        params = OptionParams(
            S=request.S,
            K=request.K,
            T=request.T,
            r=request.r,
            sigma=request.sigma,
            option_type=OptionType(request.option_type.value),
            dividend_yield=request.dividend_yield
        )
        
        result = bs_model.monte_carlo_price(
            params,
            num_simulations=request.num_simulations,
            seed=request.seed
        )
        
        return {
            "success": True,
            **result
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/v1/risk_metrics")
async def calculate_risk_metrics(request: OptionPriceRequest):
    """
    Calculate comprehensive risk metrics
    """
    try:
        params = OptionParams(
            S=request.S,
            K=request.K,
            T=request.T,
            r=request.r,
            sigma=request.sigma,
            option_type=OptionType(request.option_type.value),
            dividend_yield=request.dividend_yield
        )
        
        metrics = bs_model.calculate_risk_metrics(params)
        
        return {
            "success": True,
            **metrics
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/v1/price_curve")
async def generate_price_curve(request: PriceCurveRequest):
    """
    Generate option price curve for different spot prices
    """
    try:
        params = OptionParams(
            S=request.params.S,
            K=request.params.K,
            T=request.params.T,
            r=request.params.r,
            sigma=request.params.sigma,
            option_type=OptionType(request.params.option_type.value),
            dividend_yield=request.params.dividend_yield
        )
        
        spot_range_tuple = None
        if request.spot_range and len(request.spot_range) == 2:
            spot_range_tuple = (request.spot_range[0], request.spot_range[1])
        
        curve = bs_model.generate_price_curve(
            params,
            spot_range=spot_range_tuple,
            num_points=request.num_points
        )
        
        return {
            "success": True,
            **curve
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/v1/agent/query")
async def agent_query(request: AgentQueryRequest):
    """
    Process natural language query through AI agent
    """
    try:
        response = ai_agent.process_query(request.query)
        
        return {
            "success": True,
            "message": response.message,
            "calculations": response.calculations,
            "confidence": response.confidence,
            "requires_clarification": response.requires_clarification
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/agent/tools")
async def get_agent_tools():
    """
    Get available agent tools
    """
    return {
        "success": True,
        "tools": ai_agent.tools
    }


@app.post("/api/v1/calculations/save")
async def save_calculation(
    calculation_id: Optional[str] = Body(None),
    data: Dict = Body(...)
):
    """
    Save a calculation to file
    """
    try:
        if calculation_id is None:
            import uuid
            calculation_id = f"calc_{uuid.uuid4().hex[:8]}"
        
        filepath = file_manager.save_calculation(calculation_id, data)
        return {
            "success": True,
            "calculation_id": calculation_id,
            "filepath": filepath
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/calculations/list")
async def list_calculations(limit: int = 10):
    """
    List recent calculations
    """
    try:
        calculations = file_manager.list_calculations(limit=limit)
        return {
            "success": True,
            "calculations": calculations,
            "count": len(calculations)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/calculations/{calculation_id}")
async def get_calculation(calculation_id: str):
    """
    Get a specific calculation by ID
    """
    try:
        calculation = file_manager.load_calculation(calculation_id)
        if calculation is None:
            raise HTTPException(status_code=404, detail="Calculation not found")
        return {
            "success": True,
            "calculation": calculation
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/export/csv")
async def export_calculations_to_csv(calculation_ids: List[str] = Body(...)):
    """
    Export calculations to CSV
    """
    try:
        calculations = []
        for calc_id in calculation_ids:
            calc = file_manager.load_calculation(calc_id)
            if calc:
                calculations.append(calc)
        
        if not calculations:
            raise HTTPException(status_code=404, detail="No calculations found")
        
        csv_path = file_manager.export_to_csv(calculations)
        return {
            "success": True,
            "filepath": csv_path,
            "count": len(calculations)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/stats")
async def get_statistics():
    """
    Get file manager statistics
    """
    try:
        stats = file_manager.get_statistics()
        return {
            "success": True,
            **stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

