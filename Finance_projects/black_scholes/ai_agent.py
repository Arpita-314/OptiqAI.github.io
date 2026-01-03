"""
AI Agent for Black-Scholes Model
Integrates with private LLM for natural language interaction
"""

import json
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers not available. Using fallback mode.")

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from black_scholes_model import BlackScholesModel, OptionParams, OptionType


class AgentMode(Enum):
    """Agent operation modes"""
    LOCAL_LLM = "local_llm"
    FALLBACK = "fallback"


@dataclass
class AgentResponse:
    """Agent response structure"""
    message: str
    calculations: Optional[Dict] = None
    confidence: float = 1.0
    requires_clarification: bool = False


class BlackScholesAgent:
    """
    AI Agent that understands natural language queries about Black-Scholes options pricing
    """
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium", 
                 use_local_llm: bool = True, device: str = "cpu"):
        """
        Initialize the AI agent
        
        Args:
            model_name: HuggingFace model name for local LLM
            use_local_llm: Whether to use local LLM or fallback mode
            device: Device to run model on ('cpu' or 'cuda')
        """
        self.bs_model = BlackScholesModel()
        self.use_local_llm = use_local_llm and HAS_TRANSFORMERS
        self.device = device
        self.conversation_history: List[Dict] = []
        
        if self.use_local_llm:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(model_name)
                self.model.to(device)
                self.model.eval()
                self.llm_pipeline = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=0 if device == "cuda" else -1
                )
            except Exception as e:
                print(f"Warning: Could not load LLM model: {e}")
                self.use_local_llm = False
        
        # Tool definitions for the agent
        self.tools = {
            "price_option": {
                "description": "Calculate option price using Black-Scholes formula",
                "parameters": ["S", "K", "T", "r", "sigma", "option_type"]
            },
            "calculate_greeks": {
                "description": "Calculate option Greeks (delta, gamma, theta, vega, rho)",
                "parameters": ["S", "K", "T", "r", "sigma", "option_type"]
            },
            "implied_volatility": {
                "description": "Calculate implied volatility from market price",
                "parameters": ["market_price", "S", "K", "T", "r", "option_type"]
            },
            "monte_carlo": {
                "description": "Price option using Monte Carlo simulation",
                "parameters": ["S", "K", "T", "r", "sigma", "option_type", "num_simulations"]
            },
            "risk_metrics": {
                "description": "Calculate comprehensive risk metrics",
                "parameters": ["S", "K", "T", "r", "sigma", "option_type"]
            }
        }
    
    def extract_parameters(self, query: str) -> Dict[str, Any]:
        """
        Extract option parameters from natural language query
        
        Args:
            query: Natural language query
            
        Returns:
            Dictionary of extracted parameters
        """
        params = {}
        query_lower = query.lower()
        
        # Extract numbers with context
        # Spot price
        spot_patterns = [
            r'spot\s*(?:price|value)?\s*(?:is|of|:)?\s*\$?(\d+\.?\d*)',
            r'stock\s*(?:price|value)?\s*(?:is|of|:)?\s*\$?(\d+\.?\d*)',
            r'current\s*(?:price|value)?\s*(?:is|of|:)?\s*\$?(\d+\.?\d*)',
            r'S\s*[=:]\s*\$?(\d+\.?\d*)'
        ]
        for pattern in spot_patterns:
            match = re.search(pattern, query_lower)
            if match:
                params['S'] = float(match.group(1))
                break
        
        # Strike price
        strike_patterns = [
            r'strike\s*(?:price|value)?\s*(?:is|of|:)?\s*\$?(\d+\.?\d*)',
            r'K\s*[=:]\s*\$?(\d+\.?\d*)'
        ]
        for pattern in strike_patterns:
            match = re.search(pattern, query_lower)
            if match:
                params['K'] = float(match.group(1))
                break
        
        # Time to expiration
        time_patterns = [
            r'(\d+\.?\d*)\s*(?:days?|d)',
            r'(\d+\.?\d*)\s*(?:months?|m)',
            r'(\d+\.?\d*)\s*(?:years?|y)',
            r'T\s*[=:]\s*(\d+\.?\d*)',
            r'time\s*(?:to|until)?\s*expiration\s*(?:is|:)?\s*(\d+\.?\d*)'
        ]
        for pattern in time_patterns:
            match = re.search(pattern, query_lower)
            if match:
                value = float(match.group(1))
                # Convert to years
                if 'day' in query_lower or 'd' in query_lower:
                    value = value / 365
                elif 'month' in query_lower or 'm' in query_lower:
                    value = value / 12
                params['T'] = value
                break
        
        # Risk-free rate
        rate_patterns = [
            r'risk[-\s]free\s*rate\s*(?:is|of|:)?\s*(\d+\.?\d*)\s*%?',
            r'interest\s*rate\s*(?:is|of|:)?\s*(\d+\.?\d*)\s*%?',
            r'r\s*[=:]\s*(\d+\.?\d*)\s*%?'
        ]
        for pattern in rate_patterns:
            match = re.search(pattern, query_lower)
            if match:
                params['r'] = float(match.group(1)) / 100
                break
        
        # Volatility
        vol_patterns = [
            r'volatility\s*(?:is|of|:)?\s*(\d+\.?\d*)\s*%?',
            r'sigma\s*[=:]\s*(\d+\.?\d*)',
            r'σ\s*[=:]\s*(\d+\.?\d*)'
        ]
        for pattern in vol_patterns:
            match = re.search(pattern, query_lower)
            if match:
                params['sigma'] = float(match.group(1)) / 100
                break
        
        # Option type
        if 'call' in query_lower:
            params['option_type'] = 'call'
        elif 'put' in query_lower:
            params['option_type'] = 'put'
        
        # Market price (for implied volatility)
        market_price_patterns = [
            r'market\s*price\s*(?:is|of|:)?\s*\$?(\d+\.?\d*)',
            r'current\s*price\s*(?:is|of|:)?\s*\$?(\d+\.?\d*)'
        ]
        for pattern in market_price_patterns:
            match = re.search(pattern, query_lower)
            if match:
                params['market_price'] = float(match.group(1))
                break
        
        return params
    
    def generate_response(self, query: str, context: Optional[List[Dict]] = None) -> str:
        """
        Generate response using LLM
        
        Args:
            query: User query
            context: Conversation context
            
        Returns:
            Generated response
        """
        if not self.use_local_llm:
            return self._fallback_response(query)
        
        try:
            # Build prompt with context
            prompt = self._build_prompt(query, context)
            
            # Generate response
            response = self.llm_pipeline(
                prompt,
                max_length=200,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True
            )
            
            return response[0]['generated_text'].replace(prompt, "").strip()
        except Exception as e:
            print(f"LLM generation error: {e}")
            return self._fallback_response(query)
    
    def _build_prompt(self, query: str, context: Optional[List[Dict]] = None) -> str:
        """Build prompt for LLM"""
        system_prompt = """You are an expert financial analyst specializing in Black-Scholes option pricing.
You help users calculate option prices, Greeks, and analyze risk metrics.
Be concise, accurate, and professional in your responses."""
        
        prompt = system_prompt + "\n\n"
        
        if context:
            for msg in context[-3:]:  # Last 3 messages for context
                prompt += f"User: {msg.get('query', '')}\n"
                prompt += f"Assistant: {msg.get('response', '')}\n\n"
        
        prompt += f"User: {query}\nAssistant:"
        return prompt
    
    def _fallback_response(self, query: str) -> str:
        """Fallback response when LLM is not available"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['price', 'value', 'cost']):
            return "I can help you calculate the option price. Please provide: spot price, strike price, time to expiration, risk-free rate, and volatility."
        elif any(word in query_lower for word in ['greeks', 'delta', 'gamma', 'theta', 'vega']):
            return "I can calculate the option Greeks (delta, gamma, theta, vega, rho). Please provide the option parameters."
        elif 'implied' in query_lower or 'volatility' in query_lower:
            return "I can calculate implied volatility. Please provide the market price and option parameters."
        else:
            return "I'm here to help with Black-Scholes option pricing. What would you like to calculate?"
    
    def process_query(self, query: str) -> AgentResponse:
        """
        Process a natural language query and return response with calculations
        
        Args:
            query: Natural language query
            
        Returns:
            AgentResponse with message and calculations
        """
        # Extract parameters
        extracted_params = self.extract_parameters(query)
        
        # Determine what the user wants
        query_lower = query.lower()
        requires_clarification = False
        calculations = None
        
        # Check if we have enough parameters for basic calculations
        required_params = ['S', 'K', 'T', 'r', 'sigma', 'option_type']
        missing_params = [p for p in required_params if p not in extracted_params]
        
        if missing_params and any(word in query_lower for word in ['price', 'calculate', 'value']):
            requires_clarification = True
            missing_str = ", ".join(missing_params)
            message = f"I need more information to calculate. Missing: {missing_str}"
        else:
            # Perform calculations
            if extracted_params:
                try:
                    option_type = OptionType(extracted_params.get('option_type', 'call'))
                    params = OptionParams(
                        S=extracted_params.get('S', 100),
                        K=extracted_params.get('K', 100),
                        T=extracted_params.get('T', 0.25),
                        r=extracted_params.get('r', 0.05),
                        sigma=extracted_params.get('sigma', 0.2),
                        option_type=option_type,
                        dividend_yield=extracted_params.get('dividend_yield', 0.0)
                    )
                    
                    # Determine calculation type
                    if 'greeks' in query_lower or any(g in query_lower for g in ['delta', 'gamma', 'theta', 'vega', 'rho']):
                        greeks = self.bs_model.calculate_greeks(params)
                        price = self.bs_model.price_option(params)
                        calculations = {
                            "price": price,
                            "greeks": {
                                "delta": greeks.delta,
                                "gamma": greeks.gamma,
                                "theta": greeks.theta,
                                "vega": greeks.vega,
                                "rho": greeks.rho
                            }
                        }
                        message = f"Option price: ${price:.2f}\nGreeks: Delta={greeks.delta:.4f}, Gamma={greeks.gamma:.4f}, Theta={greeks.theta:.4f}, Vega={greeks.vega:.4f}, Rho={greeks.rho:.4f}"
                    
                    elif 'implied' in query_lower and 'market_price' in extracted_params:
                        iv = self.bs_model.implied_volatility(
                            extracted_params['market_price'], params
                        )
                        if iv:
                            calculations = {"implied_volatility": iv}
                            message = f"Implied volatility: {iv*100:.2f}%"
                        else:
                            message = "Could not calculate implied volatility. The market price may be inconsistent with the Black-Scholes model."
                    
                    elif 'monte' in query_lower or 'simulation' in query_lower:
                        num_sims = extracted_params.get('num_simulations', 100000)
                        mc_result = self.bs_model.monte_carlo_price(params, num_sims)
                        calculations = mc_result
                        message = f"Monte Carlo price: ${mc_result['price']:.2f} (95% CI: ${mc_result['confidence_interval_95'][0]:.2f} - ${mc_result['confidence_interval_95'][1]:.2f})"
                    
                    elif 'risk' in query_lower or 'metrics' in query_lower:
                        risk_metrics = self.bs_model.calculate_risk_metrics(params)
                        calculations = risk_metrics
                        message = f"Risk Metrics:\nPrice: ${risk_metrics['option_price']:.2f}\nIntrinsic: ${risk_metrics['intrinsic_value']:.2f}\nTime Value: ${risk_metrics['time_value']:.2f}\nMoneyness: {risk_metrics['moneyness']:.2f}"
                    
                    else:
                        # Default: calculate price
                        price = self.bs_model.price_option(params)
                        calculations = {"price": price}
                        message = f"The option price is ${price:.2f}"
                    
                except Exception as e:
                    message = f"Error in calculation: {str(e)}"
                    requires_clarification = True
            else:
                # Generate LLM response
                message = self.generate_response(query, self.conversation_history)
                requires_clarification = True
        
        # Add to conversation history
        self.conversation_history.append({
            "query": query,
            "response": message,
            "calculations": calculations
        })
        
        return AgentResponse(
            message=message,
            calculations=calculations,
            confidence=0.9 if calculations else 0.5,
            requires_clarification=requires_clarification
        )

