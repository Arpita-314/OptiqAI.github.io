"""
Black-Scholes Model Implementation for Renaissance Technologies
Comprehensive option pricing with Greeks, Monte Carlo, and advanced features
"""

import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize_scalar
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
import json


class OptionType(Enum):
    CALL = "call"
    PUT = "put"


@dataclass
class OptionParams:
    """Option parameters"""
    S: float  # Spot price
    K: float  # Strike price
    T: float  # Time to expiration (years)
    r: float  # Risk-free rate
    sigma: float  # Volatility
    option_type: OptionType
    dividend_yield: float = 0.0  # Dividend yield


@dataclass
class OptionGreeks:
    """Option Greeks"""
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float


class BlackScholesModel:
    """
    Black-Scholes option pricing model with comprehensive features
    """
    
    def __init__(self):
        self.name = "Black-Scholes Model"
        self.version = "1.0.0"
    
    def calculate_d1_d2(self, S: float, K: float, T: float, r: float, 
                       sigma: float, q: float = 0.0) -> Tuple[float, float]:
        """Calculate d1 and d2 for Black-Scholes formula"""
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2
    
    def price_option(self, params: OptionParams) -> float:
        """
        Calculate option price using Black-Scholes formula
        
        Args:
            params: Option parameters
            
        Returns:
            Option price
        """
        S, K, T, r, sigma = params.S, params.K, params.T, params.r, params.sigma
        q = params.dividend_yield
        
        if T <= 0:
            # At expiration
            if params.option_type == OptionType.CALL:
                return max(S - K, 0)
            else:
                return max(K - S, 0)
        
        d1, d2 = self.calculate_d1_d2(S, K, T, r, sigma, q)
        
        if params.option_type == OptionType.CALL:
            price = (S * np.exp(-q * T) * stats.norm.cdf(d1) - 
                    K * np.exp(-r * T) * stats.norm.cdf(d2))
        else:  # PUT
            price = (K * np.exp(-r * T) * stats.norm.cdf(-d2) - 
                    S * np.exp(-q * T) * stats.norm.cdf(-d1))
        
        return max(price, 0)  # Ensure non-negative
    
    def calculate_greeks(self, params: OptionParams) -> OptionGreeks:
        """
        Calculate all option Greeks
        
        Args:
            params: Option parameters
            
        Returns:
            OptionGreeks object with all Greeks
        """
        S, K, T, r, sigma = params.S, params.K, params.T, params.r, params.sigma
        q = params.dividend_yield
        
        if T <= 0:
            # At expiration, Greeks are undefined or zero
            return OptionGreeks(delta=0, gamma=0, theta=0, vega=0, rho=0)
        
        d1, d2 = self.calculate_d1_d2(S, K, T, r, sigma, q)
        
        # Delta
        if params.option_type == OptionType.CALL:
            delta = np.exp(-q * T) * stats.norm.cdf(d1)
        else:
            delta = -np.exp(-q * T) * stats.norm.cdf(-d1)
        
        # Gamma (same for call and put)
        gamma = np.exp(-q * T) * stats.norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        # Theta
        if params.option_type == OptionType.CALL:
            theta = (-(S * stats.norm.pdf(d1) * sigma * np.exp(-q * T)) / (2 * np.sqrt(T)) -
                    r * K * np.exp(-r * T) * stats.norm.cdf(d2) +
                    q * S * np.exp(-q * T) * stats.norm.cdf(d1)) / 365
        else:
            theta = (-(S * stats.norm.pdf(d1) * sigma * np.exp(-q * T)) / (2 * np.sqrt(T)) +
                    r * K * np.exp(-r * T) * stats.norm.cdf(-d2) -
                    q * S * np.exp(-q * T) * stats.norm.cdf(-d1)) / 365
        
        # Vega (same for call and put)
        vega = S * np.exp(-q * T) * stats.norm.pdf(d1) * np.sqrt(T) / 100
        
        # Rho
        if params.option_type == OptionType.CALL:
            rho = K * T * np.exp(-r * T) * stats.norm.cdf(d2) / 100
        else:
            rho = -K * T * np.exp(-r * T) * stats.norm.cdf(-d2) / 100
        
        return OptionGreeks(delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)
    
    def implied_volatility(self, market_price: float, params: OptionParams, 
                          tolerance: float = 1e-6, max_iter: int = 100) -> Optional[float]:
        """
        Calculate implied volatility from market price
        
        Args:
            market_price: Observed market price of the option
            params: Option parameters (sigma will be ignored)
            tolerance: Convergence tolerance
            max_iter: Maximum iterations
            
        Returns:
            Implied volatility or None if not found
        """
        def objective(sigma):
            test_params = OptionParams(
                S=params.S, K=params.K, T=params.T, r=params.r,
                sigma=sigma, option_type=params.option_type,
                dividend_yield=params.dividend_yield
            )
            calculated_price = self.price_option(test_params)
            return abs(calculated_price - market_price)
        
        try:
            result = minimize_scalar(objective, bounds=(0.001, 5.0), method='bounded')
            if result.fun < tolerance:
                return result.x
        except:
            pass
        
        return None
    
    def monte_carlo_price(self, params: OptionParams, num_simulations: int = 100000,
                         seed: Optional[int] = None) -> Dict[str, float]:
        """
        Price option using Monte Carlo simulation
        
        Args:
            params: Option parameters
            num_simulations: Number of Monte Carlo paths
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with price, std error, and confidence intervals
        """
        if seed is not None:
            np.random.seed(seed)
        
        S, K, T, r, sigma = params.S, params.K, params.T, params.r, params.sigma
        q = params.dividend_yield
        
        # Generate random paths
        dt = T / 252  # Daily steps
        num_steps = max(1, int(T * 252))
        
        # Simulate stock price paths
        Z = np.random.standard_normal((num_simulations, num_steps))
        S_paths = np.zeros((num_simulations, num_steps + 1))
        S_paths[:, 0] = S
        
        for i in range(num_steps):
            S_paths[:, i + 1] = S_paths[:, i] * np.exp(
                (r - q - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z[:, i]
            )
        
        # Calculate payoffs
        final_prices = S_paths[:, -1]
        if params.option_type == OptionType.CALL:
            payoffs = np.maximum(final_prices - K, 0)
        else:
            payoffs = np.maximum(K - final_prices, 0)
        
        # Discount to present value
        option_prices = np.exp(-r * T) * payoffs
        
        price = np.mean(option_prices)
        std_error = np.std(option_prices) / np.sqrt(num_simulations)
        
        # 95% confidence interval
        confidence_interval = 1.96 * std_error
        
        return {
            "price": float(price),
            "std_error": float(std_error),
            "confidence_interval_95": [float(price - confidence_interval), 
                                      float(price + confidence_interval)],
            "min_price": float(np.min(option_prices)),
            "max_price": float(np.max(option_prices))
        }
    
    def calculate_risk_metrics(self, params: OptionParams) -> Dict[str, float]:
        """
        Calculate comprehensive risk metrics
        
        Args:
            params: Option parameters
            
        Returns:
            Dictionary of risk metrics
        """
        price = self.price_option(params)
        greeks = self.calculate_greeks(params)
        
        # Intrinsic and time value
        if params.option_type == OptionType.CALL:
            intrinsic = max(params.S - params.K, 0)
        else:
            intrinsic = max(params.K - params.S, 0)
        time_value = max(price - intrinsic, 0)
        
        # Moneyness
        moneyness = params.S / params.K
        
        # Leverage
        leverage = greeks.delta * params.S / price if price > 0 else 0
        
        return {
            "option_price": price,
            "intrinsic_value": intrinsic,
            "time_value": time_value,
            "moneyness": moneyness,
            "leverage": leverage,
            "delta": greeks.delta,
            "gamma": greeks.gamma,
            "theta": greeks.theta,
            "vega": greeks.vega,
            "rho": greeks.rho
        }
    
    def generate_price_curve(self, params: OptionParams, 
                            spot_range: Tuple[float, float] = None,
                            num_points: int = 100) -> Dict[str, List[float]]:
        """
        Generate option price curve for different spot prices
        
        Args:
            params: Base option parameters
            spot_range: (min_spot, max_spot) range, defaults to ±50% of current spot
            num_points: Number of points in the curve
            
        Returns:
            Dictionary with spot prices and corresponding option prices
        """
        if spot_range is None:
            spot_range = (params.S * 0.5, params.S * 1.5)
        
        spot_prices = np.linspace(spot_range[0], spot_range[1], num_points)
        option_prices = []
        
        for spot in spot_prices:
            test_params = OptionParams(
                S=spot, K=params.K, T=params.T, r=params.r,
                sigma=params.sigma, option_type=params.option_type,
                dividend_yield=params.dividend_yield
            )
            option_prices.append(self.price_option(test_params))
        
        return {
            "spot_prices": [float(s) for s in spot_prices],
            "option_prices": [float(p) for p in option_prices]
        }
    
    def to_dict(self, params: OptionParams) -> Dict:
        """Convert option parameters to dictionary"""
        return {
            "S": params.S,
            "K": params.K,
            "T": params.T,
            "r": params.r,
            "sigma": params.sigma,
            "option_type": params.option_type.value,
            "dividend_yield": params.dividend_yield
        }

