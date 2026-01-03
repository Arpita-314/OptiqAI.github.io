"""
Tests for Black-Scholes Model
"""

import pytest
import numpy as np
from black_scholes_model import (
    BlackScholesModel, OptionParams, OptionType
)


def test_call_option_price():
    """Test call option pricing"""
    model = BlackScholesModel()
    params = OptionParams(
        S=100,
        K=100,
        T=0.25,
        r=0.05,
        sigma=0.2,
        option_type=OptionType.CALL
    )
    
    price = model.price_option(params)
    assert price > 0
    assert isinstance(price, (float, np.floating))


def test_put_option_price():
    """Test put option pricing"""
    model = BlackScholesModel()
    params = OptionParams(
        S=100,
        K=100,
        T=0.25,
        r=0.05,
        sigma=0.2,
        option_type=OptionType.PUT
    )
    
    price = model.price_option(params)
    assert price > 0


def test_greeks_calculation():
    """Test Greeks calculation"""
    model = BlackScholesModel()
    params = OptionParams(
        S=100,
        K=100,
        T=0.25,
        r=0.05,
        sigma=0.2,
        option_type=OptionType.CALL
    )
    
    greeks = model.calculate_greeks(params)
    assert -1 <= greeks.delta <= 1
    assert greeks.gamma >= 0
    assert isinstance(greeks.theta, (int, float))
    assert greeks.vega >= 0
    assert isinstance(greeks.rho, (int, float))


def test_put_call_parity():
    """Test put-call parity"""
    model = BlackScholesModel()
    
    S, K, T, r, sigma = 100, 100, 0.25, 0.05, 0.2
    
    call_params = OptionParams(S=S, K=K, T=T, r=r, sigma=sigma, option_type=OptionType.CALL)
    put_params = OptionParams(S=S, K=K, T=T, r=r, sigma=sigma, option_type=OptionType.PUT)
    
    call_price = model.price_option(call_params)
    put_price = model.price_option(put_params)
    
    # Put-call parity: C - P = S - K*e^(-r*T)
    lhs = call_price - put_price
    rhs = S - K * np.exp(-r * T)
    
    assert abs(lhs - rhs) < 0.01  # Allow small numerical error


def test_at_expiration():
    """Test option pricing at expiration"""
    model = BlackScholesModel()
    
    # In-the-money call
    params = OptionParams(
        S=110,
        K=100,
        T=0.0,  # At expiration
        r=0.05,
        sigma=0.2,
        option_type=OptionType.CALL
    )
    price = model.price_option(params)
    assert abs(price - 10.0) < 0.01  # Intrinsic value
    
    # Out-of-the-money call
    params.S = 90
    price = model.price_option(params)
    assert abs(price - 0.0) < 0.01


def test_monte_carlo():
    """Test Monte Carlo pricing"""
    model = BlackScholesModel()
    params = OptionParams(
        S=100,
        K=100,
        T=0.25,
        r=0.05,
        sigma=0.2,
        option_type=OptionType.CALL
    )
    
    result = model.monte_carlo_price(params, num_simulations=10000, seed=42)
    
    assert 'price' in result
    assert 'std_error' in result
    assert 'confidence_interval_95' in result
    assert result['price'] > 0
    assert result['std_error'] >= 0


def test_risk_metrics():
    """Test risk metrics calculation"""
    model = BlackScholesModel()
    params = OptionParams(
        S=100,
        K=100,
        T=0.25,
        r=0.05,
        sigma=0.2,
        option_type=OptionType.CALL
    )
    
    metrics = model.calculate_risk_metrics(params)
    
    assert 'option_price' in metrics
    assert 'intrinsic_value' in metrics
    assert 'time_value' in metrics
    assert 'moneyness' in metrics
    assert 'leverage' in metrics
    assert metrics['option_price'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

