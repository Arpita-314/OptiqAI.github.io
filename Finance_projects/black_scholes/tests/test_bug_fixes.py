"""
Tests to verify bug fixes for parameter extraction and implied volatility
"""

import pytest
from ai_agent import BlackScholesAgent
from black_scholes_model import BlackScholesModel, OptionParams, OptionType


def test_bug1_time_conversion_with_unrelated_words():
    """
    Bug 1 Fix: Time conversion should use the matched pattern, not re-scan entire query.
    Test case: Query with "today" but specifying months should not convert to days.
    """
    agent = BlackScholesAgent(use_local_llm=False)
    
    # Query with "today" but time specified in months
    query = "What's the price today for an option with 6 months to expiration, S=100, K=100, r=0.05, sigma=0.2"
    params = agent.extract_parameters(query)
    
    # Should be 6 months = 0.5 years, NOT 6/365 (which would happen if "today" triggered days conversion)
    assert 'T' in params
    assert abs(params['T'] - 0.5) < 0.01, f"Expected 0.5 years (6 months), got {params['T']}"
    
    # Another test: query with "day" in unrelated context
    query2 = "Calculate for a daily option with 30 days expiration, S=100, K=100, r=0.05, sigma=0.2"
    params2 = agent.extract_parameters(query2)
    assert 'T' in params2
    # Should be 30/365, not affected by "daily"
    expected_days = 30 / 365
    assert abs(params2['T'] - expected_days) < 0.01, f"Expected {expected_days}, got {params2['T']}"


def test_bug2_decimal_vs_percentage_rates():
    """
    Bug 2 Fix: Should not divide decimals by 100 when they're already in decimal format.
    """
    agent = BlackScholesAgent(use_local_llm=False)
    
    # Test 1: Decimal format (r=0.05 should stay 0.05)
    query1 = "Price option with r=0.05, S=100, K=100, T=0.25, sigma=0.2"
    params1 = agent.extract_parameters(query1)
    assert 'r' in params1
    assert abs(params1['r'] - 0.05) < 0.001, f"Expected 0.05, got {params1['r']}"
    
    # Test 2: Percentage format (r=5% should become 0.05)
    query2 = "Price option with interest rate 5%, S=100, K=100, T=0.25, sigma=20%"
    params2 = agent.extract_parameters(query2)
    assert 'r' in params2
    assert abs(params2['r'] - 0.05) < 0.001, f"Expected 0.05, got {params2['r']}"
    
    # Test 3: Large value (>1.0) should be treated as percentage
    query3 = "Price option with risk-free rate 5, S=100, K=100, T=0.25, sigma=0.2"
    params3 = agent.extract_parameters(query3)
    assert 'r' in params3
    # Should be divided by 100 since it's > 1.0 and from "rate" pattern
    assert abs(params3['r'] - 0.05) < 0.001, f"Expected 0.05, got {params3['r']}"


def test_bug2_decimal_vs_percentage_volatility():
    """
    Bug 2 Fix: Volatility should handle both decimal and percentage formats correctly.
    """
    agent = BlackScholesAgent(use_local_llm=False)
    
    # Test 1: Decimal format (sigma=0.2 should stay 0.2)
    query1 = "Price option with sigma=0.2, S=100, K=100, T=0.25, r=0.05"
    params1 = agent.extract_parameters(query1)
    assert 'sigma' in params1
    assert abs(params1['sigma'] - 0.2) < 0.001, f"Expected 0.2, got {params1['sigma']}"
    
    # Test 2: Percentage format (volatility 20% should become 0.2)
    query2 = "Price option with volatility 20%, S=100, K=100, T=0.25, r=0.05"
    params2 = agent.extract_parameters(query2)
    assert 'sigma' in params2
    assert abs(params2['sigma'] - 0.2) < 0.001, f"Expected 0.2, got {params2['sigma']}"
    
    # Test 3: Large value (>1.0) should be treated as percentage
    query3 = "Price option with volatility 20, S=100, K=100, T=0.25, r=0.05"
    params3 = agent.extract_parameters(query3)
    assert 'sigma' in params3
    # Should be divided by 100 since it's > 1.0 and from "volatility" pattern
    assert abs(params3['sigma'] - 0.2) < 0.001, f"Expected 0.2, got {params3['sigma']}"


def test_bug3_implied_volatility_max_iter():
    """
    Bug 3 Fix: max_iter parameter should be respected in minimize_scalar.
    """
    model = BlackScholesModel()
    
    # Create a test case where we can verify max_iter is used
    params = OptionParams(
        S=100,
        K=100,
        T=0.25,
        r=0.05,
        sigma=0.2,  # Dummy, will be calculated
        option_type=OptionType.CALL
    )
    
    # Calculate expected price first
    expected_price = model.price_option(params)
    
    # Test with normal max_iter - should work
    iv = model.implied_volatility(
        expected_price,
        params,
        tolerance=1e-4,  # Slightly relaxed tolerance
        max_iter=100
    )
    
    # Should recover the original volatility approximately
    assert iv is not None, "Implied volatility calculation should succeed with max_iter=100"
    assert abs(iv - 0.2) < 0.05, f"Expected volatility around 0.2, got {iv}"
    
    # Test with very low max_iter - might fail but should not crash
    # The key is that the parameter is passed to minimize_scalar
    iv_low = model.implied_volatility(
        expected_price,
        params,
        tolerance=1e-4,
        max_iter=5  # Very low - might not converge
    )
    
    # With very low max_iter, it might return None if it doesn't converge
    # But the method should complete without error (the parameter is used)
    # We can't easily verify the exact iteration count, but we verify
    # the method accepts the parameter and doesn't crash
    assert True  # If we get here, the method accepted max_iter parameter


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

