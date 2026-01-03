"""
Example usage of the Black-Scholes Model
"""

from black_scholes_model import BlackScholesModel, OptionParams, OptionType

def main():
    # Initialize the model
    model = BlackScholesModel()
    
    # Example 1: Calculate option price
    print("=" * 60)
    print("Example 1: Calculate Call Option Price")
    print("=" * 60)
    
    params = OptionParams(
        S=100.0,      # Spot price
        K=100.0,      # Strike price
        T=0.25,       # Time to expiration (3 months)
        r=0.05,       # Risk-free rate (5%)
        sigma=0.2,    # Volatility (20%)
        option_type=OptionType.CALL
    )
    
    price = model.price_option(params)
    print(f"Option Price: ${price:.4f}")
    print()
    
    # Example 2: Calculate Greeks
    print("=" * 60)
    print("Example 2: Calculate Option Greeks")
    print("=" * 60)
    
    greeks = model.calculate_greeks(params)
    print(f"Delta:  {greeks.delta:.6f}")
    print(f"Gamma:  {greeks.gamma:.6f}")
    print(f"Theta:  {greeks.theta:.6f} (per day)")
    print(f"Vega:   {greeks.vega:.6f} (per 1% vol change)")
    print(f"Rho:    {greeks.rho:.6f} (per 1% rate change)")
    print()
    
    # Example 3: Risk Metrics
    print("=" * 60)
    print("Example 3: Risk Metrics")
    print("=" * 60)
    
    risk_metrics = model.calculate_risk_metrics(params)
    print(f"Option Price:      ${risk_metrics['option_price']:.4f}")
    print(f"Intrinsic Value:   ${risk_metrics['intrinsic_value']:.4f}")
    print(f"Time Value:        ${risk_metrics['time_value']:.4f}")
    print(f"Moneyness:         {risk_metrics['moneyness']:.4f}")
    print(f"Leverage:          {risk_metrics['leverage']:.4f}")
    print()
    
    # Example 4: Implied Volatility
    print("=" * 60)
    print("Example 4: Calculate Implied Volatility")
    print("=" * 60)
    
    market_price = 4.6150  # Observed market price
    iv = model.implied_volatility(market_price, params)
    if iv:
        print(f"Market Price: ${market_price:.4f}")
        print(f"Implied Volatility: {iv:.4f} ({iv*100:.2f}%)")
    else:
        print("Could not calculate implied volatility")
    print()
    
    # Example 5: Monte Carlo Simulation
    print("=" * 60)
    print("Example 5: Monte Carlo Pricing (10,000 simulations)")
    print("=" * 60)
    
    mc_result = model.monte_carlo_price(params, num_simulations=10000, seed=42)
    print(f"MC Price: ${mc_result['price']:.4f}")
    print(f"Standard Error: ${mc_result['std_error']:.4f}")
    print(f"95% CI: [${mc_result['confidence_interval_95'][0]:.4f}, ${mc_result['confidence_interval_95'][1]:.4f}]")
    print()
    
    # Example 6: Put Option
    print("=" * 60)
    print("Example 6: Put Option")
    print("=" * 60)
    
    put_params = OptionParams(
        S=100.0,
        K=100.0,
        T=0.25,
        r=0.05,
        sigma=0.2,
        option_type=OptionType.PUT
    )
    
    put_price = model.price_option(put_params)
    put_greeks = model.calculate_greeks(put_params)
    
    print(f"Put Option Price: ${put_price:.4f}")
    print(f"Put Delta: {put_greeks.delta:.6f}")
    print()

if __name__ == "__main__":
    main()

