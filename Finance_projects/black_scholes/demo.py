"""
Demo script to showcase Black-Scholes AI Agent
Demonstrates proper file handling and system capabilities
"""

import sys
import os
import uuid
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from black_scholes_model import BlackScholesModel, OptionParams, OptionType
from ai_agent import BlackScholesAgent
from file_manager import FileManager


def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def demo_basic_calculations():
    """Demonstrate basic Black-Scholes calculations"""
    print_section("Basic Black-Scholes Calculations")
    
    model = BlackScholesModel()
    
    # Example 1: Call option
    print("Example 1: Call Option")
    params1 = OptionParams(
        S=100,      # Spot price
        K=105,      # Strike price
        T=0.25,     # 3 months
        r=0.05,     # 5% risk-free rate
        sigma=0.2,  # 20% volatility
        option_type=OptionType.CALL
    )
    
    price1 = model.price_option(params1)
    greeks1 = model.calculate_greeks(params1)
    
    print(f"  Spot Price (S): ${params1.S}")
    print(f"  Strike Price (K): ${params1.K}")
    print(f"  Time to Expiration (T): {params1.T} years ({params1.T*365:.0f} days)")
    print(f"  Risk-Free Rate (r): {params1.r*100:.1f}%")
    print(f"  Volatility (sigma): {params1.sigma*100:.1f}%")
    print(f"\n  => Option Price: ${price1:.2f}")
    print(f"  => Delta: {greeks1.delta:.4f}")
    print(f"  => Gamma: {greeks1.gamma:.4f}")
    print(f"  => Theta: {greeks1.theta:.4f} (per day)")
    print(f"  => Vega: {greeks1.vega:.4f}")
    print(f"  => Rho: {greeks1.rho:.4f}")
    
    # Example 2: Put option
    print("\nExample 2: Put Option")
    params2 = OptionParams(
        S=100,
        K=95,
        T=0.25,
        r=0.05,
        sigma=0.25,
        option_type=OptionType.PUT
    )
    
    price2 = model.price_option(params2)
    risk_metrics = model.calculate_risk_metrics(params2)
    
    print(f"  => Put Option Price: ${price2:.2f}")
    print(f"  => Intrinsic Value: ${risk_metrics['intrinsic_value']:.2f}")
    print(f"  => Time Value: ${risk_metrics['time_value']:.2f}")
    print(f"  => Moneyness: {risk_metrics['moneyness']:.2f}")
    
    return [params1, params2], [price1, price2]


def demo_ai_agent():
    """Demonstrate AI agent capabilities"""
    print_section("AI Agent - Natural Language Queries")
    
    agent = BlackScholesAgent(use_local_llm=False)  # Use fallback for demo
    
    queries = [
        "What's the price of a call option with strike 100, spot 105, 30 days to expiration, 5% risk-free rate, and 20% volatility?",
        "Calculate the Greeks for a put option: S=100, K=95, T=0.25, r=0.05, sigma=0.25",
        "What's the implied volatility if the market price is $5.50 for a call option with S=100, K=100, T=0.25, r=0.05?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 70)
        
        response = agent.process_query(query)
        print(f"Response: {response.message}")
        
        if response.calculations:
            print("\nCalculations:")
            for key, value in response.calculations.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for k, v in value.items():
                        if isinstance(v, float):
                            print(f"    {k}: {v:.4f}")
                        else:
                            print(f"    {k}: {v}")
                elif isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")


def demo_file_handling():
    """Demonstrate file handling capabilities"""
    print_section("File Handling - Data Persistence")
    
    file_manager = FileManager(data_dir="demo_data")
    
    # Save some calculations
    model = BlackScholesModel()
    
    calculations = []
    for i in range(3):
        params = OptionParams(
            S=100 + i*5,
            K=100,
            T=0.25,
            r=0.05,
            sigma=0.2 + i*0.05,
            option_type=OptionType.CALL
        )
        
        price = model.price_option(params)
        greeks = model.calculate_greeks(params)
        risk_metrics = model.calculate_risk_metrics(params)
        
        calc_data = {
            "parameters": {
                "S": params.S,
                "K": params.K,
                "T": params.T,
                "r": params.r,
                "sigma": params.sigma,
                "option_type": params.option_type.value
            },
            "price": price,
            "greeks": {
                "delta": greeks.delta,
                "gamma": greeks.gamma,
                "theta": greeks.theta,
                "vega": greeks.vega,
                "rho": greeks.rho
            },
            "risk_metrics": risk_metrics
        }
        
        calc_id = f"calc_{i+1}_{uuid.uuid4().hex[:8]}"
        filepath = file_manager.save_calculation(calc_id, calc_data)
        calculations.append({"calculation_id": calc_id, "data": calc_data})
        print(f"Saved calculation {i+1}: {os.path.basename(filepath)}")
    
    # List calculations
    print("\nRecent Calculations:")
    recent = file_manager.list_calculations(limit=5)
    for calc in recent:
        print(f"  - {calc['calculation_id']}: {calc['summary']} ({calc['timestamp']})")
    
    # Export to CSV
    print("\nExporting to CSV...")
    csv_path = file_manager.export_to_csv(calculations)
    print(f"Exported to: {csv_path}")
    
    # Save conversation history
    session_id = f"demo_{uuid.uuid4().hex[:8]}"
    messages = [
        {"role": "user", "content": "What's the price?", "timestamp": datetime.now().isoformat()},
        {"role": "assistant", "content": "The price is $5.23", "timestamp": datetime.now().isoformat()}
    ]
    history_path = file_manager.save_conversation_history(session_id, messages)
    print(f"\nSaved conversation history: {os.path.basename(history_path)}")
    
    # Statistics
    stats = file_manager.get_statistics()
    print("\nFile Manager Statistics:")
    print(f"  Total Calculations: {stats['total_calculations']}")
    print(f"  Total Sessions: {stats['total_sessions']}")
    print(f"  Total Exports: {stats['total_exports']}")
    print(f"  Total Size: {stats['total_size_mb']} MB")
    print(f"  Data Directory: {stats['data_directory']}")


def demo_advanced_features():
    """Demonstrate advanced features"""
    print_section("Advanced Features")
    
    model = BlackScholesModel()
    
    # Monte Carlo simulation
    print("Monte Carlo Simulation:")
    params = OptionParams(
        S=100,
        K=100,
        T=0.25,
        r=0.05,
        sigma=0.2,
        option_type=OptionType.CALL
    )
    
    mc_result = model.monte_carlo_price(params, num_simulations=50000, seed=42)
    print(f"  Price: ${mc_result['price']:.2f}")
    print(f"  Standard Error: ${mc_result['std_error']:.4f}")
    print(f"  95% Confidence Interval: ${mc_result['confidence_interval_95'][0]:.2f} - ${mc_result['confidence_interval_95'][1]:.2f}")
    
    # Implied Volatility
    print("\nImplied Volatility Calculation:")
    market_price = 5.23
    iv = model.implied_volatility(market_price, params)
    if iv:
        print(f"  Market Price: ${market_price:.2f}")
        print(f"  Implied Volatility: {iv*100:.2f}%")
    else:
        print("  Could not calculate implied volatility")
    
    # Price Curve
    print("\nPrice Curve Generation:")
    curve = model.generate_price_curve(params, spot_range=(80, 120), num_points=20)
    print(f"  Generated {len(curve['spot_prices'])} points")
    print(f"  Spot Range: ${min(curve['spot_prices']):.2f} - ${max(curve['spot_prices']):.2f}")
    print(f"  Price Range: ${min(curve['option_prices']):.2f} - ${max(curve['option_prices']):.2f}")


def main():
    """Run all demos"""
    print("\n" + "="*70)
    print("  BLACK-SCHOLES AI AGENT DEMO")
    print("  Renaissance Technologies - Option Pricing System")
    print("="*70)
    
    try:
        # Run demos
        demo_basic_calculations()
        demo_ai_agent()
        demo_file_handling()
        demo_advanced_features()
        
        print_section("Demo Complete!")
        print("All features demonstrated successfully.")
        print("\nTo run the full system:")
        print("  1. Backend: python backend_api.py")
        print("  2. Frontend: cd frontend && npm install && npm run dev")
        print("  3. Open: http://localhost:3000")
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

