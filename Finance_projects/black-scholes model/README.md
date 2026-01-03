# Black-Scholes Model

A comprehensive implementation of the Black-Scholes option pricing model for Renaissance Technologies.

## Features

- **Option Pricing**: Calculate call and put option prices using the Black-Scholes formula
- **Greeks Calculation**: Delta, Gamma, Theta, Vega, and Rho
- **Implied Volatility**: Calculate implied volatility from market prices
- **Monte Carlo Simulation**: Alternative pricing method with confidence intervals
- **Risk Metrics**: Intrinsic value, time value, moneyness, leverage
- **Price Curves**: Generate option price curves for different spot prices

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from black_scholes_model import BlackScholesModel, OptionParams, OptionType

# Initialize model
model = BlackScholesModel()

# Define option parameters
params = OptionParams(
    S=100.0,      # Spot price
    K=100.0,      # Strike price
    T=0.25,       # Time to expiration (3 months)
    r=0.05,       # Risk-free rate (5%)
    sigma=0.2,    # Volatility (20%)
    option_type=OptionType.CALL
)

# Calculate option price
price = model.price_option(params)
print(f"Option Price: ${price:.4f}")

# Calculate Greeks
greeks = model.calculate_greeks(params)
print(f"Delta: {greeks.delta:.6f}")
print(f"Gamma: {greeks.gamma:.6f}")
print(f"Theta: {greeks.theta:.6f}")
print(f"Vega: {greeks.vega:.6f}")
print(f"Rho: {greeks.rho:.6f}")
```

## Examples

Run the example script:

```bash
python example.py
```

This will demonstrate:
- Option pricing
- Greeks calculation
- Risk metrics
- Implied volatility
- Monte Carlo simulation
- Put option pricing

## API Reference

### BlackScholesModel

#### `price_option(params: OptionParams) -> float`
Calculate option price using Black-Scholes formula.

#### `calculate_greeks(params: OptionParams) -> OptionGreeks`
Calculate all option Greeks (Delta, Gamma, Theta, Vega, Rho).

#### `implied_volatility(market_price: float, params: OptionParams, tolerance: float = 1e-6, max_iter: int = 100) -> Optional[float]`
Calculate implied volatility from market price.

#### `monte_carlo_price(params: OptionParams, num_simulations: int = 100000, seed: Optional[int] = None) -> Dict[str, float]`
Price option using Monte Carlo simulation.

#### `calculate_risk_metrics(params: OptionParams) -> Dict[str, float]`
Calculate comprehensive risk metrics including intrinsic value, time value, moneyness, and leverage.

#### `generate_price_curve(params: OptionParams, spot_range: Tuple[float, float] = None, num_points: int = 100) -> Dict[str, List[float]]`
Generate option price curve for different spot prices.

### OptionParams

```python
@dataclass
class OptionParams:
    S: float              # Spot price
    K: float              # Strike price
    T: float              # Time to expiration (years)
    r: float              # Risk-free rate
    sigma: float          # Volatility
    option_type: OptionType  # CALL or PUT
    dividend_yield: float = 0.0  # Dividend yield (optional)
```

### OptionType

```python
class OptionType(Enum):
    CALL = "call"
    PUT = "put"
```

## Mathematical Background

The Black-Scholes model assumes:
- Constant volatility
- Constant risk-free rate
- No dividends (or constant dividend yield)
- Lognormal distribution of stock prices
- No transaction costs
- Continuous trading

The Black-Scholes formula for a call option:

```
C = S₀e^(-qT)N(d₁) - Ke^(-rT)N(d₂)
```

Where:
- `d₁ = [ln(S₀/K) + (r - q + σ²/2)T] / (σ√T)`
- `d₂ = d₁ - σ√T`
- `N(x)` is the cumulative standard normal distribution

## License

Renaissance Technologies - Internal Use

## Contributing

This is a specialized system for Renaissance Technologies. For modifications, please contact the development team.

