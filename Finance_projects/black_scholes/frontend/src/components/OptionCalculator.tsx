/**
 * Option Calculator Component
 */

import { useState } from 'react';
import { apiClient, OptionPriceRequest, GreeksResponse, RiskMetricsResponse } from '../api/client';

export default function OptionCalculator() {
  const [params, setParams] = useState<OptionPriceRequest>({
    S: 100,
    K: 100,
    T: 0.25,
    r: 0.05,
    sigma: 0.2,
    option_type: 'call',
    dividend_yield: 0.0,
  });

  const [results, setResults] = useState<{
    price?: number;
    greeks?: GreeksResponse['greeks'];
    riskMetrics?: RiskMetricsResponse;
  }>({});

  const [loading, setLoading] = useState(false);

  const handleInputChange = (field: keyof OptionPriceRequest, value: any) => {
    setParams((prev) => ({ ...prev, [field]: value }));
  };

  const calculatePrice = async () => {
    setLoading(true);
    try {
      const response = await apiClient.calculatePrice(params);
      setResults((prev) => ({ ...prev, price: response.price }));
    } catch (error) {
      alert(`Error: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setLoading(false);
    }
  };

  const calculateGreeks = async () => {
    setLoading(true);
    try {
      const response = await apiClient.calculateGreeks(params);
      setResults((prev) => ({
        ...prev,
        price: response.price,
        greeks: response.greeks,
      }));
    } catch (error) {
      alert(`Error: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setLoading(false);
    }
  };

  const calculateRiskMetrics = async () => {
    setLoading(true);
    try {
      const response = await apiClient.calculateRiskMetrics(params);
      setResults((prev) => ({ ...prev, riskMetrics: response }));
    } catch (error) {
      alert(`Error: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <h2 className="text-2xl font-bold mb-6">Option Calculator</h2>

      <div className="grid grid-cols-2 gap-4 mb-6">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Spot Price (S)
          </label>
          <input
            type="number"
            value={params.S}
            onChange={(e) => handleInputChange('S', parseFloat(e.target.value))}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Strike Price (K)
          </label>
          <input
            type="number"
            value={params.K}
            onChange={(e) => handleInputChange('K', parseFloat(e.target.value))}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Time to Expiration (T, years)
          </label>
          <input
            type="number"
            step="0.01"
            value={params.T}
            onChange={(e) => handleInputChange('T', parseFloat(e.target.value))}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Risk-Free Rate (r, decimal)
          </label>
          <input
            type="number"
            step="0.001"
            value={params.r}
            onChange={(e) => handleInputChange('r', parseFloat(e.target.value))}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Volatility (σ, decimal)
          </label>
          <input
            type="number"
            step="0.01"
            value={params.sigma}
            onChange={(e) => handleInputChange('sigma', parseFloat(e.target.value))}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Option Type
          </label>
          <select
            value={params.option_type}
            onChange={(e) => handleInputChange('option_type', e.target.value as 'call' | 'put')}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="call">Call</option>
            <option value="put">Put</option>
          </select>
        </div>
      </div>

      <div className="flex space-x-3 mb-6">
        <button
          onClick={calculatePrice}
          disabled={loading}
          className="px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 disabled:opacity-50"
        >
          Calculate Price
        </button>
        <button
          onClick={calculateGreeks}
          disabled={loading}
          className="px-4 py-2 bg-green-500 text-white rounded-md hover:bg-green-600 disabled:opacity-50"
        >
          Calculate Greeks
        </button>
        <button
          onClick={calculateRiskMetrics}
          disabled={loading}
          className="px-4 py-2 bg-purple-500 text-white rounded-md hover:bg-purple-600 disabled:opacity-50"
        >
          Risk Metrics
        </button>
      </div>

      {results.price !== undefined && (
        <div className="mt-6 p-4 bg-gray-50 rounded-md">
          <h3 className="text-lg font-semibold mb-3">Results</h3>
          <div className="text-2xl font-bold text-blue-600 mb-4">
            Option Price: ${results.price.toFixed(2)}
          </div>

          {results.greeks && (
            <div className="mt-4">
              <h4 className="font-semibold mb-2">Greeks</h4>
              <div className="grid grid-cols-2 gap-2 text-sm">
                <div>Delta: {results.greeks.delta.toFixed(4)}</div>
                <div>Gamma: {results.greeks.gamma.toFixed(4)}</div>
                <div>Theta: {results.greeks.theta.toFixed(4)}</div>
                <div>Vega: {results.greeks.vega.toFixed(4)}</div>
                <div>Rho: {results.greeks.rho.toFixed(4)}</div>
              </div>
            </div>
          )}

          {results.riskMetrics && (
            <div className="mt-4">
              <h4 className="font-semibold mb-2">Risk Metrics</h4>
              <div className="grid grid-cols-2 gap-2 text-sm">
                <div>Intrinsic Value: ${results.riskMetrics.intrinsic_value.toFixed(2)}</div>
                <div>Time Value: ${results.riskMetrics.time_value.toFixed(2)}</div>
                <div>Moneyness: {results.riskMetrics.moneyness.toFixed(2)}</div>
                <div>Leverage: {results.riskMetrics.leverage.toFixed(2)}</div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

