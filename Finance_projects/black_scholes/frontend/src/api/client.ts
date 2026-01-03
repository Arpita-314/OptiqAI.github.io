/**
 * API Client for Black-Scholes Backend
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export interface OptionPriceRequest {
  S: number;
  K: number;
  T: number;
  r: number;
  sigma: number;
  option_type: 'call' | 'put';
  dividend_yield?: number;
}

export interface AgentQueryRequest {
  query: string;
  context?: Array<Record<string, any>>;
}

export interface AgentResponse {
  success: boolean;
  message: string;
  calculations?: Record<string, any>;
  confidence: number;
  requires_clarification: boolean;
}

export interface PriceResponse {
  success: boolean;
  price: number;
  parameters: OptionPriceRequest;
}

export interface GreeksResponse {
  success: boolean;
  price: number;
  greeks: {
    delta: number;
    gamma: number;
    theta: number;
    vega: number;
    rho: number;
  };
}

export interface RiskMetricsResponse {
  success: boolean;
  option_price: number;
  intrinsic_value: number;
  time_value: number;
  moneyness: number;
  leverage: number;
  delta: number;
  gamma: number;
  theta: number;
  vega: number;
  rho: number;
}

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(error.detail || 'Request failed');
    }

    return response.json();
  }

  async calculatePrice(params: OptionPriceRequest): Promise<PriceResponse> {
    return this.request<PriceResponse>('/api/v1/price', {
      method: 'POST',
      body: JSON.stringify(params),
    });
  }

  async calculateGreeks(params: OptionPriceRequest): Promise<GreeksResponse> {
    return this.request<GreeksResponse>('/api/v1/greeks', {
      method: 'POST',
      body: JSON.stringify(params),
    });
  }

  async calculateRiskMetrics(params: OptionPriceRequest): Promise<RiskMetricsResponse> {
    return this.request<RiskMetricsResponse>('/api/v1/risk_metrics', {
      method: 'POST',
      body: JSON.stringify(params),
    });
  }

  async agentQuery(request: AgentQueryRequest): Promise<AgentResponse> {
    return this.request<AgentResponse>('/api/v1/agent/query', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  async getAgentTools() {
    return this.request('/api/v1/agent/tools');
  }
}

export const apiClient = new ApiClient();

