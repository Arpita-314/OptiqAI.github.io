/**
 * Chat Interface Component for AI Agent
 */

import { useState, useRef, useEffect } from 'react';
import { apiClient, AgentQueryRequest, AgentResponse } from '../api/client';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  calculations?: Record<string, any>;
  timestamp: Date;
}

export default function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || loading) return;

    const userMessage: Message = {
      role: 'user',
      content: input,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const context = messages.map((m) => ({
        query: m.role === 'user' ? m.content : '',
        response: m.role === 'assistant' ? m.content : '',
      }));

      const request: AgentQueryRequest = {
        query: input,
        context,
      };

      const response: AgentResponse = await apiClient.agentQuery(request);

      const assistantMessage: Message = {
        role: 'assistant',
        content: response.message,
        calculations: response.calculations,
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      const errorMessage: Message = {
        role: 'assistant',
        content: `Error: ${error instanceof Error ? error.message : 'Unknown error'}`,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="flex flex-col h-full bg-gray-50">
      {/* Chat Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && (
          <div className="text-center text-gray-500 mt-8">
            <h3 className="text-lg font-semibold mb-2">Black-Scholes AI Agent</h3>
            <p>Ask me anything about option pricing!</p>
            <p className="text-sm mt-2">Try: "What's the price of a call option with strike 100, spot 105, 30 days to expiration?"</p>
          </div>
        )}

        {messages.map((message, index) => (
          <div
            key={index}
            className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-3xl rounded-lg px-4 py-2 ${
                message.role === 'user'
                  ? 'bg-blue-500 text-white'
                  : 'bg-white text-gray-800 shadow-md'
              }`}
            >
              <div className="whitespace-pre-wrap">{message.content}</div>
              {message.calculations && (
                <div className="mt-2 pt-2 border-t border-gray-200">
                  <CalculationDisplay calculations={message.calculations} />
                </div>
              )}
              <div className="text-xs opacity-70 mt-1">
                {message.timestamp.toLocaleTimeString()}
              </div>
            </div>
          </div>
        ))}

        {loading && (
          <div className="flex justify-start">
            <div className="bg-white rounded-lg px-4 py-2 shadow-md">
              <div className="flex space-x-2">
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }}></div>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="border-t bg-white p-4">
        <div className="flex space-x-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask about option pricing, Greeks, risk metrics..."
            className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            disabled={loading}
          />
          <button
            onClick={handleSend}
            disabled={loading || !input.trim()}
            className="px-6 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}

function CalculationDisplay({ calculations }: { calculations: Record<string, any> }) {
  if (calculations.price !== undefined) {
    return (
      <div className="text-sm">
        <div className="font-semibold">Price: ${calculations.price.toFixed(2)}</div>
        {calculations.greeks && (
          <div className="mt-1 space-y-1">
            <div>Delta: {calculations.greeks.delta?.toFixed(4)}</div>
            <div>Gamma: {calculations.greeks.gamma?.toFixed(4)}</div>
            <div>Theta: {calculations.greeks.theta?.toFixed(4)}</div>
            <div>Vega: {calculations.greeks.vega?.toFixed(4)}</div>
            <div>Rho: {calculations.greeks.rho?.toFixed(4)}</div>
          </div>
        )}
      </div>
    );
  }

  if (calculations.implied_volatility !== undefined) {
    return (
      <div className="text-sm">
        <div className="font-semibold">
          Implied Volatility: {(calculations.implied_volatility * 100).toFixed(2)}%
        </div>
      </div>
    );
  }

  if (calculations.option_price !== undefined) {
    return (
      <div className="text-sm space-y-1">
        <div className="font-semibold">Option Price: ${calculations.option_price.toFixed(2)}</div>
        <div>Intrinsic Value: ${calculations.intrinsic_value?.toFixed(2)}</div>
        <div>Time Value: ${calculations.time_value?.toFixed(2)}</div>
        <div>Moneyness: {calculations.moneyness?.toFixed(2)}</div>
      </div>
    );
  }

  return null;
}

