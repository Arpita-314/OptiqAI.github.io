import { useState } from 'react';
import ChatInterface from './components/ChatInterface';
import OptionCalculator from './components/OptionCalculator';
import './App.css';

function App() {
  const [activeTab, setActiveTab] = useState<'chat' | 'calculator'>('chat');

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Header */}
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <h1 className="text-3xl font-bold text-gray-900">
            Black-Scholes AI Agent
          </h1>
          <p className="text-gray-600 mt-1">Renaissance Technologies - Option Pricing System</p>
        </div>
      </header>

      {/* Tabs */}
      <div className="max-w-7xl mx-auto px-4 py-4">
        <div className="flex space-x-4 border-b border-gray-200">
          <button
            onClick={() => setActiveTab('chat')}
            className={`px-4 py-2 font-medium ${
              activeTab === 'chat'
                ? 'text-blue-600 border-b-2 border-blue-600'
                : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            AI Agent Chat
          </button>
          <button
            onClick={() => setActiveTab('calculator')}
            className={`px-4 py-2 font-medium ${
              activeTab === 'calculator'
                ? 'text-blue-600 border-b-2 border-blue-600'
                : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            Option Calculator
          </button>
        </div>
      </div>

      {/* Content */}
      <main className="max-w-7xl mx-auto px-4 pb-8">
        {activeTab === 'chat' ? (
          <div className="bg-white rounded-lg shadow-lg" style={{ height: 'calc(100vh - 200px)' }}>
            <ChatInterface />
          </div>
        ) : (
          <OptionCalculator />
        )}
      </main>
    </div>
  );
}

export default App;

