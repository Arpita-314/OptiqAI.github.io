// Main Application Module
const App = {
    init: function() {
        this.loadComponents();
        this.bindEvents();
        this.initializeModules();
        this.updateTime();
        setInterval(() => {
            this.updateTime();
        }, 1000);
    },
    
    loadComponents: function() {
        // Load components
        Helpers.loadComponent('sidebar-container', 'components/sidebar.html');
        Helpers.loadComponent('header-container', 'components/header.html');
        Helpers.loadComponent('tabs-container', 'components/tabs.html');
    },
    
    bindEvents: function() {
        // Delegate control button events
        document.addEventListener('click', (e) => {
            if (e.target.id === 'initialize-engine-btn') {
                this.initializeEngine();
            } else if (e.target.id === 'configure-settings-btn') {
                this.configureSettings();
            } else if (e.target.id === 'view-results-btn') {
                this.viewResults();
            }
        });
    },
    
    initializeModules: function() {
        // Initialize all modules
        SystemInfo.init();
        Tabs.init();
        Analysis.init();
        Visualization.init();
        Performance.init();
    },
    
    initializeEngine: function() {
        this.showLoading(true);
        this.updateStatus('Initializing processing engine...');
        Analysis.appendToAnalysis('🚀 Engine initialization started...');
        
        setTimeout(() => {
            Analysis.appendToAnalysis('✓ Core modules loaded');
            Analysis.appendToAnalysis('✓ WebGL context created');
            Analysis.appendToAnalysis('✓ Signal processors ready');
            Analysis.appendToAnalysis('✅ Engine initialized successfully!');
            this.updateStatus('Engine ready');
            this.showLoading(false);
        }, 2000);
    },
    
    configureSettings: function() {
        Tabs.switchTab('settings');
        this.updateStatus('Configuration panel opened');
    },
    
    viewResults: function() {
        Tabs.switchTab('results');
        this.updateStatus('Results viewer opened');
        Visualization.drawResults();
    },
    
    updateStatus: function(message) {
        document.getElementById('status-text').textContent = message;
    },
    
    showLoading: function(show) {
        document.getElementById('status-spinner').style.display = show ? 'inline-block' : 'none';
    },
    
    updateTime: function() {
        const now = new Date();
        document.getElementById('status-time').textContent = `🕒 ${now.toLocaleTimeString()}`;
    }
};

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    App.init();
});