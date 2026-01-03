// Tab Management Module
const Tabs = {
    currentTab: 'dashboard',
    
    init: function() {
        this.bindEvents();
    },
    
    bindEvents: function() {
        // Delegate tab switching to parent element
        document.querySelector('.tab-header').addEventListener('click', (e) => {
            if (e.target.classList.contains('tab-button')) {
                const tabName = e.target.getAttribute('data-tab');
                this.switchTab(tabName);
            }
        });
    },
    
    switchTab: function(tabName) {
        // Update tab buttons
        document.querySelectorAll('.tab-button').forEach(btn => {
            btn.classList.remove('active');
        });
        
        // Activate clicked tab button
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
        
        // Update tab panels
        document.querySelectorAll('.tab-panel').forEach(panel => {
            panel.classList.remove('active');
        });
        document.getElementById(`${tabName}-tab`).classList.add('active');
        
        this.currentTab = tabName;
        App.updateStatus(`Switched to ${tabName} tab`);
        
        // Trigger tab-specific actions
        if (tabName === 'results') {
            Visualization.drawResults();
        }
    }
};