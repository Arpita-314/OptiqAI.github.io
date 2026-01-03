// Helper functions
const Helpers = {
    // DOM manipulation helpers
    createElement: (tag, className, content) => {
        const element = document.createElement(tag);
        if (className) element.className = className;
        if (content) element.innerHTML = content;
        return element;
    },
    
    // Load HTML components
    loadComponent: async (containerId, url) => {
        try {
            const response = await fetch(url);
            const html = await response.text();
            document.getElementById(containerId).innerHTML = html;
        } catch (error) {
            console.error(`Error loading component: ${url}`, error);
        }
    },
    
    // Format numbers
    formatBytes: (bytes, decimals = 2) => {
        if (bytes === 0) return '0 Bytes';
        
        const k = 1024;
        const dm = decimals < 0 ? 0 : decimals;
        const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
        
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        
        return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
    },
    
    // Generate random number in range
    randomInRange: (min, max) => {
        return Math.floor(Math.random() * (max - min + 1)) + min;
    }
};