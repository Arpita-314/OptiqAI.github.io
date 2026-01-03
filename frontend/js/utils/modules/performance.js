// Performance Monitoring Module
const Performance = {
    init: function() {
        // Start performance monitoring
        this.updatePerformanceMonitors();
        setInterval(() => {
            this.updatePerformanceMonitors();
        }, 2000);
    },
    
    updatePerformanceMonitors: function() {
        // Simulate CPU usage
        const cpu = Helpers.randomInRange(40, 70);
        document.getElementById('cpu-value').textContent = `${cpu}%`;
        document.getElementById('cpu-progress').style.width = `${cpu}%`;
        document.getElementById('cpu-progress').textContent = `${cpu}%`;
        
        // Simulate memory usage
        const memory = Helpers.randomInRange(50, 80);
        document.getElementById('mem-value').textContent = `${memory}%`;
        document.getElementById('memory-progress').style.width = `${memory}%`;
        document.getElementById('memory-progress').textContent = `${memory}%`;
        
        // Simulate GPU temperature
        const gpu = Helpers.randomInRange(35, 45);
        document.getElementById('gpu-value').textContent = `${gpu}°C`;
        document.getElementById('gpu-progress').style.width = `${gpu}%`;
        document.getElementById('gpu-progress').textContent = `${gpu}°C`;
    }
};