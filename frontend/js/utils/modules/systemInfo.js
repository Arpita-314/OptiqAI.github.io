// System Information Module
const SystemInfo = {
    init: function() {
        this.detectSystem();
    },
    
    detectSystem: function() {
        // Browser detection
        const userAgent = navigator.userAgent;
        let browser = 'Unknown';
        if (userAgent.indexOf('Chrome') > -1) browser = 'Chrome';
        else if (userAgent.indexOf('Safari') > -1) browser = 'Safari';
        else if (userAgent.indexOf('Firefox') > -1) browser = 'Firefox';
        else if (userAgent.indexOf('Edge') > -1) browser = 'Edge';
        
        document.getElementById('browser').textContent = browser;
        
        // Screen resolution
        document.getElementById('screen').textContent = `${window.screen.width}x${window.screen.height}`;
        
        // WebGL detection
        const canvas = document.createElement('canvas');
        const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
        if (gl) {
            const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
            if (debugInfo) {
                const renderer = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL);
                document.getElementById('gpu-renderer').textContent = renderer;
            }
            document.getElementById('webgl').textContent = 'Enabled';
        } else {
            document.getElementById('webgl').textContent = 'Disabled';
            document.getElementById('cuda-status').className = 'cuda-status cuda-unavailable';
            document.getElementById('cuda-status').innerHTML = `
                <div style="color: var(--error);">
                    <b>❌ WebGL Not Available</b><br>
                    <b>Status:</b> Limited functionality<br>
                    <b>Performance:</b> CPU mode only
                </div>
            `;
        }
        
        // Memory detection
        if (navigator.deviceMemory) {
            document.getElementById('memory').textContent = `${navigator.deviceMemory} GB`;
        }
    }
};