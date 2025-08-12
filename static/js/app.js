 // River Water Quality Management System - Main JavaScript

class RiverManagementApp {
    constructor() {
        this.network = null;
        this.graphData = { nodes: [], relationships: [] };
        this.refreshInterval = null;
        this.isInitialized = false;
        
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadInitialData();
        this.startAutoRefresh();
    }

    setupEventListeners() {
        // Smooth scrolling for navigation
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', (e) => {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            });
        });

        // Form submissions
        const predictionForm = document.getElementById('prediction-form');
        if (predictionForm) {
            predictionForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.predictTrade();
            });
        }

        // Alert threshold changes
        const alertThreshold = document.getElementById('alert-threshold');
        if (alertThreshold) {
            alertThreshold.addEventListener('change', () => {
                this.checkAlerts();
            });
        }

        // Initialize database button
        const initButton = document.querySelector('a[href="/initialize-database"]');
        if (initButton) {
            initButton.addEventListener('click', (e) => {
                e.preventDefault();
                this.initializeDatabase();
            });
        }

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey || e.metaKey) {
                switch (e.key) {
                    case 'r':
                        e.preventDefault();
                        this.refreshData();
                        break;
                    case 'd':
                        e.preventDefault();
                        window.location.href = '/dashboard';
                        break;
                    case 'h':
                        e.preventDefault();
                        window.location.href = '/';
                        break;
                }
            }
        });
    }

    async loadInitialData() {
        try {
            await Promise.all([
                this.loadGraphData(),
                this.loadWaterQualityStats(),
                this.checkAlerts()
            ]);
            this.isInitialized = true;
            this.showNotification('System loaded successfully!', 'success');
        } catch (error) {
            console.error('Error loading initial data:', error);
            this.showNotification('Error loading system data', 'error');
        }
    }

    async loadGraphData() {
        const loading = document.getElementById('graph-loading');
        const container = document.getElementById('graph-container');
        
        if (loading) loading.style.display = 'block';
        if (container) container.style.display = 'none';

        try {
            const response = await this.apiCall('GET', '/api/graph-data');
            this.graphData = response.data;
            
            // Update statistics
            this.updateStatistics();
            
            // Create network visualization
            this.createNetwork();
            
            if (loading) loading.style.display = 'none';
            if (container) container.style.display = 'block';
        } catch (error) {
            console.error('Error loading graph data:', error);
            if (loading) {
                loading.innerHTML = '<p class="text-danger">Error loading graph data</p>';
            }
        }
    }

    updateStatistics() {
        const totalNodes = document.getElementById('total-nodes');
        const totalRelationships = document.getElementById('total-relationships');
        
        if (totalNodes) totalNodes.textContent = this.graphData.nodes.length;
        if (totalRelationships) totalRelationships.textContent = this.graphData.relationships.length;
    }

    createNetwork() {
        const container = document.getElementById('graph-container');
        if (!container || !this.graphData.nodes.length) return;
        
        // Create nodes
        const nodes = new vis.DataSet(this.graphData.nodes.map(node => ({
            id: node.id,
            label: node.label,
            color: node.color,
            size: node.size || 10,
            title: this.createNodeTooltip(node.properties),
            font: {
                size: 12,
                color: '#333'
            },
            borderWidth: 2,
            shadow: true
        })));

        // Create edges
        const edges = new vis.DataSet(this.graphData.relationships.map(rel => ({
            from: rel.source,
            to: rel.target,
            label: rel.type,
            arrows: 'to',
            color: { color: '#666', opacity: 0.6 },
            width: 2,
            shadow: true,
            font: {
                size: 10,
                color: '#666'
            }
        })));

        // Network options
        const options = {
            nodes: {
                shape: 'circle',
                scaling: {
                    min: 5,
                    max: 30
                }
            },
            edges: {
                smooth: {
                    type: 'continuous',
                    forceDirection: 'none'
                }
            },
            physics: {
                enabled: true,
                barnesHut: {
                    gravitationalConstant: -2000,
                    springConstant: 0.04,
                    springLength: 200,
                    damping: 0.09
                },
                stabilization: {
                    enabled: true,
                    iterations: 1000,
                    updateInterval: 100
                }
            },
            interaction: {
                hover: true,
                zoomView: true,
                dragView: true,
                navigationButtons: true,
                keyboard: true
            },
            layout: {
                improvedLayout: true,
                hierarchical: false
            }
        };

        // Create network
        this.network = new vis.Network(container, { nodes, edges }, options);
        
        // Add event listeners
        this.network.on('click', (params) => {
            if (params.nodes.length > 0) {
                this.showNodeDetails(params.nodes[0]);
            }
        });

        this.network.on('stabilizationProgress', (params) => {
            // Show loading progress
            if (params.iterations < params.total) {
                const progress = Math.round((params.iterations / params.total) * 100);
                this.updateLoadingProgress(progress);
            }
        });

        this.network.on('stabilizationIterationsDone', () => {
            this.hideLoadingProgress();
        });
    }

    createNodeTooltip(properties) {
        if (!properties) return '';
        
        let tooltip = '<div style="font-weight: bold; margin-bottom: 10px;">Node Properties</div>';
        
        Object.entries(properties).forEach(([key, value]) => {
            if (value !== null && value !== undefined) {
                tooltip += `<div><strong>${key}:</strong> ${value}</div>`;
            }
        });
        
        return tooltip;
    }

    showNodeDetails(nodeId) {
        const node = this.graphData.nodes.find(n => n.id === nodeId.toString());
        if (!node) return;

        const modal = this.createModal('Node Details', this.createNodeDetailsHTML(node));
        document.body.appendChild(modal);
        
        // Auto-remove modal after 5 seconds
        setTimeout(() => {
            if (modal.parentNode) {
                modal.parentNode.removeChild(modal);
            }
        }, 5000);
    }

    createNodeDetailsHTML(node) {
        return `
            <div class="node-details">
                <h5>${node.label} Node</h5>
                <div class="row">
                    ${Object.entries(node.properties || {}).map(([key, value]) => `
                        <div class="col-md-6 mb-2">
                            <strong>${key}:</strong> ${value}
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    }

    async loadWaterQualityStats() {
        try {
            const response = await this.apiCall('GET', '/api/water-quality-stats');
            const stats = response.data;
            
            const avgNh3 = document.getElementById('avg-nh3');
            if (avgNh3 && stats.avg_nh3 !== null) {
                avgNh3.textContent = stats.avg_nh3.toFixed(2);
            }
        } catch (error) {
            console.error('Error loading water quality stats:', error);
        }
    }

    async checkAlerts() {
        const threshold = document.getElementById('alert-threshold')?.value || 1.0;
        const container = document.getElementById('alerts-container');
        
        try {
            const response = await this.apiCall('GET', `/api/water-quality-alerts?threshold=${threshold}`);
            const alerts = response.data;
            
            const alertsCount = document.getElementById('alerts-count');
            if (alertsCount) alertsCount.textContent = alerts.length;
            
            if (container) {
                if (alerts.length === 0) {
                    container.innerHTML = `
                        <div class="alert alert-success">
                            <i class="fas fa-check-circle me-2"></i>
                            No water quality alerts at this time.
                        </div>
                    `;
                } else {
                    let alertsHtml = '';
                    alerts.forEach(alert => {
                        alertsHtml += `
                            <div class="alert alert-danger">
                                <i class="fas fa-exclamation-triangle me-2"></i>
                                <strong>Segment ${alert.id}</strong>: NH3 concentration ${alert.nh3_concentration.toFixed(2)} mg/L 
                                (threshold: ${alert.threshold} mg/L)
                            </div>
                        `;
                    });
                    container.innerHTML = alertsHtml;
                }
            }
        } catch (error) {
            console.error('Error checking alerts:', error);
            if (container) {
                container.innerHTML = `
                    <div class="alert alert-warning">
                        <i class="fas fa-exclamation-circle me-2"></i>
                        Error checking water quality alerts.
                    </div>
                `;
            }
        }
    }

    async predictTrade() {
        const buyer = document.getElementById('buyer-segment')?.value;
        const seller = document.getElementById('seller-segment')?.value;
        const resultDiv = document.getElementById('prediction-result');
        const amountSpan = document.getElementById('predicted-amount');
        
        if (!buyer || !seller) {
            this.showNotification('Please enter both buyer and seller segment numbers', 'warning');
            return;
        }
        
        try {
            const response = await this.apiCall('POST', '/api/predict-trade', {
                buyer: parseInt(buyer),
                seller: parseInt(seller)
            });
            
            const prediction = response.data.prediction;
            if (amountSpan) amountSpan.textContent = prediction;
            if (resultDiv) resultDiv.style.display = 'block';
            
            this.showNotification(`Prediction: ${prediction} WEC units`, 'info');
        } catch (error) {
            console.error('Error predicting trade:', error);
            this.showNotification('Error predicting trade amount', 'error');
        }
    }

    async initializeDatabase() {
        try {
            this.showNotification('Initializing database...', 'info');
            
            const response = await this.apiCall('GET', '/initialize-database');
            
            // Reload data after initialization
            await this.loadGraphData();
            await this.loadWaterQualityStats();
            
            this.showNotification('Database initialized successfully!', 'success');
        } catch (error) {
            console.error('Error initializing database:', error);
            this.showNotification('Error initializing database', 'error');
        }
    }

    async refreshData() {
        this.showNotification('Refreshing data...', 'info');
        await this.loadInitialData();
        this.showNotification('Data refreshed successfully!', 'success');
    }

    startAutoRefresh() {
        // Refresh data every 5 minutes
        this.refreshInterval = setInterval(() => {
            if (this.isInitialized) {
                this.loadWaterQualityStats();
                this.checkAlerts();
            }
        }, 300000); // 5 minutes
    }

    stopAutoRefresh() {
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
            this.refreshInterval = null;
        }
    }

    async apiCall(method, url, data = null) {
        const config = {
            method,
            url,
            headers: {
                'Content-Type': 'application/json'
            }
        };

        if (data) {
            config.data = data;
        }

        return axios(config);
    }

    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
        notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
        notification.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;

        document.body.appendChild(notification);

        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 5000);
    }

    createModal(title, content) {
        const modal = document.createElement('div');
        modal.className = 'modal fade show';
        modal.style.display = 'block';
        modal.innerHTML = `
            <div class="modal-dialog">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">${title}</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        ${content}
                    </div>
                </div>
            </div>
        `;

        // Add backdrop
        const backdrop = document.createElement('div');
        backdrop.className = 'modal-backdrop fade show';
        document.body.appendChild(backdrop);

        // Close functionality
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                this.closeModal(modal, backdrop);
            }
        });

        const closeBtn = modal.querySelector('.btn-close');
        if (closeBtn) {
            closeBtn.addEventListener('click', () => {
                this.closeModal(modal, backdrop);
            });
        }

        return modal;
    }

    closeModal(modal, backdrop) {
        modal.remove();
        if (backdrop && backdrop.parentNode) {
            backdrop.parentNode.removeChild(backdrop);
        }
    }

    updateLoadingProgress(progress) {
        let progressBar = document.getElementById('loading-progress');
        if (!progressBar) {
            progressBar = document.createElement('div');
            progressBar.id = 'loading-progress';
            progressBar.className = 'progress position-fixed';
            progressBar.style.cssText = 'top: 0; left: 0; right: 0; height: 3px; z-index: 9999;';
            progressBar.innerHTML = '<div class="progress-bar" style="width: 0%"></div>';
            document.body.appendChild(progressBar);
        }
        
        const progressBarInner = progressBar.querySelector('.progress-bar');
        progressBarInner.style.width = `${progress}%`;
    }

    hideLoadingProgress() {
        const progressBar = document.getElementById('loading-progress');
        if (progressBar && progressBar.parentNode) {
            progressBar.parentNode.removeChild(progressBar);
        }
    }

    // Utility methods
    formatNumber(num) {
        return new Intl.NumberFormat().format(num);
    }

    formatDate(date) {
        return new Intl.DateTimeFormat().format(date);
    }

    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    throttle(func, limit) {
        let inThrottle;
        return function() {
            const args = arguments;
            const context = this;
            if (!inThrottle) {
                func.apply(context, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.riverApp = new RiverManagementApp();
});

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = RiverManagementApp;
}