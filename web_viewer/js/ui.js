/**
 * ui.js - User Interface Management
 * 
 * Handles all UI elements, image cross-fade, and user interactions
 */

class UIManager {
    constructor(config) {
        this.config = config;
        
        // Get DOM elements
        this.loadingScreen = document.getElementById('loading-screen');
        this.progressFill = document.getElementById('progress-fill');
        this.loadingStatus = document.getElementById('loading-status');
        
        this.currentImage = document.getElementById('current-image');
        this.nextImage = document.getElementById('next-image');
        
        this.cameraLabel = document.getElementById('camera-label');
        this.cameraInfo = document.getElementById('camera-info');
        
        this.togglePointsBtn = document.getElementById('toggle-points');
        this.toggleNodesBtn = document.getElementById('toggle-nodes');
        this.helpBtn = document.getElementById('help-btn');
        
        this.navHint = document.getElementById('nav-hint');
        this.helpModal = document.getElementById('help-modal');
        this.cameraList = document.getElementById('camera-list');
        
        // State
        this.pointsVisible = false;
        this.nodesVisible = true;
        
        // Callbacks (set by main viewer)
        this.onTogglePoints = null;
        this.onToggleNodes = null;
        this.onCameraSelect = null;
        
        // Setup event listeners
        this.setupEventListeners();
    }

    /**
     * Setup all UI event listeners
     */
    setupEventListeners() {
        // Toggle points button
        this.togglePointsBtn.addEventListener('click', () => {
            this.pointsVisible = !this.pointsVisible;
            this.togglePointsBtn.textContent = this.pointsVisible ? 'Hide Points' : 'Show Points';
            
            if (this.onTogglePoints) {
                this.onTogglePoints(this.pointsVisible);
            }
            
            Utils.log('Point cloud toggled', { visible: this.pointsVisible });
        });

        // Toggle navigation nodes button
        this.toggleNodesBtn.addEventListener('click', () => {
            this.nodesVisible = !this.nodesVisible;
            this.toggleNodesBtn.textContent = this.nodesVisible ? 'Hide Navigation' : 'Show Navigation';
            
            if (this.onToggleNodes) {
                this.onToggleNodes(this.nodesVisible);
            }
            
            Utils.log('Navigation nodes toggled', { visible: this.nodesVisible });
        });

        // Help button
        this.helpBtn.addEventListener('click', () => {
            this.helpModal.classList.add('active');
        });

        // Help modal close button
        const closeBtn = this.helpModal.querySelector('.close');
        closeBtn.addEventListener('click', () => {
            this.helpModal.classList.remove('active');
        });

        // Close modal on outside click
        this.helpModal.addEventListener('click', (e) => {
            if (e.target === this.helpModal) {
                this.helpModal.classList.remove('active');
            }
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            switch(e.key) {
                case 'h':
                case 'H':
                    this.helpModal.classList.toggle('active');
                    break;
                case 'p':
                case 'P':
                    this.togglePointsBtn.click();
                    break;
                case 'n':
                case 'N':
                    this.toggleNodesBtn.click();
                    break;
            }
        });
    }

    /**
     * Update loading progress bar
     * 
     * @param {number} percent - Progress percentage (0-100)
     * @param {string} status - Status message to display
     */
    updateLoadingProgress(percent, status) {
        this.progressFill.style.width = `${percent}%`;
        
        if (status) {
            this.loadingStatus.textContent = status;
        }
        
        // Log major milestones
        if (percent === 100) {
            Utils.log('Loading complete');
        }
    }

    /**
     * Hide loading screen with fade animation
     */
    hideLoadingScreen() {
        setTimeout(() => {
            this.loadingScreen.classList.add('hidden');
            
            // Show navigation hint after loading
            setTimeout(() => {
                this.showNavigationHint('💡 Click on orange spheres to navigate between viewpoints', 5000);
            }, 500);
        }, 500);
    }

    /**
     * Update camera information display
     * 
     * @param {Object} cameraData - Camera data with id, image, position, etc.
     */
    updateCameraInfo(cameraData) {
        // Format camera label
        this.cameraLabel.textContent = `Camera ${cameraData.id + 1}`;
        
        // Format position
        const pos = cameraData.threeJS.position;
        const posStr = `(${pos.x.toFixed(2)}, ${pos.y.toFixed(2)}, ${pos.z.toFixed(2)})`;
        
        // Update info
        this.cameraInfo.textContent = `${cameraData.image} • Position: ${posStr}`;
        
        Utils.log('UI updated', { camera: cameraData.id });
    }

    /**
     * Cross-fade between two images during navigation
     * 
     * @param {string} currentImageSrc - Current image source
     * @param {string} nextImageSrc - Next image source
     * @param {number} duration - Fade duration in ms
     * @param {Function} onComplete - Callback when fade complete
     */
    crossFadeImages(currentImageSrc, nextImageSrc, duration, onComplete) {
        // Ensure images are set
        this.currentImage.src = currentImageSrc;
        this.nextImage.src = nextImageSrc;

        // Wait for next image to load
        this.nextImage.onload = () => {
            // Start cross-fade animation
            let progress = 0;
            const startTime = performance.now();

            const animate = () => {
                const elapsed = performance.now() - startTime;
                progress = Math.min(elapsed / duration, 1.0);

                // Apply easing
                const easedProgress = Utils.easeInOutCubic(progress);

                // Update opacities
                this.currentImage.style.opacity = 1 - easedProgress;
                this.nextImage.style.opacity = easedProgress;

                if (progress < 1.0) {
                    requestAnimationFrame(animate);
                } else {
                    // Fade complete - swap images
                    this.currentImage.src = nextImageSrc;
                    this.currentImage.style.opacity = 1;
                    this.nextImage.style.opacity = 0;
                    
                    if (onComplete) {
                        onComplete();
                    }
                }
            };

            animate();
        };

        // Handle image load error
        this.nextImage.onerror = () => {
            console.error('Failed to load image:', nextImageSrc);
            Utils.showError('Failed to load image');
            
            // Fallback: just swap immediately
            this.currentImage.src = nextImageSrc;
            if (onComplete) onComplete();
        };
    }

    /**
     * Show navigation hint message
     * 
     * @param {string} message - Message to display
     * @param {number} duration - How long to show (ms)
     */
    showNavigationHint(message, duration = 3000) {
        this.navHint.textContent = message;
        this.navHint.classList.remove('hidden');
        this.navHint.style.display = 'block';

        setTimeout(() => {
            this.navHint.classList.add('hidden');
            setTimeout(() => {
                this.navHint.style.display = 'none';
            }, 500);
        }, duration);
    }

    /**
     * Populate camera list sidebar (optional feature)
     * 
     * @param {Array} cameras - Array of camera data
     * @param {number} currentCameraId - ID of current camera
     */
    populateCameraList(cameras, currentCameraId) {
        const thumbnailsContainer = document.getElementById('camera-thumbnails');
        thumbnailsContainer.innerHTML = '';

        cameras.forEach((camera, index) => {
            const thumbDiv = document.createElement('div');
            thumbDiv.className = 'camera-thumb';
            if (camera.id === currentCameraId) {
                thumbDiv.classList.add('active');
            }

            thumbDiv.innerHTML = `
                <p><strong>Camera ${camera.id + 1}</strong></p>
                <small>${camera.image}</small>
            `;

            thumbDiv.addEventListener('click', () => {
                if (this.onCameraSelect && camera.id !== currentCameraId) {
                    this.onCameraSelect(camera);
                }
            });

            thumbnailsContainer.appendChild(thumbDiv);
        });
    }

    /**
     * Update active camera in list
     */
    updateActiveCameraInList(cameraId) {
        const thumbs = document.querySelectorAll('.camera-thumb');
        thumbs.forEach((thumb, index) => {
            if (index === cameraId) {
                thumb.classList.add('active');
            } else {
                thumb.classList.remove('active');
            }
        });
    }

    /**
     * Show/hide camera list sidebar
     */
    toggleCameraList(visible) {
        if (visible) {
            this.cameraList.classList.remove('hidden');
        } else {
            this.cameraList.classList.add('hidden');
        }
    }

    /**
     * Show loading indicator (for async operations)
     */
    showLoadingIndicator(message = 'Loading...') {
        // Simple implementation - could be enhanced
        this.loadingStatus.textContent = message;
    }

    /**
     * Show error message
     */
    showError(message) {
        Utils.showError(message);
    }

    /**
     * Get current UI state
     */
    getState() {
        return {
            pointsVisible: this.pointsVisible,
            nodesVisible: this.nodesVisible,
            helpModalActive: this.helpModal.classList.contains('active')
        };
    }

    /**
     * Reset UI to initial state
     */
    reset() {
        this.pointsVisible = false;
        this.nodesVisible = true;
        this.togglePointsBtn.textContent = 'Show Points';
        this.toggleNodesBtn.textContent = 'Hide Navigation';
        this.helpModal.classList.remove('active');
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = UIManager;
}