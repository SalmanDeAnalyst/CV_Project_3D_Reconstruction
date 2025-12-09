/**
 * camera_controller.js - Camera Navigation & Animation
 * 
 * Handles smooth camera transitions using lerp (position) and slerp (rotation)
 */

class CameraController {
    constructor(scene, renderer, config) {
        this.scene = scene;
        this.renderer = renderer;
        this.config = config;
        
        this.currentCamera = null;
        this.currentCameraData = null;
        this.isAnimating = false;
        this.animationProgress = 0;
        
        // Animation state
        this.startPosition = new THREE.Vector3();
        this.endPosition = new THREE.Vector3();
        this.startQuaternion = new THREE.Quaternion();
        this.endQuaternion = new THREE.Quaternion();
        this.animationStartTime = 0;
        this.animationDuration = 2000; // ms
        
        // Next camera to transition to
        this.nextCameraData = null;
        
        // Callbacks
        this.onNavigationStart = null;
        this.onNavigationComplete = null;
        this.onNavigationProgress = null;
    }

    /**
     * Set current camera from SfM data
     * 
     * @param {Object} cameraData - Camera data with R, t, image, etc.
     * @param {THREE.PerspectiveCamera} threeJSCamera - Three.js camera to control
     */
    setCamera(cameraData, threeJSCamera) {
        this.currentCameraData = cameraData;
        this.currentCamera = threeJSCamera;
        
        // Convert SfM to Three.js
        const converted = Utils.sfmCameraToThreeJS(cameraData);
        
        // Set camera position and rotation
        this.currentCamera.position.copy(converted.position);
        this.currentCamera.quaternion.copy(converted.quaternion);
        this.currentCamera.updateMatrixWorld();
        
        Utils.log('Camera set', {
            id: cameraData.id,
            position: converted.position,
            image: cameraData.image
        });
    }

    /**
     * Navigate to target camera with smooth animation
     * 
     * @param {Object} targetCameraData - Target camera data
     * @param {number} duration - Animation duration in ms (optional)
     */
    navigateTo(targetCameraData, duration) {
        if (this.isAnimating) {
            Utils.log('Already animating, ignoring navigation request');
            return;
        }

        // Can't navigate to same camera
        if (targetCameraData.id === this.currentCameraData.id) {
            Utils.log('Already at target camera');
            return;
        }

        // Convert target camera to Three.js format
        const targetConverted = Utils.sfmCameraToThreeJS(targetCameraData);

        // Store animation state
        this.startPosition.copy(this.currentCamera.position);
        this.startQuaternion.copy(this.currentCamera.quaternion);
        this.endPosition.copy(targetConverted.position);
        this.endQuaternion.copy(targetConverted.quaternion);
        
        // Start animation
        this.isAnimating = true;
        this.animationStartTime = performance.now();
        this.animationDuration = duration || this.config.navigation.transitionDuration;
        this.animationProgress = 0;
        
        // Store target as next current (will be applied at end)
        this.nextCameraData = targetCameraData;
        
        // Trigger start callback
        if (this.onNavigationStart) {
            this.onNavigationStart(this.currentCameraData, targetCameraData);
        }

        Utils.log('Navigation started', {
            from: this.currentCameraData.id,
            to: targetCameraData.id,
            duration: this.animationDuration
        });
    }

    /**
     * Update animation (call every frame in render loop)
     */
    update() {
        if (!this.isAnimating) return;

        const now = performance.now();
        const elapsed = now - this.animationStartTime;
        const rawProgress = Math.min(elapsed / this.animationDuration, 1.0);
        
        // Apply easing for smooth acceleration/deceleration
        this.animationProgress = Utils.easeInOutCubic(rawProgress);

        // Interpolate position using Lerp
        Utils.lerpVector3(
            this.startPosition,
            this.endPosition,
            this.animationProgress,
            this.currentCamera.position
        );

        // Interpolate rotation using Slerp (spherical linear interpolation)
        Utils.slerpQuaternion(
            this.startQuaternion,
            this.endQuaternion,
            this.animationProgress,
            this.currentCamera.quaternion
        );

        // Update camera matrix
        this.currentCamera.updateMatrixWorld();

        // Trigger progress callback (for UI updates)
        if (this.onNavigationProgress) {
            this.onNavigationProgress(this.animationProgress);
        }

        // Check if animation complete
        if (rawProgress >= 1.0) {
            this.completeAnimation();
        }
    }

    /**
     * Complete the animation
     */
    completeAnimation() {
        this.isAnimating = false;
        this.animationProgress = 0;
        
        // Update current camera
        const prevCamera = this.currentCameraData;
        this.currentCameraData = this.nextCameraData;
        this.nextCameraData = null;

        // Ensure final position is exact
        const finalConverted = Utils.sfmCameraToThreeJS(this.currentCameraData);
        this.currentCamera.position.copy(finalConverted.position);
        this.currentCamera.quaternion.copy(finalConverted.quaternion);
        this.currentCamera.updateMatrixWorld();

        // Trigger completion callback
        if (this.onNavigationComplete) {
            this.onNavigationComplete(prevCamera, this.currentCameraData);
        }

        Utils.log('Navigation complete', {
            now: this.currentCameraData.id,
            image: this.currentCameraData.image
        });
    }

    /**
     * Get current animation state
     * 
     * @returns {Object} Animation state info
     */
    getState() {
        return {
            isAnimating: this.isAnimating,
            progress: this.animationProgress,
            currentCamera: this.currentCameraData,
            nextCamera: this.nextCameraData
        };
    }

    /**
     * Cancel ongoing animation (if needed)
     */
    cancelAnimation() {
        if (!this.isAnimating) return;

        this.isAnimating = false;
        this.animationProgress = 0;
        this.nextCameraData = null;

        Utils.log('Animation cancelled');
    }

    /**
     * Get current camera data
     */
    getCurrentCameraData() {
        return this.currentCameraData;
    }

    /**
     * Check if currently animating
     */
    isCurrentlyAnimating() {
        return this.isAnimating;
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = CameraController;
}