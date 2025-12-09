/**
 * utils.js - Utility functions for the virtual tour viewer
 */

const Utils = {
    /**
     * Linear interpolation between two values
     */
    lerp: (start, end, t) => {
        return start * (1 - t) + end * t;
    },

    /**
     * Linear interpolation for THREE.Vector3
     */
    lerpVector3: (start, end, t, target) => {
        target.x = Utils.lerp(start.x, end.x, t);
        target.y = Utils.lerp(start.y, end.y, t);
        target.z = Utils.lerp(start.z, end.z, t);
        return target;
    },

    /**
     * Spherical linear interpolation for THREE.Quaternion
     */
    slerpQuaternion: (start, end, t, target) => {
        target.slerpQuaternions(start, end, t);
        return target;
    },

    /**
     * Easing function for smooth animations (ease-in-out)
     */
    easeInOutCubic: (t) => {
        return t < 0.5
            ? 4 * t * t * t
            : 1 - Math.pow(-2 * t + 2, 3) / 2;
    },

    /**
     * Convert SfM camera (R, t) to Three.js camera (position, quaternion)
     * 
     * SfM format:
     * - R: 3x3 rotation matrix
     * - t: 3x1 translation vector (NOT camera center!)
     * 
     * THREE.js format:
     * - position: Vector3 (camera center in world coords)
     * - quaternion: Quaternion (rotation)
     */
    sfmCameraToThreeJS: (sfmCamera) => {
        // Extract R and t
        const R = sfmCamera.R;
        const t = sfmCamera.t;

        // Flatten t if needed
        let t_vec;
        if (Array.isArray(t[0])) {
            // t is [[tx], [ty], [tz]]
            t_vec = new THREE.Vector3(t[0][0], t[1][0], t[2][0]);
        } else {
            // t is [tx, ty, tz]
            t_vec = new THREE.Vector3(t[0], t[1], t[2]);
        }

        // Convert R to THREE.Matrix4
        const R_matrix4 = new THREE.Matrix4();
        R_matrix4.set(
            R[0][0], R[0][1], R[0][2], 0,
            R[1][0], R[1][1], R[1][2], 0,
            R[2][0], R[2][1], R[2][2], 0,
            0, 0, 0, 1
        );

        // Compute camera center: C = -R^T @ t
        const R_transpose = new THREE.Matrix3();
        R_transpose.set(
            R[0][0], R[1][0], R[2][0],
            R[0][1], R[1][1], R[2][1],
            R[0][2], R[1][2], R[2][2]
        );

        const center = new THREE.Vector3();
        center.x = -(R_transpose.elements[0] * t_vec.x + R_transpose.elements[1] * t_vec.y + R_transpose.elements[2] * t_vec.z);
        center.y = -(R_transpose.elements[3] * t_vec.x + R_transpose.elements[4] * t_vec.y + R_transpose.elements[5] * t_vec.z);
        center.z = -(R_transpose.elements[6] * t_vec.x + R_transpose.elements[7] * t_vec.y + R_transpose.elements[8] * t_vec.z);

        // Convert rotation matrix to quaternion
        const quaternion = new THREE.Quaternion();
        quaternion.setFromRotationMatrix(R_matrix4);

        // CRITICAL: Coordinate system conversion (SfM to Three.js)
        // SfM: Y-down, Z-forward
        // Three.js: Y-up, Z-backward
        // Apply 180° rotation around X-axis
        const coordConversion = new THREE.Quaternion();
        coordConversion.setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI);
        quaternion.multiply(coordConversion);

        return {
            position: center,
            quaternion: quaternion
        };
    },

    /**
     * Load JSON file
     */
    loadJSON: async (url) => {
        try {
            const response = await fetch(url);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            console.error(`Error loading ${url}:`, error);
            throw error;
        }
    },

    /**
     * Load PLY file using THREE.PLYLoader
     */
    loadPLY: (url, onProgress) => {
        return new Promise((resolve, reject) => {
            const loader = new THREE.PLYLoader();
            loader.load(
                url,
                (geometry) => {
                    resolve(geometry);
                },
                (xhr) => {
                    if (onProgress) {
                        const percentComplete = (xhr.loaded / xhr.total) * 100;
                        onProgress(percentComplete);
                    }
                },
                (error) => {
                    reject(error);
                }
            );
        });
    },

    /**
     * Preload image
     */
    preloadImage: (src) => {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => resolve(img);
            img.onerror = reject;
            img.src = src;
        });
    },

    /**
     * Compute distance between two Vector3 points
     */
    distance: (v1, v2) => {
        return v1.distanceTo(v2);
    },

    /**
     * Check if point is in camera's field of view
     */
    isInFOV: (camera, point, fovMargin = 10) => {
        // Project point to camera space
        const frustum = new THREE.Frustum();
        const cameraViewProjectionMatrix = new THREE.Matrix4();
        cameraViewProjectionMatrix.multiplyMatrices(
            camera.projectionMatrix,
            camera.matrixWorldInverse
        );
        frustum.setFromProjectionMatrix(cameraViewProjectionMatrix);

        return frustum.containsPoint(point);
    },

    /**
     * Format camera info for display
     */
    formatCameraInfo: (camera, cameraData) => {
        const pos = camera.position;
        return {
            id: cameraData.id,
            image: cameraData.image,
            position: `(${pos.x.toFixed(2)}, ${pos.y.toFixed(2)}, ${pos.z.toFixed(2)})`,
            label: `Camera ${cameraData.id + 1}`
        };
    },

    /**
     * Debounce function (useful for resize events)
     */
    debounce: (func, wait) => {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },

    /**
     * Check if device is mobile
     */
    isMobile: () => {
        return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
    },

    /**
     * Clamp value between min and max
     */
    clamp: (value, min, max) => {
        return Math.min(Math.max(value, min), max);
    },

    /**
     * Get random color (for debugging)
     */
    randomColor: () => {
        return Math.floor(Math.random() * 16777215);
    },

    /**
     * Log with timestamp (for debugging)
     */
    log: (message, ...args) => {
        const timestamp = new Date().toISOString().split('T')[1].split('.')[0];
        console.log(`[${timestamp}] ${message}`, ...args);
    },

    /**
     * Show error message to user
     */
    showError: (message) => {
        const errorDiv = document.createElement('div');
        errorDiv.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: rgba(255, 0, 0, 0.9);
            color: white;
            padding: 20px 40px;
            border-radius: 10px;
            z-index: 10000;
            font-size: 1.2em;
            text-align: center;
        `;
        errorDiv.textContent = message;
        document.body.appendChild(errorDiv);

        setTimeout(() => {
            errorDiv.remove();
        }, 5000);
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = Utils;
}