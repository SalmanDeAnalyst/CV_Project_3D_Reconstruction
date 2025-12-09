/**
 * view_graph.js - View Graph Management
 * 
 * Handles camera connectivity and navigation nodes
 */

class ViewGraph {
    constructor(viewGraphData, cameras) {
        this.graph = viewGraphData.view_graph;
        this.metadata = viewGraphData.metadata;
        this.cameras = cameras;
        this.cameraDict = {};
        
        // Build camera lookup
        cameras.forEach(cam => {
            this.cameraDict[cam.id] = cam;
        });

        Utils.log('ViewGraph initialized', {
            nodes: Object.keys(this.graph).length,
            cameras: cameras.length
        });
    }

    /**
     * Get neighbors for a camera
     * 
     * @param {number} cameraId - ID of current camera
     * @returns {Array} Array of neighbor camera IDs with metadata
     */
    getNeighbors(cameraId) {
        const neighbors = this.graph[String(cameraId)] || [];
        
        // Filter out invalid neighbors
        return neighbors.filter(neighbor => {
            return this.cameraDict[neighbor.target_camera] !== undefined;
        });
    }

    /**
     * Check if two cameras are connected
     */
    areConnected(cameraId1, cameraId2) {
        const neighbors = this.getNeighbors(cameraId1);
        return neighbors.some(n => n.target_camera === cameraId2);
    }

    /**
     * Get camera data by ID
     */
    getCamera(cameraId) {
        return this.cameraDict[cameraId];
    }

    /**
     * Find nearest cameras spatially (fallback if no graph connection)
     */
    findNearestCameras(currentCamera, maxDistance = 10, count = 5) {
        const currentPos = currentCamera.threeJS.position;
        const distances = [];

        this.cameras.forEach(cam => {
            if (cam.id === currentCamera.id) return;
            
            const camPos = cam.threeJS.position;
            const dist = Utils.distance(currentPos, camPos);
            
            if (dist <= maxDistance) {
                distances.push({
                    camera: cam,
                    distance: dist
                });
            }
        });

        // Sort by distance
        distances.sort((a, b) => a.distance - b.distance);
        
        return distances.slice(0, count);
    }

    /**
     * Get statistics
     */
    getStatistics() {
        const totalNodes = Object.keys(this.graph).length;
        const degrees = Object.values(this.graph).map(neighbors => neighbors.length);
        const avgDegree = degrees.reduce((a, b) => a + b, 0) / degrees.length;
        const maxDegree = Math.max(...degrees);
        const minDegree = Math.min(...degrees);

        return {
            totalNodes,
            avgDegree: avgDegree.toFixed(2),
            maxDegree,
            minDegree,
            totalCameras: this.cameras.length
        };
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ViewGraph;
}