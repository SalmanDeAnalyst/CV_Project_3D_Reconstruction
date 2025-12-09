/**
 * viewer.js - Main Virtual Tour Viewer Application
 * 
 * Orchestrates all components: scene, camera, navigation, UI
 */

class VirtualTourViewer {
    constructor(config) {
        this.config = config;
        
        // Three.js core objects
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        
        // Data
        this.pointCloud = null;
        this.cameras = [];
        this.viewGraph = null;
        
        // Controllers
        this.cameraController = null;
        this.uiManager = null;
        
        // Navigation visualization
        this.navigationNodes = [];
        this.navigationGroup = new THREE.Group();
        
        // Raycaster for click detection
        this.raycaster = new THREE.Raycaster();
        this.mouse = new THREE.Vector2();
        
        // State
        this.initialized = false;
    }

    /**
     * Initialize the viewer
     */
    async init() {
        try {
            Utils.log('Starting Virtual Tour Viewer initialization...');
            
            // Initialize UI manager
            this.uiManager = new UIManager(this.config);
            
            // Setup Three.js scene
            this.setupThreeJS();
            
            // Load all data
            await this.loadData();
            
            // Setup camera controller
            this.setupCameraController();
            
            // Setup navigation system
            this.setupNavigation();
            
            // Setup UI interactions
            this.setupUICallbacks();
            
            // Start render loop
            this.animate();
            
            // Hide loading screen
            this.uiManager.hideLoadingScreen();
            
            this.initialized = true;
            Utils.log('✅ Viewer initialized successfully!');
            
        } catch (error) {
            console.error('❌ Initialization failed:', error);
            Utils.showError('Failed to load virtual tour. Check console for details.');
            throw error;
        }
    }

    /**
     * Setup Three.js scene, camera, renderer, lights
     */
    setupThreeJS() {
        Utils.log('Setting up Three.js...');
        
        // Create scene
        this.scene = new THREE.Scene();
        this.scene.background = null; // Transparent for background images

        // Create camera
        const aspect = window.innerWidth / window.innerHeight;
        this.camera = new THREE.PerspectiveCamera(
            this.config.camera.fov,
            aspect,
            this.config.camera.near,
            this.config.camera.far
        );

        // Create renderer
        const canvas = document.getElementById('canvas');
        this.renderer = new THREE.WebGLRenderer({ 
            canvas, 
            alpha: true,
            antialias: true 
        });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);

        // Add lights
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        this.scene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.4);
        directionalLight.position.set(10, 10, 10);
        this.scene.add(directionalLight);

        // Setup orbit controls (optional, disabled during navigation)
        if (this.config.controls.enableOrbit) {
            this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
            this.controls.enableDamping = true;
            this.controls.dampingFactor = 0.05;
            this.controls.enablePan = this.config.controls.enablePan;
            this.controls.enableZoom = this.config.controls.enableZoom;
        }

        // Event listeners
        window.addEventListener('resize', () => this.onWindowResize(), false);
        canvas.addEventListener('click', (e) => this.onCanvasClick(e), false);
        canvas.addEventListener('mousemove', (e) => this.onMouseMove(e), false);
        
        Utils.log('✓ Three.js setup complete');
    }

    /**
     * Load all data: point cloud, cameras, view graph
     */
    async loadData() {
        Utils.log('Loading data files...');
        
        // Step 1: Load point cloud
        this.uiManager.updateLoadingProgress(10, 'Loading point cloud...');
        
        const geometry = await Utils.loadPLY(
            this.config.data.pointCloud,
            (progress) => {
                const percent = 10 + (progress * 0.4); // 10% to 50%
                this.uiManager.updateLoadingProgress(percent, 'Loading point cloud...');
            }
        );

        const material = new THREE.PointsMaterial({ 
            size: 0.02, 
            vertexColors: true 
        });
        this.pointCloud = new THREE.Points(geometry, material);
        this.pointCloud.visible = false; // Hidden by default
        this.scene.add(this.pointCloud);

        const numPoints = geometry.attributes.position.count;
        Utils.log(`✓ Point cloud loaded: ${numPoints.toLocaleString()} points`);

        // Step 2: Load cameras
        this.uiManager.updateLoadingProgress(50, 'Loading cameras...');
        
        const camerasData = await Utils.loadJSON(this.config.data.cameras);
        this.cameras = Array.isArray(camerasData) ? camerasData : camerasData.cameras || [];

        // Convert all cameras to Three.js format
        this.cameras.forEach(cam => {
            cam.threeJS = Utils.sfmCameraToThreeJS(cam);
        });

        Utils.log(`✓ Cameras loaded: ${this.cameras.length}`);

        // Step 3: Load view graph
        this.uiManager.updateLoadingProgress(70, 'Loading view graph...');
        
        const viewGraphData = await Utils.loadJSON(this.config.data.viewGraph);
        this.viewGraph = new ViewGraph(viewGraphData, this.cameras);

        const stats = this.viewGraph.getStatistics();
        Utils.log('✓ View graph loaded:', stats);

        // Step 4: Preload first few images
        this.uiManager.updateLoadingProgress(90, 'Preloading images...');
        
        const firstImages = this.cameras.slice(0, 3).map(cam => 
            this.config.data.imagesPath + cam.image
        );
        
        await Promise.all(firstImages.map(src => Utils.preloadImage(src).catch(() => {
            console.warn('Failed to preload image:', src);
        })));

        this.uiManager.updateLoadingProgress(100, 'Ready!');
        Utils.log('✓ All data loaded successfully');
    }

    /**
     * Setup camera controller and initial camera
     */
    setupCameraController() {
        Utils.log('Setting up camera controller...');
        
        this.cameraController = new CameraController(
            this.scene,
            this.renderer,
            this.config
        );

        // Set initial camera (first one)
        const initialCamera = this.cameras[0];
        this.cameraController.setCamera(initialCamera, this.camera);

        // Load initial image
        const initialImagePath = this.config.data.imagesPath + initialCamera.image;
        this.uiManager.currentImage.src = initialImagePath;

        // Update UI
        this.uiManager.updateCameraInfo(initialCamera);

        // Setup navigation callbacks
        this.cameraController.onNavigationStart = (from, to) => {
            this.onNavigationStart(from, to);
        };

        this.cameraController.onNavigationProgress = (progress) => {
            this.onNavigationProgress(progress);
        };

        this.cameraController.onNavigationComplete = (from, to) => {
            this.onNavigationComplete(from, to);
        };

        Utils.log('✓ Camera controller ready');
    }

    /**
     * Setup navigation nodes and view graph visualization
     */
    setupNavigation() {
        Utils.log('Setting up navigation system...');
        
        // Add navigation group to scene
        this.scene.add(this.navigationGroup);

        // Create initial navigation nodes
        this.updateNavigationNodes();

        Utils.log('✓ Navigation system ready');
    }

    /**
     * Update navigation nodes based on current camera
     */
    updateNavigationNodes() {
        // Clear existing nodes
        this.navigationGroup.clear();
        this.navigationNodes = [];

        const currentCamera = this.cameraController.currentCameraData;
        const neighbors = this.viewGraph.getNeighbors(currentCamera.id);

        // Create sphere for each neighbor camera
        neighbors.forEach(neighbor => {
            const neighborCam = this.viewGraph.getCamera(neighbor.target_camera);
            if (!neighborCam) return;

            // Create sphere geometry
            const geometry = new THREE.SphereGeometry(this.config.navigation.nodeSize, 16, 16);
            const material = new THREE.MeshBasicMaterial({ 
                color: this.config.navigation.nodeColor,
                transparent: true,
                opacity: 0.8
            });
            const sphere = new THREE.Mesh(geometry, material);

            // Position at neighbor camera location
            sphere.position.copy(neighborCam.threeJS.position);
            
            // Store metadata
            sphere.userData = {
                cameraId: neighborCam.id,
                cameraData: neighborCam,
                isNavigationNode: true,
                sharedPoints: neighbor.shared_points,
                distance: neighbor.distance
            };

            this.navigationGroup.add(sphere);
            this.navigationNodes.push(sphere);
        });

        Utils.log(`Navigation nodes updated: ${this.navigationNodes.length} connections`);
    }

    /**
     * Setup UI callbacks
     */
    setupUICallbacks() {
        // Point cloud toggle
        this.uiManager.onTogglePoints = (visible) => {
            if (!this.cameraController.isAnimating) {
                this.pointCloud.visible = visible;
            }
        };

        // Navigation nodes toggle
        this.uiManager.onToggleNodes = (visible) => {
            this.navigationGroup.visible = visible;
        };

        // Camera selection from list
        this.uiManager.onCameraSelect = (cameraData) => {
            if (!this.cameraController.isAnimating) {
                this.cameraController.navigateTo(
                    cameraData, 
                    this.config.navigation.transitionDuration
                );
            }
        };

        // Populate camera list (optional)
        // this.uiManager.populateCameraList(this.cameras, this.cameras[0].id);
    }

    /**
     * Handle canvas click for navigation
     */
    onCanvasClick(event) {
        if (this.cameraController.isAnimating) return;

        // Calculate normalized mouse coordinates
        this.mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
        this.mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

        // Perform raycasting
        this.raycaster.setFromCamera(this.mouse, this.camera);
        const intersects = this.raycaster.intersectObjects(this.navigationNodes);

        if (intersects.length > 0) {
            const clicked = intersects[0].object;
            
            if (clicked.userData.isNavigationNode) {
                const targetCamera = clicked.userData.cameraData;
                
                Utils.log('Navigation node clicked', {
                    target: targetCamera.id,
                    sharedPoints: clicked.userData.sharedPoints
                });
                
                // Navigate to target camera
                this.cameraController.navigateTo(
                    targetCamera, 
                    this.config.navigation.transitionDuration
                );
            }
        }
    }

    /**
     * Handle mouse move for hover effects (optional)
     */
    onMouseMove(event) {
        if (this.cameraController.isAnimating) return;

        // Calculate mouse position
        this.mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
        this.mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

        // Raycast to check if hovering over navigation node
        this.raycaster.setFromCamera(this.mouse, this.camera);
        const intersects = this.raycaster.intersectObjects(this.navigationNodes);

        // Update cursor
        if (intersects.length > 0 && intersects[0].object.userData.isNavigationNode) {
            document.body.style.cursor = 'pointer';
        } else {
            document.body.style.cursor = 'default';
        }
    }

    /**
     * Navigation started callback
     */
    onNavigationStart(fromCamera, toCamera) {
        Utils.log('Navigation animation starting...', {
            from: fromCamera.id,
            to: toCamera.id
        });

        // Disable orbit controls during navigation
        if (this.controls) {
            this.controls.enabled = false;
        }

        // Start image cross-fade
        const fromImagePath = this.config.data.imagesPath + fromCamera.image;
        const toImagePath = this.config.data.imagesPath + toCamera.image;

        this.uiManager.crossFadeImages(
            fromImagePath,
            toImagePath,
            this.config.navigation.transitionDuration
        );
    }

    /**
     * Navigation progress callback
     */
    onNavigationProgress(progress) {
        // Show point cloud during transition if enabled
        if (this.config.navigation.showPointsDuringTransition) {
            this.pointCloud.visible = true;
        }
    }

    /**
     * Navigation completed callback
     */
    onNavigationComplete(fromCamera, toCamera) {
        Utils.log('Navigation animation complete', {
            now: toCamera.id
        });

        // Re-enable orbit controls
        if (this.controls) {
            this.controls.enabled = true;
        }

        // Hide point cloud (unless user toggled it on)
        if (!this.uiManager.pointsVisible) {
            this.pointCloud.visible = false;
        }

        // Update navigation nodes for new camera
        this.updateNavigationNodes();

        // Update UI
        this.uiManager.updateCameraInfo(toCamera);
        
        // Update camera list if using it
        // this.uiManager.updateActiveCameraInList(toCamera.id);
    }

    /**
     * Handle window resize
     */
    onWindowResize() {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
    }

    /**
     * Main animation loop
     */
    animate() {
        requestAnimationFrame(() => this.animate());

        // Update camera controller (handles navigation animation)
        this.cameraController.update();

        // Update orbit controls
        if (this.controls) {
            this.controls.update();
        }

        // Render scene
        this.renderer.render(this.scene, this.camera);
    }

    /**
     * Cleanup resources (if needed)
     */
    dispose() {
        if (this.renderer) {
            this.renderer.dispose();
        }
        
        if (this.controls) {
            this.controls.dispose();
        }

        Utils.log('Viewer disposed');
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = VirtualTourViewer;
}