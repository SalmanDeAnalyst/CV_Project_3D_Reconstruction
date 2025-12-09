# Phase 3A: Python Alignment Tools - README


# Option 1: Automated (recommended)
python complete_alignment_workflow.py --config my_config.json

# Option 2: Manual (step-by-step)
python merge_clouds.py --interactive
python transform_cameras.py --cameras wall_B/camera_poses.json ...
python build_view_graph.py --cameras all_cameras.json ...

## Overview

This toolkit provides three Python scripts to prepare your SfM reconstructions for the Three.js virtual tour:

1. **`merge_clouds.py`** - Align and merge multiple point clouds
2. **`transform_cameras.py`** - Apply same transformations to camera poses
3. **`build_view_graph.py`** - Compute camera connectivity for navigation

Plus **`complete_alignment_workflow.py`** - runs all three in sequence!

---

## Prerequisites

```bash
pip install open3d numpy
```

---

## Quick Start (Automated Workflow)

### Step 1: Create Config File

Copy `alignment_config_example.json` and edit it:

```json
{
  "sections": [
    {
      "name": "wall_A",
      "ply_path": "output/wall_A/final_reconstruction.ply",
      "cameras_path": "output/wall_A/camera_poses.json",
      "observations_path": "output/wall_A/observations.json"
    },
    {
      "name": "wall_B",
      "ply_path": "output/wall_B/final_reconstruction.ply",
      "cameras_path": "output/wall_B/camera_poses.json",
      "observations_path": "output/wall_B/observations.json"
    }
  ],
  "transforms": {
    "wall_B": {
      "rotation": 90,
      "translation": [3.0, 0.0, 0.0],
      "axis": "y"
    }
  }
}
```

### Step 2: Run Complete Workflow

```bash
python complete_alignment_workflow.py --config my_alignment_config.json
```

**Done!** This will generate:
- `output/merged/merged_scene.ply` - Merged point cloud
- `output/merged/all_cameras.json` - All camera poses (aligned)
- `output/merged/view_graph.json` - Navigation graph
- `output/merged/transformations.json` - Transform matrices

---

## Manual Usage (Individual Tools)

### Tool 1: Merge Point Clouds

**Interactive Mode (Easiest):**
```bash
python merge_clouds.py --interactive
```

**Config File Mode:**
```bash
python merge_clouds.py --config alignment_config.json --preview
```

**Manual Mode:**
```bash
python merge_clouds.py \
  --sections wall_A:output/wall_A/final_reconstruction.ply \
             wall_B:output/wall_B/final_reconstruction.ply \
  --output output/merged/merged_scene.ply \
  --preview
```

**Output:**
- `merged_scene.ply` - Merged point cloud
- `transformations.json` - Transform matrices (CRITICAL for next step!)

---

### Tool 2: Transform Cameras

**Transform a Single Section:**
```bash
python transform_cameras.py \
  --cameras output/wall_B/camera_poses.json \
  --transform output/merged/transformations.json \
  --section wall_B \
  --output output/merged/cameras_wall_B_transformed.json
```

**Merge All Cameras:**
```bash
python transform_cameras.py --merge \
  --cameras output/merged/cameras_wall_A.json \
            output/merged/cameras_wall_B_transformed.json \
            output/merged/cameras_wall_C_transformed.json \
  --output output/merged/all_cameras.json
```

**Output:**
- `all_cameras.json` - All camera poses in global coordinates

---

### Tool 3: Build View Graph

```bash
python build_view_graph.py \
  --cameras output/merged/all_cameras.json \
  --observations output/wall_A/observations.json \
                output/wall_B/observations.json \
  --output output/merged/view_graph.json \
  --min-shared 20 \
  --max-distance 10.0 \
  --add-spatial-fallback
```

**Parameters:**
- `--min-shared`: Minimum shared 3D points to connect cameras (default: 20)
- `--max-distance`: Maximum distance in meters to connect cameras (default: 10.0)
- `--add-spatial-fallback`: Connect isolated cameras by proximity

**Output:**
- `view_graph.json` - Camera connectivity graph

---

## Understanding the Workflow

### Why Three Steps?

```
Step 1: merge_clouds.py
  Input:  Multiple .ply files (each in own coordinate system)
  Output: Single merged .ply + transformation matrices
  
Step 2: transform_cameras.py
  Input:  Original camera_poses.json + transformation matrices
  Output: Transformed camera poses (aligned with merged point cloud)
  
Step 3: build_view_graph.py
  Input:  All cameras + observations
  Output: Navigation graph (which cameras connect to which)
```

### Critical Requirement

**The transformation matrices from Step 1 MUST be applied to cameras in Step 2!**

If you don't do this, cameras will be in wrong positions relative to the point cloud.

---

## How to Define Transformations

### Method 1: Use Room Measurements

If you know the physical layout:

```json
{
  "wall_B": {
    "rotation": 90,
    "translation": [3.5, 0.0, 0.0],
    "axis": "y"
  }
}
```

Meaning: "Wall B is perpendicular to Wall A, 3.5 meters to the right"

### Method 2: Use Interactive Mode

```bash
python merge_clouds.py --interactive
```

This will:
1. Show you each point cloud
2. Let you input transformations
3. Show aligned result
4. Save transformations automatically

### Method 3: Use MeshLab (Visual Alignment)

1. Open MeshLab
2. Load all .ply files
3. Use manipulator tool to visually align them
4. Note the transformations you applied
5. Input those into the config file

---

## Troubleshooting

### Issue: "No cameras share 3D points"

**Cause:** Your sections don't overlap (no common 3D points visible).

**Solution:** Use `--add-spatial-fallback` to connect cameras by proximity instead.

```bash
python build_view_graph.py ... --add-spatial-fallback
```

### Issue: "Cameras are in wrong positions in Three.js"

**Cause:** You transformed the point cloud but forgot to transform cameras, or used different transformations.

**Solution:** 
1. Make sure you ran `transform_cameras.py` 
2. Use the SAME `transformations.json` file for both point clouds and cameras

### Issue: "View graph is empty"

**Causes:**
1. `observations.json` files are missing or empty
2. `min_shared_points` threshold is too high
3. Cameras are too far apart (exceed `max_distance`)

**Solutions:**
1. Make sure you exported observations from SfM (see main README)
2. Lower `--min-shared` to 10 or 5
3. Increase `--max-distance` to 20.0 or more

### Issue: "Point clouds don't align well"

**Cause:** Transformation parameters are incorrect.

**Solution:**
1. Use `--preview` flag to visualize before/after
2. Try interactive mode to adjust transformations
3. Use MeshLab to align visually first

---

## Output File Formats

### `merged_scene.ply`
Standard PLY point cloud file. Load in Three.js with PLYLoader.

### `all_cameras.json`
```json
[
  {
    "id": 0,
    "image": "1.jpeg",
    "R": [[...], [...], [...]],
    "t": [[tx], [ty], [tz]],
    "center": [cx, cy, cz]
  },
  ...
]
```

### `view_graph.json`
```json
{
  "view_graph": {
    "0": [
      {"target_camera": 1, "shared_points": 350, "distance": 0.5},
      {"target_camera": 2, "shared_points": 280, "distance": 0.8}
    ],
    ...
  },
  "metadata": {
    "num_cameras": 37,
    "num_nodes": 37,
    "min_shared_points": 20,
    "max_spatial_distance": 10.0
  }
}
```

---

## Next: Phase 3B (Three.js Viewer)

Once you have:
- ✅ `merged_scene.ply`
- ✅ `all_cameras.json`
- ✅ `view_graph.json`

You're ready to build the Three.js viewer!

---

## Tips for Best Results

### 1. Image Capture

**Capture overlapping images at section boundaries!**

```
Wall A images: [..., 13, 14, 15]
                          ↓  ↓  corner images
Wall B images:           [14, 15, 16, ...]
```

Images 14-15 should show BOTH walls → enables alignment.

### 2. Transformation Accuracy

**Start with approximate transformations, then refine:**

1. Use room measurements for initial guess
2. Visualize with `--preview`
3. Adjust translation/rotation
4. Re-run until alignment looks good

### 3. View Graph Parameters

**Adjust based on your scene:**

- **Dense captures** (images every 10cm): `min_shared=30`, `max_distance=3.0`
- **Sparse captures** (images every 50cm): `min_shared=10`, `max_distance=15.0`
- **Disconnected sections**: `add_spatial_fallback=True`

---

## Files in This Package

```
phase_3a_tools/
├── merge_clouds.py                  # Tool 1: Merge point clouds
├── transform_cameras.py             # Tool 2: Transform cameras
├── build_view_graph.py              # Tool 3: Build view graph
├── complete_alignment_workflow.py   # Run all three
├── alignment_config_example.json    # Example config
└── README.md                        # This file
```

---

## Questions?

**Common Issues:**
- See Troubleshooting section above
- Check that all input files exist
- Verify observations.json is exported from SfM

**For Help:**
- Check tool output messages (they're very descriptive!)
- Use `--help` flag on any script
- Enable `--preview` to visualize results

**Ready for Phase 3B?**
Move on to the Three.js viewer implementation! 🚀