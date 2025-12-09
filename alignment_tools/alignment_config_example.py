{
  "sections": [
    {
      "name": "wall_A",
      "ply_path": "output/wall_A/final_reconstruction.ply",
      "cameras_path": "output/wall_A/camera_poses.json",
      "observations_path": "output/wall_A/observations.json",
      "images_dir": "data/wall_A_images"
    },
    {
      "name": "wall_B",
      "ply_path": "output/wall_B/final_reconstruction.ply",
      "cameras_path": "output/wall_B/camera_poses.json",
      "observations_path": "output/wall_B/observations.json",
      "images_dir": "data/wall_B_images"
    },
    {
      "name": "wall_C",
      "ply_path": "output/wall_C/final_reconstruction.ply",
      "cameras_path": "output/wall_C/camera_poses.json",
      "observations_path": "output/wall_C/observations.json",
      "images_dir": "data/wall_C_images"
    }
  ],
  "transforms": {
    "wall_B": {
      "rotation": 90,
      "translation": [3.0, 0.0, 0.0],
      "axis": "y",
      "description": "Wall B is 90 degrees rotated, 3 meters to the right"
    },
    "wall_C": {
      "rotation": 180,
      "translation": [3.0, 0.0, 3.0],
      "axis": "y",
      "description": "Wall C is 180 degrees rotated, at corner position"
    }
  },
  "output": {
    "merged_ply": "output/merged/merged_scene.ply",
    "merged_cameras": "output/merged/all_cameras.json",
    "view_graph": "output/merged/view_graph.json",
    "transformations": "output/merged/transformations.json"
  },
  "view_graph_params": {
    "min_shared_points": 20,
    "max_spatial_distance": 10.0,
    "add_spatial_fallback": true
  }
}