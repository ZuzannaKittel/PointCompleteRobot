import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree

def extract_gt_object(mesh_path, anchor_point, output_path, label_property='label'):
    """
    Extracts a specific object instance from a semantic mesh based on a 3D anchor.
    """
    print(f"📂 Loading high-res semantic mesh: {mesh_path}...")
    # Using O3D to load the mesh
    mesh = o3d.io.read_point_cloud(mesh_path)
    points = np.asarray(mesh.points)
    
    # ScanNet++ semantic PLYs often store labels in vertex colors or custom properties.
    # We'll assume the standard layout where labels are mapped to points.
    # Note: If 'label' isn't a direct attribute in O3D, we'd use 'plyfile' or 'trimesh'.
    # For now, let's assume we can access them or use color-based mapping if labels are colors.
    labels = np.asarray(mesh.colors) # Fallback: many semantic PLYs encode labels as colors

    print(f"🔍 Finding semantic label at anchor {anchor_point}...")
    tree = cKDTree(points)
    _, idx = tree.query(anchor_point)
    target_label = labels[idx]
    
    print(f"🎯 Target label identified: {target_label}. Filtering mesh...")
    
    # 1. Semantic Filter: Keep only points with the same "identity"
    # We use a small tolerance if the labels are float-based colors
    mask = np.all(np.isclose(labels, target_label, atol=0.01), axis=1)
    semantic_points = points[mask]
    
    if len(semantic_points) == 0:
        print("❌ Error: No points found with the target label!")
        return None

    # 2. Spatial Clustering: Isolate the specific desk instance
    print(f"🧮 Performing Euclidean Clustering on {len(semantic_points)} semantic points...")
    temp_pcd = o3d.geometry.PointCloud()
    temp_pcd.points = o3d.utility.Vector3dVector(semantic_points)
    
    # labels here are cluster indices (0, 1, 2...)
    # eps=0.1 means points within 10cm are considered 'connected'
    cluster_indices = np.array(temp_pcd.cluster_dbscan(eps=0.1, min_points=50))
    
    # Find which cluster contains our anchor point
    # We re-query the tree but only against the semantic subset
    semantic_tree = cKDTree(semantic_points)
    _, sub_idx = semantic_tree.query(anchor_point)
    target_cluster_id = cluster_indices[sub_idx]
    
    if target_cluster_id == -1:
        print("⚠️ Warning: Anchor point is considered noise by DBSCAN. Picking nearest cluster.")
        target_cluster_id = cluster_indices[np.where(cluster_indices >= 0)[0][0]]

    # 3. Final Extraction
    gt_points = semantic_points[cluster_indices == target_cluster_id]
    
    # Center and Normalize (Critical for the Decoder)
    centroid = np.mean(gt_points, axis=0)
    gt_points_norm = gt_points - centroid
    scale = np.max(np.linalg.norm(gt_points_norm, axis=1))
    gt_points_norm /= scale

    print(f"✅ Extraction Successful! Saved {len(gt_points)} points to {output_path}")
    
    # Save as .ply for inspection
    gt_pcd = o3d.geometry.PointCloud()
    gt_pcd.points = o3d.utility.Vector3dVector(gt_points)
    o3d.io.write_point_cloud(output_path, gt_pcd)
    
    return gt_points, {"centroid": centroid, "scale": scale}