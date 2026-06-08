"""
Deterministic 3D Point Cloud Registration - CLUSTER-FIRST APPROACH
==========================================
NEW STRATEGY: Cluster abnormal points FIRST, then classify entire clusters
This prevents defects from being fragmented across multiple classes.
"""

import numpy as np
import open3d as o3d
import copy
import time
import os
import re
import json
import datetime

# Fixed random seed
np.random.seed(42)
o3d.utility.random.seed(42)


def load_or_sample_gear(mesh_path, cache_path=None, num_points=30000):
    """Load cached gear points or sample and cache them."""
    if cache_path is None:
        cache_path = mesh_path.replace(".stl", "_sampled.npy")
    
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    
    # Stitch mesh: remove duplicated vertices to fix boundary detection
    print(f"  Stitching mesh {mesh_path}...")
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    
    mesh.compute_vertex_normals()
    
    if os.path.exists(cache_path):
        # Load cached points for consistency
        print(f"  Loading cached gear points from {cache_path}")
        points = np.load(cache_path)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
    else:
        # Sample and cache
        print(f"  Sampling {num_points} points from mesh...")
        pcd = mesh.sample_points_uniformly(number_of_points=num_points)
        np.save(cache_path, np.asarray(pcd.points))
        print(f"  Cached to {cache_path}")
    
    return pcd, mesh


def preprocess(pcd, voxel_size, remove_outliers=True):
    """Preprocess point cloud."""
    if remove_outliers:
        pcd_clean, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    else:
        pcd_clean = pcd
    
    pcd_down = pcd_clean.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )
    pcd_down.orient_normals_consistent_tangent_plane(k=15)
    
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100)
    )
    return pcd_down, fpfh


def global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    """RANSAC global registration."""
    distance_threshold = voxel_size * 2.0
    
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        mutual_filter=False,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
            max_iteration=10000000, confidence=0.99999
        )
    )
    return result


def icp_refine(source, target, init_transform, voxel_size):
    """ICP refinement with multiple scales for better convergence."""
    if not source.has_normals():
        source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
    if not target.has_normals():
        target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
    
    # Scale 1: Coarse (3.0x voxel size)
    result_coarse = o3d.pipelines.registration.registration_icp(
        source, target, voxel_size * 3.0, init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
    )
    
    # Scale 2: Fine (1.5x voxel size)
    result_fine = o3d.pipelines.registration.registration_icp(
        source, target, voxel_size * 1.5, result_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
    )
    return result_fine


def compute_quality(source, target, transformation, threshold):
    """Compute alignment quality."""
    source_temp = copy.deepcopy(source)
    source_temp.transform(transformation)
    
    target_tree = o3d.geometry.KDTreeFlann(target)
    source_pts = np.asarray(source_temp.points)
    
    inliers = 0
    total_dist = 0.0
    
    for pt in source_pts:
        [_, idx, dist] = target_tree.search_knn_vector_3d(pt, 1)
        d = np.sqrt(dist[0])
        if d < threshold:
            inliers += 1
            total_dist += d
    
    fitness = inliers / len(source_pts)
    rmse = total_dist / max(inliers, 1)
    return fitness, rmse, inliers

class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def save_registration_metadata(filename, transformation, rmse, thresholds, mesh_footprint=None):
    """Save registration metadata to JSON file."""
    metadata = {
        "timestamp": datetime.datetime.now().isoformat(),
        "transformation": transformation,
        "rmse": rmse,
        "thresholds": thresholds,
        "mesh_footprint": mesh_footprint
    }
    
    with open(filename, 'w') as f:
        json.dump(metadata, f, cls=NumpyEncoder, indent=2)
    print(f"  Saved registration metadata to {filename}")


def load_registration_metadata(filename):
    """Load registration metadata from JSON file."""
    if not os.path.exists(filename):
        return None
        
    try:
        with open(filename, 'r') as f:
            metadata = json.load(f)
        
        # Convert lists back to numpy arrays
        metadata["transformation"] = np.array(metadata["transformation"])
        if metadata.get("mesh_footprint"):
            metadata["mesh_footprint"] = np.array(metadata["mesh_footprint"])
            
        print(f"  Loaded registration metadata from {filename}")
        return metadata
    except Exception as e:
        print(f"  Error loading metadata: {e}")
        return None


def run_registration(mesh_path="gear.stl", data_path="data.csv", 
                     num_attempts=20, num_points=30000, cache_path=None):
    """Run deterministic registration."""
    print("="*60)
    print("DETERMINISTIC POINT CLOUD REGISTRATION")
    print("="*60)
    
    # Load data with caching
    print("\n[1. Loading Data]")
    pcd_gear, mesh = load_or_sample_gear(mesh_path, cache_path, num_points)
    print(f"  Gear: {len(pcd_gear.points)} points")
    
    data = np.genfromtxt(data_path, delimiter=",")
    data = data[~np.isnan(data).any(axis=1)]
    pcd_data = o3d.geometry.PointCloud()
    pcd_data.points = o3d.utility.Vector3dVector(data)
    print(f"  Scan: {len(pcd_data.points)} points")
    
    gear_pts = np.asarray(pcd_gear.points)
    data_pts = np.asarray(pcd_data.points)
    gear_range = np.max(gear_pts.max(axis=0) - gear_pts.min(axis=0))
    data_range = np.max(data_pts.max(axis=0) - data_pts.min(axis=0))
    
    print(f"\n  Gear extent: {gear_range:.2f} units")
    print(f"  Data extent: {data_range:.2f} units")
    
    # Initial alignment
    print("\n[2. Initial Center Alignment]")
    init_transform = np.eye(4)
    init_transform[:3, 3] = data_pts.mean(axis=0) - gear_pts.mean(axis=0)
    print(f"  Translation: {init_transform[:3, 3]}")
    
    pcd_gear_init = copy.deepcopy(pcd_gear)
    pcd_gear_init.transform(init_transform)
    
    voxel_coarse = gear_range / 20.0
    voxel_fine = gear_range / 50.0
    
    print(f"\n  Coarse voxel: {voxel_coarse:.2f}")
    print(f"  Fine voxel: {voxel_fine:.2f}")
    
    # Preprocessing
    print("\n[3. Preprocessing]")
    start = time.time()
    
    gear_down, gear_fpfh = preprocess(pcd_gear_init, voxel_coarse, remove_outliers=False)
    print(f"  Gear: {len(gear_down.points)} points")
    
    data_down, data_fpfh = preprocess(pcd_data, voxel_coarse, remove_outliers=True)
    print(f"  Data: {len(data_down.points)} points")
    
    # Multiple RANSAC attempts - select by BEST RMSE when fitness > 0.95
    print(f"\n[4. Global Registration ({num_attempts} attempts)]")
    
    best_result = None
    best_score = -999
    
    for i in range(num_attempts):
        result = global_registration(gear_down, data_down, gear_fpfh, data_fpfh, voxel_coarse)
        
        # Score: prioritize high fitness, then low RMSE
        if result.fitness >= 0.99:
            score = 100 - result.inlier_rmse  # When fitness is perfect, minimize RMSE
        else:
            score = result.fitness * 10 - result.inlier_rmse / 10
        
        status = "★" if score > best_score else " "
        print(f"  {status} Attempt {i+1:2d}: Fitness={result.fitness:.4f}, RMSE={result.inlier_rmse:.4f}")
        
        if score > best_score:
            best_score = score
            best_result = result
    
    print(f"\n  Best: Fitness={best_result.fitness:.4f}, RMSE={best_result.inlier_rmse:.4f}")
    
    current_transform = best_result.transformation @ init_transform
    
    # ICP refinement
    print("\n[5. ICP Refinement]")
    if best_result.fitness > 0.5:
        print("  Applying ICP...")
        
        pcd_gear_aligned = copy.deepcopy(pcd_gear)
        pcd_gear_aligned.transform(current_transform)
        
        gear_fine, _ = preprocess(pcd_gear_aligned, voxel_fine, remove_outliers=False)
        data_fine, _ = preprocess(pcd_data, voxel_fine, remove_outliers=False)
        
        icp_result = icp_refine(gear_fine, data_fine, np.eye(4), voxel_fine)
        print(f"  ICP Fitness: {icp_result.fitness:.4f}")
        print(f"  ICP RMSE: {icp_result.inlier_rmse:.4f}")
        
        if icp_result.fitness >= 0.5 and icp_result.fitness >= best_result.fitness * 0.9:
            final_transform = icp_result.transformation @ current_transform
        else:
            print(f"  Refinement degraded fitness ({best_result.fitness:.4f} -> {icp_result.fitness:.4f}). Keeping global result.")
            final_transform = current_transform
    else:
        final_transform = current_transform
    
    elapsed = time.time() - start
    
    # Quality check
    print("\n[6. Final Quality]")
    pcd_gear_final = copy.deepcopy(pcd_gear)
    pcd_gear_final.transform(final_transform)
    
    fitness, rmse, inliers = compute_quality(pcd_gear_final, pcd_data, np.eye(4), voxel_fine * 3)
    
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Alignment Fitness: {fitness:.4f} (inliers: {inliers}/{len(pcd_gear.points)})")
    print(f"Alignment RMSE: {rmse:.4f}")
    print(f"Time: {elapsed:.2f}s")
    print(f"\nTransformation:\n{final_transform}")
    
    return final_transform, pcd_gear, pcd_data, mesh, rmse


def refine_alignment_normal_only(pcd_gear, pcd_data, mesh, initial_transform, voxel_size, num_iterations=3):
    """
    Refine alignment using only 'normal' points (excluding outliers/defects).
    """
    print("\n[Normal-Only Alignment Refinement]")
    
    current_transform = initial_transform.copy()
    
    for iteration in range(num_iterations):
        # Transform gear to current position
        gear_temp = copy.deepcopy(pcd_gear)
        gear_temp.transform(current_transform)
        
        # Compute distances from data points to mesh
        mesh_t = copy.deepcopy(mesh)
        mesh_t.transform(current_transform)
        
        mesh_tensor = o3d.t.geometry.TriangleMesh.from_legacy(mesh_t)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(mesh_tensor)
        
        pts = np.asarray(pcd_data.points).astype(np.float32)
        closest = scene.compute_closest_points(o3d.core.Tensor(pts))
        closest_pts = closest['points'].numpy()
        distances = np.linalg.norm(pts - closest_pts, axis=1)
        
        # IQR-based outlier detection
        median_dist = np.median(distances)
        reasonable_mask = distances < median_dist * 3
        
        if np.sum(reasonable_mask) < 100:
            print(f"  Iteration {iteration+1}: Too few reasonable points, stopping")
            break
        
        reasonable_distances = distances[reasonable_mask]
        q1, q3 = np.percentile(reasonable_distances, [25, 75])
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        
        normal_mask = distances <= upper_bound
        num_normal = np.sum(normal_mask)
        
        print(f"  Iteration {iteration+1}: {num_normal}/{len(distances)} normal points (threshold={upper_bound:.3f})")
        
        if num_normal < 100:
            print(f"  Too few normal points, stopping refinement")
            break
        
        # Create point cloud of only normal points
        pcd_normal = o3d.geometry.PointCloud()
        pcd_normal.points = o3d.utility.Vector3dVector(pts[normal_mask])
        
        # Preprocess for ICP
        pcd_normal_down = pcd_normal.voxel_down_sample(voxel_size)
        pcd_normal_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
        )
        
        gear_down = gear_temp.voxel_down_sample(voxel_size)
        gear_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
        )
        
        # Run ICP on normal points only
        result = o3d.pipelines.registration.registration_icp(
            gear_down, pcd_normal_down, voxel_size * 2, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
        )
        
        # Update transformation only if fitness is good
        if result.fitness < 0.9:
             print(f"    ICP fitness too low ({result.fitness:.4f}), stopping refinement to prevent drift")
             break
             
        current_transform = result.transformation @ current_transform
        print(f"    ICP fitness: {result.fitness:.4f}, RMSE: {result.inlier_rmse:.4f}")
        
        # Check convergence
        if result.fitness > 0.95 and result.inlier_rmse < voxel_size * 0.5:
            print(f"  Converged at iteration {iteration+1}")
            break
    
    return current_transform


def compute_mesh_footprint(mesh, transformation, padding=0.0):
    """Compute the 2D footprint (convex hull) of the mesh projected onto XY plane."""
    from scipy.spatial import ConvexHull
    
    mesh_t = copy.deepcopy(mesh)
    mesh_t.transform(transformation)
    
    vertices = np.asarray(mesh_t.vertices)
    
    # Project to XY plane
    xy_points = vertices[:, :2]
    
    # Compute convex hull
    try:
        hull = ConvexHull(xy_points)
        hull_points = xy_points[hull.vertices]
        
        # Add padding if specified
        if padding > 0:
            centroid = np.mean(hull_points, axis=0)
            directions = hull_points - centroid
            norms = np.linalg.norm(directions, axis=1, keepdims=True)
            norms[norms == 0] = 1
            directions = directions / norms
            hull_points = hull_points + directions * padding
    except Exception as e:
        print(f"  Warning: Convex hull failed ({e}), using bounding box")
        x_min, y_min = xy_points.min(axis=0) - padding
        x_max, y_max = xy_points.max(axis=0) + padding
        hull_points = np.array([
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max]
        ])
    
    z_min, z_max = vertices[:, 2].min(), vertices[:, 2].max()
    
    print(f"  Mesh footprint: {len(hull_points)} hull vertices")
    print(f"  Z range: [{z_min:.2f}, {z_max:.2f}]")
    
    return hull_points, (z_min, z_max)


def is_inside_footprint(points_2d, hull_points):
    """Check if 2D points are inside the convex hull footprint."""
    from matplotlib.path import Path
    
    hull_path = Path(hull_points)
    inside = hull_path.contains_points(points_2d)
    
    return inside


def segment_points(pcd_data, mesh, transformation, threshold=1.0):
    """Identify gear vs background points."""
    mesh_t = copy.deepcopy(mesh)
    mesh_t.transform(transformation)
    
    mesh_tensor = o3d.t.geometry.TriangleMesh.from_legacy(mesh_t)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_tensor)
    
    pts = np.asarray(pcd_data.points).astype(np.float32)
    closest = scene.compute_closest_points(o3d.core.Tensor(pts))
    closest_pts = closest['points'].numpy()
    
    distances = np.linalg.norm(pts - closest_pts, axis=1)
    is_gear = distances < threshold
    
    print(f"\n[Segmentation (threshold={threshold})]")
    print(f"  STL points: {np.sum(is_gear)} / {len(distances)}")
    print(f"  Percentage: {np.sum(is_gear)/len(distances)*100:.1f}%")
    
    return is_gear, distances


def classify_points_cluster_first(pcd_data, mesh, transformation, hull_points=None, z_range=None, forced_normal_threshold=None):
    """
    NEW APPROACH: Cluster abnormal points FIRST, then classify clusters.
    
    Strategy:
    1. Compute distances to mesh
    2. Identify "normal" points (P75 threshold)
    3. Cluster ALL remaining points (both edges and defects)
    4. Classify clusters based on size, coherence, and average distance
    
    Classes:
        1 = NORMAL: Points close to mesh surface
        2 = ANOMALY: Large, cohesive clusters with high distance (defects)
        3 = BACKGROUND: Points outside footprint
        4 = EDGE_ARTIFACT: Small clusters or scattered points near gear edges
    """
    mesh_t = copy.deepcopy(mesh)
    mesh_t.transform(transformation)
    
    pts = np.asarray(pcd_data.points).astype(np.float32)
    
    # --- Step 1: Compute distances to mesh ---
    mesh_tensor = o3d.t.geometry.TriangleMesh.from_legacy(mesh_t)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_tensor)
    
    query = o3d.core.Tensor(pts)
    closest = scene.compute_closest_points(query)
    closest_pts = closest['points'].numpy()
    distances = np.linalg.norm(pts - closest_pts, axis=1)
    
    print(f"\n[Distance Statistics]")
    p25, p50, p75, p90, p95, p99 = np.percentile(distances, [25, 50, 75, 90, 95, 99])
    print(f"  P25={p25:.3f}, Median={p50:.3f}, P75={p75:.3f}")
    print(f"  P90={p90:.3f}, P95={p95:.3f}, P99={p99:.3f}")
    
    # --- Step 2: Footprint-based background detection ---
    if hull_points is not None:
        print(f"\n[Footprint-Based Background Detection]")
        pts_xy = pts[:, :2]
        inside_footprint = is_inside_footprint(pts_xy, hull_points)
        
        if z_range is not None:
            z_min, z_max = z_range
            z_margin = (z_max - z_min) * 1.0  # Allow defects to protrude
            in_z_range = (pts[:, 2] >= z_min - z_margin) & (pts[:, 2] <= z_max + z_margin)
            inside_footprint_strict = inside_footprint & in_z_range
        else:
            inside_footprint_strict = inside_footprint
        
        outside_footprint = ~inside_footprint
        print(f"  Inside footprint: {np.sum(inside_footprint_strict)}")
        print(f"  Outside footprint: {np.sum(outside_footprint)}")
    else:
        outside_footprint = np.zeros(len(pts), dtype=bool)
        inside_footprint_strict = np.ones(len(pts), dtype=bool)
    
    # --- Step 3: Identify "NORMAL" points ---
    if forced_normal_threshold is not None:
        normal_threshold = forced_normal_threshold
        print(f"\n[Normal Threshold]")
        print(f"  Using CACHED Normal threshold: {normal_threshold:.3f}")
    else:
        # Use P75 threshold - 75% of footprint points are definitely normal
        footprint_distances = distances[inside_footprint_strict]
        if len(footprint_distances) > 100:
            normal_threshold = np.percentile(footprint_distances, 75)
        else:
            normal_threshold = p75
        
        # Ensure minimum threshold
        normal_threshold = max(normal_threshold, p50 * 1.5, 0.15)
        
        print(f"\n[Normal Threshold]")
        print(f"  NORMAL threshold: {normal_threshold:.3f} (P75 of footprint)")
    
    mask_normal = (distances <= normal_threshold) & inside_footprint_strict
    
    # --- Step 4: Cluster ALL abnormal points (inside footprint) ---
    # This includes both defects AND background/part artifacts
    
    # Filter out floor/base points (bottom 5%) to treat as background if not normal
    # This addresses the user's request to classify the base (table/platform) as Background
    z_vals = pts[:, 2]
    floor_threshold = np.min(z_vals) + (np.max(z_vals) - np.min(z_vals)) * 0.05
    is_floor = z_vals < floor_threshold
    
    # Abnormal candidates are inside footprint, not matching surface, AND not floor
    mask_abnormal_inside = (distances > normal_threshold) & inside_footprint_strict & (~is_floor)
    abnormal_indices = np.where(mask_abnormal_inside)[0]
    
    print(f"\n[Clustering Abnormal Points]")
    print(f"  Abnormal candidates: {len(abnormal_indices)}")
    
    # Initialize classes
    classes = np.full(len(distances), 3, dtype=int)  # Default: BACKGROUND
    classes[outside_footprint] = 3  # Definitely BACKGROUND
    classes[mask_normal] = 1  # Definitely NORMAL
    
    if len(abnormal_indices) > 0:
        pcd_abnormal = o3d.geometry.PointCloud()
        pcd_abnormal.points = o3d.utility.Vector3dVector(pts[abnormal_indices])
        
        # Calculate mesh scale
        vertices = np.asarray(mesh_t.vertices)
        mesh_extent = np.max(vertices.max(axis=0) - vertices.min(axis=0))
        
        # CLUSTERING PARAMETERS
        # eps: Defects should be spatially coherent
        eps_val = max(mesh_extent * 0.04, normal_threshold * 3.0, 0.8)
        
        # min_points: Keep it low to capture the entire defect cluster
        # We'll filter by cluster size AFTER clustering
        min_pts = max(10, int(len(abnormal_indices) * 0.005))
        
        print(f"  Clustering: eps={eps_val:.2f}, min_points={min_pts}")
        
        labels = np.array(pcd_abnormal.cluster_dbscan(eps=eps_val, min_points=min_pts))
        
        # Get unique cluster IDs
        unique_labels = np.unique(labels[labels >= 0])
        print(f"  Found {len(unique_labels)} clusters")
        
        # --- Step 5: Classify each cluster ---
        for cluster_id in unique_labels:
            cluster_mask = labels == cluster_id
            cluster_indices = abnormal_indices[cluster_mask]
            cluster_points = pts[cluster_indices]
            cluster_distances = distances[cluster_indices]
            
            # Cluster statistics
            cluster_size = len(cluster_indices)
            avg_distance = np.mean(cluster_distances)
            max_distance = np.max(cluster_distances)
            median_distance = np.median(cluster_distances)
            
            # Cluster spatial extent
            cluster_extent = np.max(cluster_points.max(axis=0) - cluster_points.min(axis=0))
            
            # DECISION CRITERIA for ANOMALY vs BACKGROUND_OR_PART
            
            # Criterion 1: Size - Real defects have many points
            size_score = cluster_size >= 50  # At least 50 points
            
            # Criterion 2: High average distance
            distance_score = avg_distance > normal_threshold * 2.0
            
            # Criterion 3: Spatial coherence - defects have extent
            coherence_score = cluster_extent > mesh_extent * 0.05
            
            # Criterion 4: Consistent high distance (not just edge noise)
            consistency_score = median_distance > normal_threshold * 1.5
            
            # Decision: At least 3 out of 4 criteria
            is_defect = sum([size_score, distance_score, coherence_score, consistency_score]) >= 3
            
            if is_defect:
                classes[cluster_indices] = 2  # ANOMALY
                print(f"    Cluster {cluster_id}: DEFECT - {cluster_size} pts, avg_dist={avg_distance:.3f}, extent={cluster_extent:.2f}")
            else:
                # USER REQUEST: Comment out BACKGROUND_OR_PART logic and classify as NORMAL
                # classes[cluster_indices] = 4  # BACKGROUND_OR_PART
                classes[cluster_indices] = 1  # NORMAL
                # print(f"    Cluster {cluster_id}: BACKGROUND_OR_PART - {cluster_size} pts, avg_dist={avg_distance:.3f}, extent={cluster_extent:.2f}")
                print(f"    Cluster {cluster_id}: NORMAL (reclassified) - {cluster_size} pts, avg_dist={avg_distance:.3f}")
        
        # Noise points (not in any cluster) → BACKGROUND_OR_PART
        noise_mask = labels == -1
        noise_indices = abnormal_indices[noise_mask]
        # USER REQUEST: Comment out BACKGROUND_OR_PART logic and classify as NORMAL
        # classes[noise_indices] = 4
        classes[noise_indices] = 1 # Reclassify as Normal
        # print(f"  Noise points: {len(noise_indices)} → BACKGROUND_OR_PART")
        print(f"  Noise points: {len(noise_indices)} → NORMAL (reclassified)")
    
    # --- Step 6: Print statistics ---
    print(f"\n[Final Classification]")
    print(f"  Class 1 (NORMAL):       {np.sum(classes == 1):6d} ({np.sum(classes == 1)/len(classes)*100:5.1f}%)")
    print(f"  Class 2 (DEFECT):       {np.sum(classes == 2):6d} ({np.sum(classes == 2)/len(classes)*100:5.1f}%)")
    print(f"  Class 3 (BACKGROUND):   {np.sum(classes == 3):6d} ({np.sum(classes == 3)/len(classes)*100:5.1f}%)")
    # print(f"  Class 4 (BACKGROUND_OR_PART): {np.sum(classes == 4):6d} ({np.sum(classes == 4)/len(classes)*100:5.1f}%)")
    
    # Return thresholds for caching
    thresholds = {"normal": normal_threshold}
    
    return classes, distances, thresholds


def visualize(pcd_gear, pcd_data, transformation, classes, title="Point Cloud Visualization"):
    """Visualization with 4 classes."""
    print(f"\n[Visualization: {title}]")
    print("  GREEN   = NORMAL (class 1)")
    print("  RED     = DEFECT (class 2)")
    print("  BLUE    = BACKGROUND (class 3)")
    # print("  MAGENTA = BACKGROUND_OR_PART (class 4)")
    
    gear_reg = copy.deepcopy(pcd_gear)
    gear_reg.transform(transformation)
    gear_reg.paint_uniform_color([0.5, 0.5, 0.5])
    
    data_colored = copy.deepcopy(pcd_data)
    colors = np.zeros((len(data_colored.points), 3))
    
    colors[classes == 1] = [0, 1, 0]  # Green
    colors[classes == 2] = [1, 0, 0]  # Red
    colors[classes == 3] = [0, 0, 1]  # Blue
    # colors[classes == 4] = [1, 0, 1]  # Magenta
    
    data_colored.colors = o3d.utility.Vector3dVector(colors)
    
    o3d.visualization.draw_plotly([gear_reg, data_colored])


def discover_datasets(directory="."):
    """Automatically find all gear<N>.stl and data<N>.csv pairs."""
    files = os.listdir(directory)
    gears = {}
    datas = {}
    
    gear_re = re.compile(r"gear(\d*)\.stl", re.IGNORECASE)
    data_re = re.compile(r"data(\d*)\.csv", re.IGNORECASE)
    
    for f in files:
        m_gear = gear_re.match(f)
        if m_gear:
            suffix = m_gear.group(1)
            gears[suffix] = f
            
        m_data = data_re.match(f)
        if m_data:
            suffix = m_data.group(1)
            datas[suffix] = f
    
    common_suffixes = sorted(list(set(gears.keys()) & set(datas.keys())), key=lambda x: int(x) if x else 0)
    
    datasets = []
    for s in common_suffixes:
        datasets.append((gears[s], datas[s], s))
        
    return datasets


if __name__ == "__main__":
    datasets = discover_datasets()
    
    if not datasets:
        print("No datasets found (gear<N>.stl and data<N>.csv pairs).")
    else:
        print(f"Found {len(datasets)} datasets: {[d[0] for d in datasets]}")
    
    for mesh_file, data_file, suffix in datasets:
        print(f"\n\n{'#'*80}")
        print(f"PROCESSING DATASET: {mesh_file} / {data_file}")
        print(f"{'#'*80}")
        
        metadata_file = f"registration_metadata{suffix}.json"
        metadata = load_registration_metadata(metadata_file)
        
        if metadata:
            print(f"  Using cached results from {metadata['timestamp']}")
            refined_transform = metadata["transformation"]
            cached_thresholds = metadata["thresholds"]
            
            # Load Data (skip registration)
            print("\n[Loading Data]")
            pcd_gear, mesh = load_or_sample_gear(mesh_file, num_points=30000)
            
            data = np.genfromtxt(data_file, delimiter=",")
            data = data[~np.isnan(data).any(axis=1)]
            pcd_data = o3d.geometry.PointCloud()
            pcd_data.points = o3d.utility.Vector3dVector(data)
            
            # Use cached footprint if available
            if metadata.get("mesh_footprint") is not None:
                print("\n[Using Cached Footprint]")
                hull_points = metadata["mesh_footprint"]
                # Recompute Z range quickly
                mesh_t = copy.deepcopy(mesh)
                mesh_t.transform(refined_transform)
                vertices = np.asarray(mesh_t.vertices)
                z_range = (vertices[:, 2].min(), vertices[:, 2].max())
            else:
                print("\n[Computing Mesh Footprint]")
                hull_points, z_range = compute_mesh_footprint(mesh, refined_transform, padding=1.0)
                
            # Dummy variables for skipped steps
            rmse = metadata["rmse"]
            is_gear = np.zeros(len(data), dtype=bool) # Placeholder
            
        else:
            cache_file = f"gear_sampled{suffix}.npy"
            cached_thresholds = None
            
            # Step 1: Initial registration
            transform, pcd_gear, pcd_data, mesh, rmse = run_registration(
                mesh_file, data_file, num_attempts=50, num_points=30000, cache_path=cache_file
            )
            
            # Step 2: Binary segmentation
            is_gear, initial_distances = segment_points(pcd_data, mesh, transform, threshold=1.0)
            
            # Step 3: Refine alignment
            gear_range = np.max(np.asarray(pcd_gear.points).max(axis=0) - np.asarray(pcd_gear.points).min(axis=0))
            voxel_size = gear_range / 50.0
            
            refined_transform = refine_alignment_normal_only(
                pcd_gear, pcd_data, mesh, transform, voxel_size, num_iterations=3
            )
            
            # Step 4: Compute footprint
            print("\n[Computing Mesh Footprint]")
            hull_points, z_range = compute_mesh_footprint(mesh, refined_transform, padding=1.0)
        
        # Step 5: CLUSTER-FIRST classification
        classes, distances, computed_thresholds = classify_points_cluster_first(
            pcd_data, mesh, refined_transform,
            hull_points=hull_points,
            z_range=z_range
        )
        
        # Save metadata if it was a fresh run
        if metadata is None:
            save_registration_metadata(
                metadata_file, 
                refined_transform, 
                rmse, 
                computed_thresholds,
                hull_points
            )
        
        # Save files
        print(f"\n[Saving Files for dataset {suffix}]")
        np.savetxt(f"registration_transformation{suffix}.txt", refined_transform)
        np.savetxt(f"point_stl_membership{suffix}.txt", is_gear.astype(int), fmt='%d')
        np.savetxt(f"point_to_mesh_distances{suffix}.txt", distances)
        
        gear_reg = copy.deepcopy(pcd_gear)
        gear_reg.transform(refined_transform)
        o3d.io.write_point_cloud(f"gear_registered{suffix}.ply", gear_reg)
        
        data_points = np.asarray(pcd_data.points)
        classified_data = np.column_stack((data_points, classes))
        
        header = "x,y,z,class"
        np.savetxt(f"classified_data{suffix}.csv", classified_data, delimiter=",", header=header, comments="", fmt=["%.8f", "%.8f", "%.8f", "%d"])
        print(f"  ✓ classified_data{suffix}.csv")
        
        # Visualize
        visualize(pcd_gear, pcd_data, refined_transform, classes, title=f"Dataset {suffix if suffix else '0'}")
    
    print("\nAll datasets processed!")