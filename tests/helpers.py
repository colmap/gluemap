"""Shared test helpers for synthetic SfM reconstruction tests."""

import copy

import numpy as np
import pycolmap
import torch


# TODO: add an option to generate grid like reconstructions
# TODO: thoroughly verify the file
def create_synthetic_reconstruction(
    num_frames=8, num_points3D=100, num_rigs=1, seed=0
):
    """Create a pycolmap synthetic reconstruction with known ground truth."""
    pycolmap.set_random_seed(seed)
    opts = pycolmap.SyntheticDatasetOptions()
    opts.num_rigs = num_rigs
    opts.num_cameras_per_rig = 1
    opts.num_frames_per_rig = num_frames
    opts.num_points3D = num_points3D
    return pycolmap.synthesize_dataset(opts)


def extract_gt(reconstruction, zero_indexed=False):
    """Extract ground-truth rotations and centers from a reconstruction.

    Args:
        reconstruction: pycolmap reconstruction
        zero_indexed: if True, remap image_ids to 0..N-1 (needed for functions
            that assume node 0 exists, like initialize_mst_structures)
    """
    image_ids = sorted(reconstruction.reg_image_ids())
    gt_rotations = {}
    gt_centers = {}
    for img_id in image_ids:
        img = reconstruction.image(img_id)
        pose = img.cam_from_world()
        gt_rotations[img_id] = np.array(pose.rotation.matrix())
        gt_centers[img_id] = np.array(img.projection_center())

    if zero_indexed:
        id_map = {old: new for new, old in enumerate(image_ids)}
        image_ids = list(range(len(image_ids)))
        gt_rotations = {id_map[k]: v for k, v in gt_rotations.items()}
        gt_centers = {id_map[k]: v for k, v in gt_centers.items()}

    return image_ids, gt_rotations, gt_centers


def build_star_topology_full(image_ids):
    """One star per image, fully connected to all others."""
    stars = {}
    for i, center_id in enumerate(image_ids):
        neighbors = [img_id for img_id in image_ids if img_id != center_id]
        stars[i] = [center_id] + neighbors
    return stars


def build_star_topology_sparse(image_ids, gt_centers, k=3):
    """One star per image, connected to k nearest neighbors."""
    centers_array = np.array([gt_centers[i] for i in image_ids])
    stars = {}
    for i, center_id in enumerate(image_ids):
        dists = np.linalg.norm(centers_array - centers_array[i], axis=1)
        neighbor_indices = np.argsort(dists)[1 : k + 1]
        neighbors = [image_ids[j] for j in neighbor_indices]
        stars[i] = [center_id] + neighbors
    return stars


def build_predictions_dict(
    gt_rotations,
    gt_centers,
    stars,
    gt_scales,
    translation_noise_std=0.0,
    outlier_ratio=0.0,
    outlier_score=0.1,
    generate_points3d_virtual=False,
    num_points3d=50,
    rng=None,
):
    """Build predictions_dict from ground-truth data and star topology.

    Args:
        gt_rotations: dict image_id -> (3,3) rotation matrix
        gt_centers: dict image_id -> (3,) center
        stars: dict idx_star -> list of image_ids (first is center)
        gt_scales: list/array of floats, one per star
        translation_noise_std: stddev of Gaussian noise on translations
        outlier_ratio: fraction of edges to corrupt with random translations
        outlier_score: pose_score assigned to outlier edges
        generate_points3d_virtual: if True, generate synthetic 3D points per star
            and omit median_tri_angle (for initialize_mst_structures tests).
            If False (default), populate median_tri_angle directly.
        num_points3d: number of 3D points per star (used when generate_points3d_virtual=True)
        rng: numpy random generator
    """
    if rng is None:
        rng = np.random.default_rng(42)

    predictions_dict = {
        "indexes": {},
        "pose_scores": {},
        "extrinsics": {},
    }

    if not generate_points3d_virtual:
        predictions_dict["median_tri_angle"] = {}
    else:
        predictions_dict["points3d_virtual"] = {}

    for idx_star, img_ids in stars.items():
        n = len(img_ids)
        center_id = img_ids[0]
        scale = gt_scales[idx_star]

        extrinsics = torch.zeros(1, n, 3, 4)
        scores = torch.ones(1, n)

        for j in range(1, n):
            neighbor_id = img_ids[j]
            R_j = gt_rotations[neighbor_id]
            direction = gt_centers[neighbor_id] - gt_centers[center_id]
            t_ij = -R_j @ (scale * direction)

            if translation_noise_std > 0:
                t_ij = t_ij + rng.normal(0, translation_noise_std, size=3)

            if outlier_ratio > 0 and rng.random() < outlier_ratio:
                t_ij = rng.normal(0, 1, size=3)
                scores[0, j] = outlier_score

            extrinsics[0, j, :3, :3] = torch.from_numpy(
                R_j @ gt_rotations[center_id].T
            ).float()
            extrinsics[0, j, :3, 3] = torch.from_numpy(t_ij).float()

        predictions_dict["indexes"][idx_star] = img_ids
        predictions_dict["pose_scores"][idx_star] = scores
        predictions_dict["extrinsics"][idx_star] = extrinsics

        if not generate_points3d_virtual:
            predictions_dict["median_tri_angle"][idx_star] = torch.full(
                (n - 1,), 15.0
            )
        else:
            # Random 3D points in center camera frame with good depth spread
            xyz = np.zeros((num_points3d, 3))
            xyz[:, 0] = rng.uniform(-2, 2, size=num_points3d)
            xyz[:, 1] = rng.uniform(-2, 2, size=num_points3d)
            xyz[:, 2] = rng.uniform(5, 15, size=num_points3d)
            predictions_dict["points3d_virtual"][idx_star] = (
                torch.from_numpy(xyz).float().unsqueeze(0)
            )

    return predictions_dict


def perturb_points3D(reconstruction, fraction=1.0, noise_std=1.0, rng=None):
    """Add Gaussian noise to a fraction of 3D points in-place.

    Args:
        reconstruction: pycolmap.Reconstruction (modified in-place).
        fraction: fraction of points to perturb (0.0 to 1.0).
        noise_std: standard deviation of the Gaussian noise added to xyz.
        rng: numpy random generator; created with default seed if None.

    Returns:
        Set of perturbed point3D IDs.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    ids = [pid for pid, _ in reconstruction.points3D.items()]
    n_perturb = int(len(ids) * fraction)
    perturb_ids = rng.choice(ids, size=n_perturb, replace=False)
    for pid in perturb_ids:
        reconstruction.points3D[pid].xyz += rng.normal(0, noise_std, size=3)
    return set(perturb_ids)


def remap_to_original_ids(data_dict, reconstruction):
    """Remap 0-indexed dict keys back to the original pycolmap image IDs."""
    original_ids = sorted(reconstruction.reg_image_ids())
    return {original_ids[k]: v for k, v in data_dict.items()}


def evaluate_centers(gt_reconstruction, gt_rotations, recovered_centers):
    """Align recovered centers to GT via compare_reconstructions and return errors.

    Deep-copies the GT reconstruction, overwrites poses with recovered centers
    (keeping GT rotations), then uses pycolmap.compare_reconstructions to
    align via Sim3 and compute per-image errors.

    Returns list of ImageAlignmentError objects.
    """
    image_ids = sorted(recovered_centers.keys())

    rec = copy.deepcopy(gt_reconstruction)
    for img_id in image_ids:
        img = rec.image(img_id)
        R = gt_rotations[img_id]
        c = recovered_centers[img_id]
        t = -R @ c
        mat = np.hstack([R, t.reshape(3, 1)])
        new_pose = pycolmap.Rigid3d(mat)
        img.frame.set_cam_from_world(img.camera_id, new_pose)

    result = pycolmap.compare_reconstructions(
        rec,
        gt_reconstruction,
        alignment_error="proj_center",
        max_proj_center_error=100.0,
    )

    assert result is not None, (
        "compare_reconstructions returned None (alignment failed)"
    )
    return result["errors"]


def max_center_error(errors):
    """Extract max projection center error from alignment errors."""
    return max(e.proj_center_error for e in errors)
