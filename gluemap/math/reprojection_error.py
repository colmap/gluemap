"""Reprojection-error computation and the canonical reconstruction filter that
consumes those errors. Mutating filters live here (rather than under
``controllers/``) so that error computation and error-based filtering form a
single self-contained unit without circular imports."""

import logging
from enum import Enum

import numpy as np
import pycolmap

logger = logging.getLogger(__name__)


class ReprojectionErrorType(Enum):
    PIXEL = "pixel"
    NORMALIZED = "normalized"
    ANGULAR = "angular"


def compute_point_error(
    world_point: np.ndarray,
    R: np.ndarray,
    center: np.ndarray,
    observed: np.ndarray,
    camera: pycolmap.Camera,
    error_type: ReprojectionErrorType = ReprojectionErrorType.PIXEL,
    is_negative_depth: bool = False,
) -> float:
    """
    Compute error for a single 3D point observation.

    Args:
        world_point: 3D point in world coordinates (3,).
        R: Rotation matrix world-to-camera (3, 3).
        center: Camera center in world coordinates (3,).
        observed: Observed 2D point (2,).
        camera: pycolmap.Camera with intrinsics.
        error_type: PIXEL (pixels), NORMALIZED (pixels / focal), or ANGULAR (degrees).
        is_negative_depth: If True, negate X_cam before projection.

    Returns:
        Error value (float), or float("inf") for degenerate cases.
    """
    X_cam = R @ (world_point - center)

    if is_negative_depth:
        X_cam = -X_cam

    if error_type == ReprojectionErrorType.ANGULAR:
        if not is_negative_depth and X_cam[2] <= 0:
            return float("inf")
        norm_3d = np.linalg.norm(X_cam)
        if norm_3d < 1e-10:
            return float("inf")
        ray_3d = X_cam / norm_3d
        ray_2d = camera.cam_from_img(observed)
        ray_obs = np.array([ray_2d[0], ray_2d[1], 1.0])
        ray_obs = ray_obs / np.linalg.norm(ray_obs)
        cos_angle = np.clip(np.dot(ray_3d, ray_obs), -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))
    else:
        if X_cam[2] <= 0:
            return float("inf")
        projected = camera.img_from_cam(X_cam)
        pixel_error = np.sqrt(
            (projected[0] - observed[0]) ** 2
            + (projected[1] - observed[1]) ** 2
        )
        if error_type == ReprojectionErrorType.NORMALIZED:
            return pixel_error / camera.focal_length
        return pixel_error


def _compute_errors_batch(
    X_cam: np.ndarray,
    observed: np.ndarray,
    camera: pycolmap.Camera,
    neg_mask: np.ndarray,
    error_type: ReprojectionErrorType,
) -> np.ndarray:
    """
    Compute errors for a batch of observations sharing the same camera.

    Args:
        X_cam: Camera-space points (N, 3).
        observed: Observed 2D points (N, 2).
        camera: pycolmap.Camera.
        neg_mask: Boolean mask (N,) — True for negative-depth observations.
        error_type: PIXEL, NORMALIZED, or ANGULAR.

    Returns:
        Errors array (N,).
    """
    N = X_cam.shape[0]
    errors = np.full(N, float("inf"))

    # Negate camera-space coords for negative-depth observations
    X_proc = X_cam.copy()
    if np.any(neg_mask):
        X_proc[neg_mask] = -X_proc[neg_mask]

    if error_type == ReprojectionErrorType.ANGULAR:
        # For non-negative-depth, require positive Z
        valid = X_proc[:, 2] > 0
        norms = np.linalg.norm(X_proc, axis=1)
        valid &= norms > 1e-10
        if not np.any(valid):
            return errors

        rays_3d = X_proc[valid] / norms[valid, np.newaxis]
        rays_2d = camera.cam_from_img(observed[valid])  # (M, 2)
        rays_obs = np.column_stack([rays_2d, np.ones(rays_2d.shape[0])])
        rays_obs = rays_obs / np.linalg.norm(rays_obs, axis=1, keepdims=True)
        cos_angles = np.clip(np.sum(rays_3d * rays_obs, axis=1), -1.0, 1.0)
        errors[valid] = np.degrees(np.arccos(cos_angles))
    else:
        valid = X_proc[:, 2] > 0
        if not np.any(valid):
            return errors

        projected = camera.img_from_cam(X_proc[valid])  # (M, 2)
        pixel_errors = np.linalg.norm(projected - observed[valid], axis=1)
        if error_type == ReprojectionErrorType.NORMALIZED:
            pixel_errors = pixel_errors / camera.focal_length
        errors[valid] = pixel_errors

    return errors


def compute_all_errors_from_reconstruction(
    reconstruction: pycolmap.Reconstruction,
    error_type: ReprojectionErrorType = ReprojectionErrorType.PIXEL,
    negative_depth_observations: dict[int, set] | None = None,
    virtual_point_start: dict[int, int] | None = None,
    fisheye_cameras: dict[int, pycolmap.Camera] | None = None,
) -> dict[int, list[tuple[int, int, float]]]:
    """
    Compute errors for all 3D point track observations in a reconstruction.

    Args:
        reconstruction: pycolmap.Reconstruction with cameras, images, and points3D.
        error_type: PIXEL, NORMALIZED, or ANGULAR.
        negative_depth_observations: Dict[image_id, Set[point2D_idx]] for
            observations where the 3D point has negative depth.
        virtual_point_start: Dict[image_id, int] — where virtual points start per image.
            If provided along with fisheye_cameras, virtual observations use the
            fisheye camera for error computation.
        fisheye_cameras: Dict[camera_id, pycolmap.Camera] — SIMPLE_FISHEYE cameras
            indexed by camera_id, used for virtual point observations.

    Returns:
        Dict mapping point3D_id -> List[(image_id, point2D_idx, error)].
    """
    # --- First pass: gather observations per image ---
    # obs_by_image[image_id] = list of (point3D_id, pt_idx, world_xyz)
    obs_by_image: dict[int, list[tuple[int, int, np.ndarray]]] = {}
    # Track invalid observations (missing image/camera/out-of-bounds)
    invalid_obs: dict[int, list[tuple[int, int]]] = {}

    for point3D_id, point3D in reconstruction.points3D.items():
        world_point = point3D.xyz
        if world_point is None or np.all(world_point == 0):
            continue

        for elem in point3D.track.elements:
            image_id, pt_idx = elem.image_id, elem.point2D_idx

            if image_id not in reconstruction.images:
                invalid_obs.setdefault(point3D_id, []).append(
                    (image_id, pt_idx)
                )
                continue

            image = reconstruction.images[image_id]
            if pt_idx >= len(image.points2D):
                invalid_obs.setdefault(point3D_id, []).append(
                    (image_id, pt_idx)
                )
                continue

            if image.camera_id not in reconstruction.cameras:
                invalid_obs.setdefault(point3D_id, []).append(
                    (image_id, pt_idx)
                )
                continue

            obs_by_image.setdefault(image_id, []).append(
                (point3D_id, pt_idx, world_point)
            )

    # --- Second pass: vectorized per-image error computation ---
    # error_results[(point3D_id, image_id, pt_idx)] = error
    error_results: dict[tuple[int, int, int], float] = {}

    for image_id, obs_list in obs_by_image.items():
        image = reconstruction.images[image_id]
        camera_id = image.camera_id
        camera = reconstruction.cameras[camera_id]
        n_pts2d = len(image.points2D)
        pose = image.cam_from_world()

        # Determine virtual point start for this image
        vp_start = (
            virtual_point_start.get(image_id, n_pts2d)
            if virtual_point_start is not None
            else n_pts2d
        )
        has_fisheye = (
            fisheye_cameras is not None and camera_id in fisheye_cameras
        )

        # Separate pinhole vs fisheye observations
        pinhole_obs = []
        fisheye_obs = []
        for point3D_id, pt_idx, world_xyz in obs_list:
            if pt_idx >= vp_start and has_fisheye:
                fisheye_obs.append((point3D_id, pt_idx, world_xyz))
            else:
                pinhole_obs.append((point3D_id, pt_idx, world_xyz))

        # Process each camera group
        for cam, group in [(camera, pinhole_obs)]:
            if not group:
                continue
            _process_obs_group(
                group,
                image,
                pose,
                cam,
                image_id,
                negative_depth_observations,
                error_type,
                error_results,
            )

        if has_fisheye and fisheye_obs:
            _process_obs_group(
                fisheye_obs,
                image,
                pose,
                fisheye_cameras[camera_id],
                image_id,
                negative_depth_observations,
                error_type,
                error_results,
            )

    # --- Third pass: assemble errors_per_track ---
    errors_per_track: dict[int, list[tuple[int, int, float]]] = {}

    for point3D_id, point3D in reconstruction.points3D.items():
        world_point = point3D.xyz
        if world_point is None or np.all(world_point == 0):
            continue

        track_errors = []

        # Add invalid observations
        if point3D_id in invalid_obs:
            for image_id, pt_idx in invalid_obs[point3D_id]:
                track_errors.append((image_id, pt_idx, float("inf")))

        # Add computed errors
        for elem in point3D.track.elements:
            image_id, pt_idx = elem.image_id, elem.point2D_idx
            key = (point3D_id, image_id, pt_idx)
            if key in error_results:
                track_errors.append((image_id, pt_idx, error_results[key]))

        errors_per_track[point3D_id] = track_errors

    return errors_per_track


def _process_obs_group(
    group: list[tuple[int, int, np.ndarray]],
    image: pycolmap.Image,
    pose,
    camera: pycolmap.Camera,
    image_id: int,
    negative_depth_observations: dict[int, set] | None,
    error_type: ReprojectionErrorType,
    error_results: dict[tuple[int, int, int], float],
) -> None:
    """Process a group of observations for one image+camera, writing results into error_results."""
    point3D_ids = [g[0] for g in group]
    pt_idxs = [g[1] for g in group]
    world_points = np.array([g[2] for g in group])  # (N, 3)
    observed = np.array([image.points2D[idx].xy for idx in pt_idxs])  # (N, 2)

    # Batched world-to-camera transform
    X_cam = pose * world_points  # (N, 3)

    # Build negative-depth mask
    if (
        negative_depth_observations is not None
        and image_id in negative_depth_observations
    ):
        neg_set = negative_depth_observations[image_id]
        neg_mask = np.array([idx in neg_set for idx in pt_idxs], dtype=bool)
    else:
        neg_mask = np.zeros(len(group), dtype=bool)

    errors = _compute_errors_batch(
        X_cam, observed, camera, neg_mask, error_type
    )

    for i, (p3d_id, pt_idx) in enumerate(zip(point3D_ids, pt_idxs)):
        error_results[(p3d_id, image_id, pt_idx)] = float(errors[i])


def filter_observations_by_error(
    reconstruction: pycolmap.Reconstruction,
    errors_per_track: dict[int, list[tuple[int, int, float]]],
    reproj_threshold: float,
    min_track_length: int,
) -> tuple[int, int]:
    """
    Filter observations with high reprojection errors from a reconstruction.

    Wholesale track removal goes through ``Reconstruction.delete_point3D``.
    Per-observation outliers are dropped by hand with
    ``Point3D.track.delete_element`` + ``Image.reset_point3D_for_point2D``
    (mirroring COLMAP's C++ core), which keeps ``image.points2D[i].point3D_id``
    consistent with the track without triggering pycolmap's auto-delete of
    tracks that drop below 2 elements.

    Args:
        reconstruction: pycolmap.Reconstruction (modified in place).
        errors_per_track: Reprojection errors from
            ``compute_all_errors_from_reconstruction``.
        reproj_threshold: Maximum allowed reprojection error.
        min_track_length: Minimum observations to keep a track; tracks with
            fewer surviving inliers are deleted wholesale.

    Returns:
        (num_observations_removed, num_tracks_removed).
    """
    num_observations_removed = 0
    num_tracks_removed = 0

    for point3D_id, track_errors in errors_per_track.items():
        if point3D_id not in reconstruction.points3D:
            continue

        point3D = reconstruction.points3D[point3D_id]
        elements = list(point3D.track.elements)

        outlier_obs = [] # (image_id, pt_idx) pairs to delete individaully
        inlier_count = 0

        for (image_id, pt_idx, error), elem in zip(track_errors, elements):
            if error < reproj_threshold:
                inlier_count += 1
            else:
                outlier_obs.append((elem.image_id, elem.point2D_idx))

        if inlier_count < min_track_length:
            # Not enough inliers the track at all.
            reconstruction.delete_point3D(point3D_id)
            num_observations_removed += len(elements)
            num_tracks_removed += 1
        else:
            for image_id, pt_idx in outlier_obs:
                image = reconstruction.images[image_id]
                point3D.track.delete_element(image_id, pt_idx)
                image.reset_point3D_for_point2D(pt_idx)
                num_observations_removed += 1

    return num_observations_removed, num_tracks_removed


_PYCOLMAP_ERROR_TYPE = {
    ReprojectionErrorType.PIXEL: pycolmap.ReprojectionErrorType.PIXEL,
    ReprojectionErrorType.NORMALIZED: pycolmap.ReprojectionErrorType.NORMALIZED,
    ReprojectionErrorType.ANGULAR: pycolmap.ReprojectionErrorType.ANGULAR,
}


def filter_reconstruction_by_reprojection_error_colmap(
    reconstruction: pycolmap.Reconstruction,
    error_type: ReprojectionErrorType,
    error_threshold: float,
    min_track_length: int = 2,
    log_level: int = logging.INFO,
    log_prefix: str = "",
) -> tuple[int, int]:
    """Filter the reconstruction in place via pycolmap's ObservationManager.

    Cannot account for virtual points, negative-depth observations, or
    per-image fisheye overrides — callers must only invoke this when those are
    absent. Returns (observations_removed, tracks_removed).
    """
    num_points_before = len(reconstruction.points3D)
    obs_manager = pycolmap.ObservationManager(reconstruction)
    points3d_ids = set(reconstruction.points3D.keys())
    obs_removed = obs_manager.filter_points3D_with_large_reprojection_error(
        error_threshold,
        points3d_ids,
        _PYCOLMAP_ERROR_TYPE[error_type],
    )
    obs_manager.filter_points3D_with_short_tracks(min_track_length=min_track_length)
    tracks_removed = num_points_before - len(reconstruction.points3D)
    logger.log(
        log_level,
        f"{log_prefix}pycolmap filter removed {obs_removed} observations, "
        f"{tracks_removed} tracks",
    )
    return obs_removed, tracks_removed


def filter_reconstruction_by_reprojection_error(
    reconstruction: pycolmap.Reconstruction,
    error_type: ReprojectionErrorType,
    error_threshold: float,
    min_track_length: int = 2,
    negative_depth_observations: dict[int, set] | None = None,
    virtual_point_start: dict[int, int] | None = None,
    fisheye_cameras: dict[int, pycolmap.Camera] | None = None,
    log_level: int = logging.INFO,
    log_prefix: str = "",
) -> tuple[int, int]:
    """
    Compute reprojection errors for one reconstruction, log error statistics,
    and filter observations / tracks above ``error_threshold`` in place.

    When the reconstruction has no virtual points, negative-depth observations,
    or per-image fisheye overrides, the fast pycolmap ``ObservationManager``
    filter is attempted first, falling back to the in-Python error computation
    on any exception. Returns ``(observations_removed, tracks_removed)``.
    """
    can_use_colmap = (
        not negative_depth_observations
        and not virtual_point_start
        and fisheye_cameras is None
    )
    if can_use_colmap:
        try:
            return filter_reconstruction_by_reprojection_error_colmap(
                reconstruction,
                error_type,
                error_threshold,
                min_track_length=min_track_length,
                log_level=log_level,
                log_prefix=log_prefix,
            )
        except Exception as e:
            logger.warning(
                f"{log_prefix}pycolmap observation filter failed ({e}); "
                f"falling back to in-Python reprojection-error filter"
            )

    errors_per_track = compute_all_errors_from_reconstruction(
        reconstruction,
        error_type,
        negative_depth_observations or {},
        virtual_point_start=virtual_point_start or {},
        fisheye_cameras=fisheye_cameras,
    )

    unit = " deg" if error_type == ReprojectionErrorType.ANGULAR else ""
    finite_errors = [
        e
        for errs in errors_per_track.values()
        for _, _, e in errs
        if e < float("inf")
    ]
    if finite_errors:
        arr = np.asarray(finite_errors)
        pct = 100 * np.sum(arr < error_threshold) / len(arr)
        logger.log(
            log_level,
            f"{log_prefix}{error_type.value} reprojection errors: "
            f"mean={np.mean(arr):.4f}{unit}, "
            f"median={np.median(arr):.4f}{unit}, "
            f"max={np.max(arr):.4f}{unit}, "
            f"< {error_threshold}{unit}: {pct:.1f}%",
        )

    obs_removed, tracks_removed = filter_observations_by_error(
        reconstruction,
        errors_per_track,
        error_threshold,
        min_track_length,
    )
    logger.log(
        log_level,
        f"{log_prefix}removed {obs_removed} observations, "
        f"{tracks_removed} tracks",
    )
    return obs_removed, tracks_removed


