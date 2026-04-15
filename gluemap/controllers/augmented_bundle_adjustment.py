import logging
import os
import shutil
import time
from copy import deepcopy

import numpy as np
import pycolmap
import pygluemap
import torch

from gluemap.controllers.bundle_adjustment import (
    IterativeBAOptions,
    build_negative_depth_observations,
    build_reconstruction_for_ba,
    build_virtual_point_start,
    filter_observations_by_error,
    initialize_world_points,
    iterative_bundle_adjustment,
)
from gluemap.estimators.track_establishment import (
    TrackEstablishmentOptions,
    establish_tracks_from_tracks_dict,
)
from gluemap.math.reprojection_error import (
    ReprojectionErrorType,
    compute_all_errors_from_reconstruction,
)
from gluemap.utils.colmap import (
    camera_from_intrinsics_matrix,
    merge_colmap_databases,
    prepare_glomap_prior,
)

logger = logging.getLogger(__name__)


def create_fisheye_cameras_and_rectify(
    reconstruction: pycolmap.Reconstruction,
    virtual_point_start: dict[int, int],
):
    """
    Create duplicate SIMPLE_FISHEYE cameras and rectify virtual 2D points.

    For each camera, creates a fisheye duplicate with the same params (f, cx, cy).
    For virtual point observations, unprojects with the original camera model and
    reprojects with the fisheye model so that 2D points are consistent with SIMPLE_FISHEYE.

    Args:
        reconstruction: pycolmap.Reconstruction (images' virtual points2D modified in-place)
        virtual_point_start: Dict[image_id, int] — where virtual points start per image

    Returns:
        Tuple of:
            fisheye_intrinsics_params: List[np.ndarray] indexed by camera_id, separate arrays
                for use as fixed parameter blocks in BA
            fisheye_cameras: Dict[camera_id, pycolmap.Camera] — SIMPLE_FISHEYE cameras
    """
    # Create fisheye cameras with same params as originals
    fisheye_cameras = {}
    for camera_id, camera in reconstruction.cameras.items():
        fisheye_cameras[camera_id] = pycolmap.Camera(
            model=pycolmap.CameraModelId.SIMPLE_FISHEYE,
            width=camera.width,
            height=camera.height,
            params=camera.params,
            camera_id=camera_id,
        )

    # Rectify virtual 2D points: unproject with original, reproject with fisheye
    num_rectified = 0
    for image_id, image in reconstruction.images.items():
        vp_start = virtual_point_start.get(image_id, len(image.points2D))
        if vp_start >= len(image.points2D):
            continue
        original_camera = reconstruction.cameras[image.camera_id]
        fisheye_cam = fisheye_cameras[image.camera_id]
        for pt_idx in range(vp_start, len(image.points2D)):
            pt2d = image.points2D[pt_idx].xy
            uv = original_camera.cam_from_img(pt2d)
            ray = np.array([uv[0], uv[1], 1.0])
            new_pt2d = fisheye_cam.img_from_cam(ray)
            image.points2D[pt_idx] = pycolmap.Point2D(new_pt2d)
            num_rectified += 1
    logger.info(
        f"Rectified {num_rectified} virtual 2D points to SIMPLE_FISHEYE projection"
    )

    # Build fisheye_intrinsics_params list (separate numpy arrays, will be fixed in BA)
    max_camera_id = max(reconstruction.cameras.keys())
    fisheye_intrinsics_params = [None] * (max_camera_id + 1)
    for camera_id, fisheye_cam in fisheye_cameras.items():
        fisheye_intrinsics_params[camera_id] = np.array(
            fisheye_cam.params, dtype=np.float64
        )

    return fisheye_intrinsics_params, fisheye_cameras


def select_tracks_from_merged(
    reconstruction: pycolmap.Reconstruction,
    sift_count: dict[int, int],
    min_num_support_abs: int = 512,
) -> None:
    """
    Selectively prune non-SIFT tracks from a merged reconstruction.
    Extracts track data as numpy arrays, delegates hot loops to C++ in pygluemap,
    then applies deletions via pycolmap.
    """
    point3d_ids = []
    track_img_ids = []
    track_pt2d_idxs = []
    track_lengths = []
    for p3d_id, p3d in reconstruction.points3D.items():
        elems = list(p3d.track.elements)
        point3d_ids.append(p3d_id)
        track_lengths.append(len(elems))
        for e in elems:
            track_img_ids.append(e.image_id)
            track_pt2d_idxs.append(e.point2D_idx)

    point3d_ids_np = np.array(point3d_ids, dtype=np.int64)
    track_img_ids_np = np.array(track_img_ids, dtype=np.int64)
    track_pt2d_idxs_np = np.array(track_pt2d_idxs, dtype=np.int64)
    track_lengths_np = np.array(track_lengths, dtype=np.int32)

    sc = {int(k): int(v) for k, v in sift_count.items()}

    ids_to_delete = pygluemap.compute_tracks_to_delete(
        point3d_ids_np,
        track_img_ids_np,
        track_pt2d_idxs_np,
        track_lengths_np,
        sc,
        min_num_support_abs,
    )

    for p3d_id in ids_to_delete:
        reconstruction.delete_point3D(int(p3d_id))


def update_poses_from_reconstruction(
    source_recon: pycolmap.Reconstruction,
    target_recon: pycolmap.Reconstruction,
) -> None:
    """
    Copy BA-optimized poses and camera intrinsics from source to target reconstruction.
    Matches images by name.
    """
    source_by_name = {
        img.name: (img_id, img) for img_id, img in source_recon.images.items()
    }
    for target_id, target_img in target_recon.images.items():
        if target_img.name in source_by_name:
            src_id, src_img = source_by_name[target_img.name]
            target_recon.frames[
                target_id
            ].rig_from_world = src_img.cam_from_world()
    # Copy camera intrinsics
    for cam_id, cam in source_recon.cameras.items():
        if cam_id in target_recon.cameras:
            target_recon.cameras[cam_id].params = cam.params


def filter_reconstruction_by_angular_error(
    reconstruction: pycolmap.Reconstruction,
    angular_error_threshold_deg: float,
    negative_depth_observations: dict,
    virtual_point_start: dict,
    fisheye_cameras=None,
):
    """Filter the reconstruction in place using an angular reprojection error
    threshold, logging error statistics and the number of removed observations
    / tracks."""
    logger.info(
        f"Filtering reconstruction by angular error "
        f"(threshold={angular_error_threshold_deg} deg)..."
    )
    angular_errors = compute_all_errors_from_reconstruction(
        reconstruction,
        ReprojectionErrorType.ANGULAR,
        negative_depth_observations,
        virtual_point_start=virtual_point_start,
        fisheye_cameras=fisheye_cameras,
    )

    all_angular = [
        e
        for errs in angular_errors.values()
        for _, _, e in errs
        if e < float("inf")
    ]
    if len(all_angular) > 0:
        all_angular_arr = np.array(all_angular)
        logger.info(
            f"  Angular errors: mean={np.mean(all_angular_arr):.2f} deg, "
            f"median={np.median(all_angular_arr):.2f} deg, "
            f"max={np.max(all_angular_arr):.2f} deg"
        )
        logger.info(
            f"  < {angular_error_threshold_deg} deg: "
            f"{100 * np.sum(all_angular_arr < angular_error_threshold_deg) / len(all_angular_arr):.1f}%"
        )

    num_points_before = len(reconstruction.points3D)
    obs_removed, tracks_removed = filter_observations_by_error(
        reconstruction,
        angular_errors,
        angular_error_threshold_deg,
        min_track_length=2,
    )
    logger.info(
        f"  Angular filter: removed {obs_removed} observations, "
        f"{tracks_removed} tracks"
    )
    logger.info(
        f"  Points3D: {num_points_before} -> {len(reconstruction.points3D)}"
    )


def run_refinement_pipeline(
    args,
    predictions_dict: dict,
    global_rotations,
    global_centers,
    global_intrinsics,
    dataset_pair,
    num_images: int,
    use_triangulation_first: bool = True,
    angular_error_threshold_deg: float = 0.5,
    num_refinement_iterations: int = 2,
    track_mode: str = "SPV",
) -> pycolmap.Reconstruction:
    """
    Run the refinement pipeline: triangulation, track establishment, and bundle adjustment.

    Args:
        args: Argument namespace (needs curr_path, images_path)
        predictions_dict: Predictions from star inference
        global_rotations: Global rotation matrices for all images
        global_centers: Global camera center positions
        global_intrinsics: Camera intrinsic parameters for all images
        dataset_pair: Dataset pair object (needs camera_model, intrinsics_mapping, images_shape_ori, images_list)
        num_images: Number of images in the dataset
        use_triangulation_first: If True, triangulate SIFT + real tracks first, then add only virtual points. If False (default), triangulate SIFT only and establish both real tracks and virtual points.
        track_mode: Combination of S(IFT), P(rior), V(irtual) tracks to use.
            Valid modes: "SPV", "SP", "SV", "PV", "S", "P".

    Returns:
        pycolmap.Reconstruction: The bundle-adjusted reconstruction
    """
    t_refinement_start = time.perf_counter()
    refinement_timing = {}

    # Parse track mode flags
    use_sift = "S" in track_mode
    use_prior = "P" in track_mode
    use_virtual = "V" in track_mode
    logger.info(
        f"Track mode: {track_mode} (SIFT={use_sift}, Prior={use_prior}, Virtual={use_virtual})"
    )

    # Step 1: Triangulate 3D points
    logger.info("Triangulating points with pycolmap...")
    t0 = time.perf_counter()
    suffix = getattr(args, "output_suffix", "")
    coarse_dir = f"coarse{suffix}"
    coarse_reconstruction = pycolmap.Reconstruction()
    coarse_reconstruction.read(args.curr_path + "/" + coarse_dir)
    refinement_timing["load_coarse"] = time.perf_counter() - t0

    # Step 1b: Determine parameters based on track mode
    if use_prior:
        database_name = "database_tracks.db"
        add_tracks = True
        log_message = "Creating tracks database (with prior tracks)..."
    else:
        database_name = "database_empty.db"
        add_tracks = False
        log_message = "Creating tracks database (empty, no prior tracks)..."

    # Step 1c: Create database with tracks (or empty)
    logger.info(log_message)
    t0 = time.perf_counter()
    prepare_glomap_prior(
        args.curr_path,
        dataset_pair.images_shape_ori,
        dataset_pair.images_list,
        global_intrinsics,
        predictions_dict,
        dataset_pair.intrinsics_mapping,
        camera_model=dataset_pair.camera_model,
        add_tracks=add_tracks,
        add_virtual_points=False,
        database_name=database_name,
    )
    refinement_timing["prepare_prior"] = time.perf_counter() - t0

    # Step 1c.5: Read SIFT DB keypoint counts (= sift_count per image)
    t0 = time.perf_counter()
    if use_sift:
        sift_db = pycolmap.Database.open(args.curr_path + "/database_sift.db")
        sift_count_by_name = {}
        for img in sift_db.read_all_images():
            kp = sift_db.read_keypoints(img.image_id)
            sift_count_by_name[img.name] = (
                len(kp) if kp is not None and len(kp) > 0 else 0
            )
    else:
        sift_count_by_name = {}
    refinement_timing["read_sift"] = time.perf_counter() - t0

    # Step 1d: Merge SIFT database with the created database (or copy if no SIFT)
    t0 = time.perf_counter()
    merged_db_path = args.curr_path + "/database_merged.db"
    if use_sift:
        logger.info("Merging SIFT and tracks databases...")
        merge_colmap_databases(
            db_path_primary=args.curr_path + "/" + database_name,
            db_path_secondary=args.curr_path + "/database_sift.db",
            output_path=merged_db_path,
            primary_features_first=False,  # SIFT features should be at the front for correct indexing
        )
    else:
        logger.info("Copying tracks database (no SIFT merge)...")
        shutil.copy2(args.curr_path + "/" + database_name, merged_db_path)
    refinement_timing["merge_databases"] = time.perf_counter() - t0

    # Step 2: Establish tracks from predictions
    t0 = time.perf_counter()
    track_options = TrackEstablishmentOptions(track_min_num_views_per_track=2)

    add_virtual_points_flag = use_virtual
    if use_triangulation_first and use_prior:
        # Real tracks already in DB for triangulation; only establish virtual points
        add_tracks_flag = False
    elif use_prior:
        # Establish real tracks into reconstruction directly
        add_tracks_flag = True
    else:
        # No prior tracks requested
        add_tracks_flag = False

    (
        points3D,
        keypoints_per_image,
        pts2d_idx_all,
        pts2d_idx_virtual_all,
        pts2d_idx_inv,
        pts2d_idx_virtual_inv,
        image_to_point3D,
        images_points2d_virtual_isnegative,
    ) = establish_tracks_from_tracks_dict(
        tracks_dict=predictions_dict,
        num_images=num_images,
        options=track_options,
        add_tracks=add_tracks_flag,
        add_virtual_points=add_virtual_points_flag,
        device="cuda",
    )
    torch.cuda.empty_cache()
    refinement_timing["establish_tracks"] = time.perf_counter() - t0

    # Step 3: Initialize 3D world points
    t0 = time.perf_counter()
    cameras = [
        (
            camera_from_intrinsics_matrix(intr[0], dataset_pair.camera_model)
            if intr is not None
            else None
        )
        for intr in global_intrinsics
    ]
    negative_depth_observations = build_negative_depth_observations(
        pts2d_idx_inv, images_points2d_virtual_isnegative
    )
    points3D = initialize_world_points(
        predictions_dict,
        global_rotations,
        global_centers,
        points3D,
        pts2d_idx_inv,
        pts2d_idx_virtual_inv,
        keypoints_per_image=keypoints_per_image,
        cameras=cameras,
        intrinsics_mapping=dataset_pair.intrinsics_mapping,
        angular_error_threshold_deg=angular_error_threshold_deg,
        negative_depth_observations=negative_depth_observations,
    )
    refinement_timing["initialize_points"] = time.perf_counter() - t0

    # Step 4: Configure bundle adjustment
    ba_options = IterativeBAOptions(
        max_ba_iterations=200,
        max_filter_iterations=3,
        normalized_reproj_threshold=1e-2,
        min_track_length=2,
        fix_rotations_first_pass=False,
    )

    # Step 5: Build reconstruction from current data
    t0 = time.perf_counter()
    virtual_reconstruction = build_reconstruction_for_ba(
        global_rotations,
        global_centers,
        global_intrinsics,
        dataset_pair.intrinsics_mapping,
        points3D,
        keypoints_per_image,
        image_sizes=dataset_pair.images_shape_ori,
        images_list=dataset_pair.images_list,
        camera_model=dataset_pair.camera_model,
    )
    refinement_timing["build_reconstruction"] = time.perf_counter() - t0
    reconstruction = deepcopy(virtual_reconstruction)

    # Step 6: Build negative depth and virtual point data
    # Original: build from pts2d_idx_inv
    virtual_point_start = build_virtual_point_start(pts2d_idx_inv)

    negative_depth_observations = build_negative_depth_observations(
        pts2d_idx_inv, images_points2d_virtual_isnegative
    )

    # build_reconstruction_for_ba produces a 1-indexed reconstruction; shift
    # the per-image dicts by +1 so their keys line up with reconstruction image IDs.
    # TODO: fix this
    virtual_point_start = {
        # img_id + 1: v for img_id, v in virtual_point_start.items()
        img_id + 1: 100000
        for img_id, v in virtual_point_start.items()
    }
    negative_depth_observations = {
        img_id + 1: s for img_id, s in negative_depth_observations.items()
    }

    database_path = args.curr_path + "/database_merged.db"
    triangulated_output_path = args.curr_path + "/coarse_triangulated"

    iteration_timings = []
    for outer_iter in range(num_refinement_iterations):
        logger.info(f"{'=' * 60}")
        logger.info(
            f"Refinement iteration {outer_iter + 1}/{num_refinement_iterations}"
        )
        logger.info(f"{'=' * 60}")
        t_iter_start = time.perf_counter()

        # Step 1e: Triangulate on merged database
        t_tri_start = time.perf_counter()
        opt_triang = pycolmap.IncrementalPipelineOptions()
        opt_triang.triangulation.min_angle = 1.0
        opt_triang.triangulation.merge_max_reproj_error = 15.0
        opt_triang.triangulation.complete_max_reproj_error = 15.0
        opt_triang.triangulation.ignore_two_view_tracks = False
        opt_triang.triangulation.create_max_angle_error = (
            angular_error_threshold_deg
        )
        opt_triang.ba_global_max_refinements = 0

        reconstruction = pycolmap.triangulate_points(
            reconstruction,
            database_path,
            ".",  # skip color extraction
            triangulated_output_path,
            clear_points=True,
            refine_intrinsics=False,
            options=opt_triang,
        )
        t_tri_end = time.perf_counter()

        # Step 7a: Selectively prune prior/virtual tracks (SelectTrack logic)
        sift_count = {}
        for recon_id, img in reconstruction.images.items():
            sift_count[recon_id] = sift_count_by_name.get(img.name, 0)

        select_tracks_from_merged(
            reconstruction=reconstruction,
            sift_count=sift_count,
            min_num_support_abs=512,
        )

        # Step 7b: Create fisheye cameras and rectify virtual 2D points
        # fisheye_intrinsics_params, fisheye_cameras = create_fisheye_cameras_and_rectify(
        #     reconstruction, virtual_point_start
        # )
        fisheye_intrinsics_params = None
        fisheye_cameras = None

        # Step 7.5: Filter tracks before bundle adjustment
        t_filter_start = time.perf_counter()
        if angular_error_threshold_deg > 0:
            # Real reconstruction: try pycolmap's built-in observation filter,
            # fall back to filter_reconstruction_by_angular_error on failure
            try:
                num_points_before = len(reconstruction.points3D)
                obs_manager = pycolmap.ObservationManager(reconstruction)
                points3d_ids = set(reconstruction.points3D.keys())
                num_filtered = (
                    obs_manager.filter_points3D_with_large_reprojection_error(
                        angular_error_threshold_deg,
                        points3d_ids,
                        pycolmap.ReprojectionErrorType.ANGULAR,  # error_type
                    )
                )
                obs_manager.filter_points3D_with_short_tracks(
                    min_track_length=2
                )
                logger.info(
                    f"Real reconstruction: pycolmap filter removed "
                    f"{num_filtered} observations, points3D "
                    f"{num_points_before} -> {len(reconstruction.points3D)}"
                )
            except Exception as e:
                logger.warning(
                    f"pycolmap observation filter failed ({e}); "
                    f"falling back to filter_reconstruction_by_angular_error"
                )
                filter_reconstruction_by_angular_error(
                    reconstruction,
                    angular_error_threshold_deg,
                    dict(),  # no negative depth filtering for real tracks
                    dict(),
                )

            # Virtual reconstruction: keep the angular-error filter path
            filter_reconstruction_by_angular_error(
                virtual_reconstruction,
                angular_error_threshold_deg,
                negative_depth_observations,
                virtual_point_start,
                fisheye_cameras=fisheye_cameras,
            )

        t_filter_end = time.perf_counter()

        # Step 7c: Limit number of tracks before BA
        max_num_tracks = getattr(args, "max_num_tracks", None)
        if (
            max_num_tracks is not None
            and len(reconstruction.points3D) > max_num_tracks
        ):
            sorted_ids = sorted(
                reconstruction.points3D.keys(),
                key=lambda pid: len(
                    list(reconstruction.points3D[pid].track.elements)
                ),
                reverse=True,
            )
            ids_to_remove = sorted_ids[max_num_tracks:]
            for pid in ids_to_remove:
                reconstruction.delete_point3D(pid)
            logger.info(
                f"  Track limit: kept {max_num_tracks}, "
                f"removed {len(ids_to_remove)} tracks"
            )

        # Step 8: Run iterative bundle adjustment
        t_ba_start = time.perf_counter()
        reconstruction, virtual_reconstruction = iterative_bundle_adjustment(
            reconstruction,
            virtual_reconstruction,
            negative_depth_observations,
            virtual_point_start,
            options=ba_options,
            fisheye_intrinsics_params=fisheye_intrinsics_params,
        )
        t_ba_end = time.perf_counter()

        iter_timing = {
            "triangulation": t_tri_end - t_tri_start,
            "filter": t_filter_end - t_filter_start,
            "ba": t_ba_end - t_ba_start,
            "total": t_ba_end - t_iter_start,
        }
        iteration_timings.append(iter_timing)
        logger.info(
            f"[Profiling] Iteration {outer_iter + 1}: triangulation={iter_timing['triangulation']:.2f}s, "
            f"filter={iter_timing['filter']:.2f}s, "
            f"ba={iter_timing['ba']:.2f}s, total={iter_timing['total']:.2f}s"
        )

    # Clean up triangulated reconstruction output
    if os.path.exists(triangulated_output_path):
        shutil.rmtree(triangulated_output_path)

    # Step 9: Filter out 3D points that have any virtual observations
    t0 = time.perf_counter()
    if use_virtual:
        # A 3D point is removed if ANY of its track observations are virtual
        virtual_point3D_ids = []
        for point3D_id, point3D in reconstruction.points3D.items():
            is_any_virtual = False
            for elem in point3D.track.elements:
                image_id = elem.image_id
                point2D_idx = elem.point2D_idx
                vp_start = virtual_point_start.get(image_id, float("inf"))
                if point2D_idx >= vp_start:
                    is_any_virtual = True
                    break
            if is_any_virtual:
                virtual_point3D_ids.append(point3D_id)

        for point3D_id in virtual_point3D_ids:
            reconstruction.delete_point3D(point3D_id)

        logger.info(
            f"Removed {len(virtual_point3D_ids)} virtual 3D points, {len(reconstruction.points3D)} real points remaining"
        )
    else:
        logger.info("No virtual points to remove (track mode has no V).")
    refinement_timing["remove_virtual"] = time.perf_counter() - t0

    # Step 10: Write bundle adjusted results to COLMAP format
    t0 = time.perf_counter()
    suffix = getattr(args, "output_suffix", "")
    file_dir = f"gluemap_aba{suffix}"
    logger.info(
        "Writing bundle adjusted reconstruction: %s",
        args.curr_path + "/" + file_dir,
    )
    os.makedirs(args.curr_path + "/" + file_dir, exist_ok=True)
    reconstruction.write(args.curr_path + "/" + file_dir)
    refinement_timing["write_output"] = time.perf_counter() - t0

    refinement_timing["iterations"] = iteration_timings
    refinement_timing["total"] = time.perf_counter() - t_refinement_start

    logger.info("[Profiling] Refinement Summary:")
    logger.info(
        f"  Setup: load_coarse={refinement_timing['load_coarse']:.2f}s, "
        f"prepare_prior={refinement_timing['prepare_prior']:.2f}s, "
        f"merge_db={refinement_timing['merge_databases']:.2f}s, "
        f"establish_tracks={refinement_timing['establish_tracks']:.2f}s, "
        f"init_points={refinement_timing['initialize_points']:.2f}s, "
        f"build_recon={refinement_timing['build_reconstruction']:.2f}s"
    )
    logger.info(
        f"  Iterations: {sum(it['total'] for it in iteration_timings):.2f}s "
        f"({len(iteration_timings)} iters)"
    )
    logger.info(
        f"  Cleanup: remove_virtual={refinement_timing['remove_virtual']:.2f}s, "
        f"write={refinement_timing['write_output']:.2f}s"
    )
    logger.info(f"  Total refinement: {refinement_timing['total']:.2f}s")

    return file_dir, refinement_timing
