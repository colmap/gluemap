import logging

import numpy as np
import pyceres
import pycolmap
import pygluemap

logger = logging.getLogger(__name__)


def add_virtual_track_residuals(
    problem,
    virtual_reconstruction: pycolmap.Reconstruction | None,
    reference_reconstruction: pycolmap.Reconstruction,
    negative_depth_observations,
    fisheye_intrinsics_params=None,
):
    """
    Add reprojection residuals for virtual tracks to an existing ceres problem.

    Pose and intrinsic parameter blocks are resolved through
    ``reference_reconstruction`` (the real reconstruction handed to
    ``pycolmap.create_default_ceres_bundle_adjuster``) so that their numpy
    buffers are the same ones pycolmap is already optimizing -- the virtual
    residuals thus contribute to the same parameter blocks rather than
    detached copies.

    Virtual points3D are read from ``virtual_reconstruction``; their xyz
    arrays become new parameter blocks in ``problem`` via the residual block.

    Arctan loss is always used for virtual points. If
    ``fisheye_intrinsics_params`` is supplied, virtual residuals are built
    against a SIMPLE_FISHEYE camera model using the (fixed) fisheye params,
    mirroring the previous behaviour at the disabled
    ``create_fisheye_cameras_and_rectify`` call site.
    """
    if virtual_reconstruction is None or len(virtual_reconstruction.points3D) == 0:
        return

    fisheye_model_id = pycolmap.CameraModelId.SIMPLE_FISHEYE
    loss_arctan = pyceres.LossFunction(
        {"name": "arctan", "params": [5.0], "magnitude": 1.0}
    )

    num_constraints = 0
    num_negative = 0
    num_skipped = 0
    num_none = 0
    fixed_fisheye_blocks = set()

    for point3D in virtual_reconstruction.points3D.values():
        world_point = point3D.xyz
        if world_point is None or np.all(world_point == 0):
            num_none += 1
            continue

        for elem in point3D.track.elements:
            image_id, pt_idx = elem.image_id, elem.point2D_idx

            if image_id not in reference_reconstruction.images:
                num_skipped += 1
                continue
            if image_id not in virtual_reconstruction.images:
                num_skipped += 1
                continue

            image = virtual_reconstruction.images[image_id]
            if pt_idx >= len(image.points2D):
                num_skipped += 1
                continue
            point2D = image.points2D[pt_idx].xy

            camera_id = reference_reconstruction.images[image_id].camera_id

            # Pose & intrinsics come from the reference reconstruction so the
            # underlying numpy buffers are shared with pycolmap's residuals.
            cam_pose = reference_reconstruction.frames[image_id].rig_from_world.params
            if (
                fisheye_intrinsics_params is not None
                and camera_id < len(fisheye_intrinsics_params)
                and fisheye_intrinsics_params[camera_id] is not None
            ):
                camera_params = fisheye_intrinsics_params[camera_id]
                active_model_id = fisheye_model_id
            else:
                camera_params = reference_reconstruction.cameras[camera_id].params
                active_model_id = reference_reconstruction.cameras[camera_id].model

            is_negative = (
                image_id in negative_depth_observations
                and pt_idx in negative_depth_observations[image_id]
            )
            if is_negative:
                cost = pygluemap.ReprojErrorCostWithNegativeDepth(
                    active_model_id, point2D
                )
                num_negative += 1
            else:
                cost = pygluemap.ReprojErrorCost(active_model_id, point2D)

            problem.add_residual_block(
                cost,
                loss_arctan,
                [world_point, cam_pose, camera_params],
            )
            num_constraints += 1

            # Fisheye intrinsics are fixed (not optimized).
            if (
                fisheye_intrinsics_params is not None
                and active_model_id == fisheye_model_id
                and id(camera_params) not in fixed_fisheye_blocks
                and problem.has_parameter_block(camera_params)
            ):
                problem.set_parameter_block_constant(camera_params)
                fixed_fisheye_blocks.add(id(camera_params))

    logger.info(
        f"Added {num_constraints} virtual reprojection constraints "
        f"({num_negative} with negative depth, "
        f"{num_skipped} skipped, {num_none} with no xyz)"
    )


def bundle_adjustment(
    reconstruction: pycolmap.Reconstruction,
    virtual_reconstruction: pycolmap.Reconstruction | None,
    negative_depth_observations,
    max_num_iterations: int = 200,
    fisheye_intrinsics_params=None,
) -> tuple[pycolmap.Reconstruction, pycolmap.Reconstruction | None]:
    """
    Bundle adjustment over real + virtual reconstructions.

    The real reconstruction is optimized via pycolmap's built-in ceres
    bundle adjuster (handles manifolds, gauge fixing, solver selection,
    and default Huber loss). Virtual residuals are appended manually to
    the same ceres problem via ``add_virtual_track_residuals`` so that
    they share the pose/intrinsic parameter blocks with the real residuals
    but use Arctan loss (and optionally a fixed SIMPLE_FISHEYE camera
    model with the provided ``fisheye_intrinsics_params``).

    Args:
        reconstruction: pycolmap.Reconstruction holding the real tracks
            plus authoritative poses and intrinsics. Optimized in-place.
        virtual_reconstruction: pycolmap.Reconstruction whose points3D
            are virtual; may be None or empty for a pure real BA. Its
            points3D.xyz values are optimized in-place as part of the
            joint solve.
        negative_depth_observations: Dict[image_id, Set[point2D_idx]]
            marking observations that should use the negative-depth cost.
        max_num_iterations: Max Ceres iterations.
        fisheye_intrinsics_params: Optional List[np.ndarray] indexed by
            camera_id giving fixed fisheye intrinsics; when present,
            virtual residuals use SIMPLE_FISHEYE against these params.

    Returns:
        (reconstruction, virtual_reconstruction) with parameters
        updated in-place.
    """
    logger.info(
        f"Bundle adjustment: {len(reconstruction.points3D)} real tracks, "
        f"{len(virtual_reconstruction.points3D) if virtual_reconstruction is not None else 0} virtual tracks"
    )

    # --- Build pycolmap BA over the real reconstruction --------------------
    ba_options = pycolmap.BundleAdjustmentOptions()
    # Restore stock Ceres convergence tolerances.
    ba_options.ceres.solver_options = pyceres.SolverOptions()
    ba_options.ceres.solver_options.max_num_iterations = max_num_iterations
    
    # Disable Schur auto-selection: the linear_solver_ordering that pycolmap
    # builds in ``create_solver_options`` does not match the parameter blocks
    # in the pyceres ``Problem`` (observed vertices.size() != ordering->size()),
    # which trips a Ceres DCHECK and aborts the solve. This is a pyceres /
    # pycolmap handoff issue, independent of virtual tracks. Fall back to
    # SPARSE_NORMAL_CHOLESKY which does not require a custom ordering
    # (matches the previous hand-rolled solver config).
    # TODO: fix the underlying issue and re-enable auto-selection (which picks a more efficient sparse schur solver for typical BA problems).
    ba_options.ceres.auto_select_solver_type = False
    ba_options.ceres.solver_options.linear_solver_type = (
        pyceres.LinearSolverType.SPARSE_NORMAL_CHOLESKY
    )
    # Wrap real-track reprojection residuals in a Huber loss (matches the
    # hand-rolled solver's prior behaviour). loss_function_scale stays at its
    # default 1.0 which reproduces the previous Huber[1.0] magnitude=1.0.
    ba_options.ceres.loss_function_type = pycolmap.LossFunctionType.HUBER

    ba_config = pycolmap.BundleAdjustmentConfig()
    for image_id in reconstruction.images:
        ba_config.add_image(image_id)
    for point3D_id in reconstruction.points3D:
        ba_config.add_variable_point(point3D_id)
    ba_config.fix_gauge(pycolmap.BundleAdjustmentGauge.TWO_CAMS_FROM_WORLD)

    bundle_adjuster = pycolmap.create_default_ceres_bundle_adjuster(
        ba_options, ba_config, reconstruction
    )
    problem = bundle_adjuster.problem

    logger.info(
        f"After pycolmap BA construction: "
        f"{problem.num_residual_blocks()} residual blocks, "
        f"{problem.num_parameter_blocks()} parameter blocks, "
        f"{problem.num_residuals()} residuals"
    )

    # --- Append virtual residuals to the same problem ----------------------
    add_virtual_track_residuals(
        problem,
        virtual_reconstruction=virtual_reconstruction,
        reference_reconstruction=reconstruction,
        negative_depth_observations=negative_depth_observations,
        fisheye_intrinsics_params=fisheye_intrinsics_params,
    )

    logger.info(
        f"After virtual residual add: "
        f"{problem.num_residual_blocks()} residual blocks, "
        f"{problem.num_parameter_blocks()} parameter blocks, "
        f"{problem.num_residuals()} residuals"
    )

    # --- Solve -------------------------------------------------------------
    solver_options = ba_options.ceres.create_solver_options(
        ba_config, problem
    )
    summary = pyceres.SolverSummary()
    pyceres.solve(solver_options, problem, summary)
    logger.info(summary.BriefReport())

    # --- Sync poses/intrinsics into the virtual reconstruction -------------
    # Only the real reconstruction's numpy buffers flowed into the ceres
    # problem (see ``add_virtual_track_residuals``); the virtual
    # reconstruction still holds the pre-solve values. Copy optimized
    # poses and per-camera intrinsics over so downstream consumers
    # reading from virtual_reconstruction observe consistent state.
    if virtual_reconstruction is not None:
        # Lazy import to avoid a circular estimators -> controllers import
        # at module load time.
        from gluemap.controllers.augmented_bundle_adjustment import (
            update_poses_from_reconstruction,
        )

        update_poses_from_reconstruction(reconstruction, virtual_reconstruction)

    return reconstruction, virtual_reconstruction
