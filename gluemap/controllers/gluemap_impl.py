import logging
import os
import time

import torch

logger = logging.getLogger(__name__)
from gluemap.controllers.augmented_bundle_adjustment import (
    run_refinement_pipeline,
)
from gluemap.controllers.global_merger import GlobalGluer
from gluemap.controllers.star_collection import run_star_collection
from gluemap.controllers.star_inference import run_star_inference
from gluemap.controllers.twoview_inference import run_twoview_inference
from gluemap.estimators.rotation_averaging import (
    collect_relative_rotations_ministar,
)
from gluemap.estimators.track_snapping import TrackSnapping
from gluemap.estimators.virtual_tracks import VirtualTrackPreparation
from gluemap.math.scaling import (
    keep_inframes,
    rescale_intrinsics,
    rescale_tracks,
)
from gluemap.utils.colmap import (
    prepare_sift_database,
    write_to_colmap_format,
)


class GlueMapPipeline:
    """
    Main GLUEMAP pipeline: twoview -> star -> global mapping -> refinement.

    Stable configuration is stored as instance attributes. Per-dataset inputs
    are passed to ``run()`` or ``run_postprocessing()``.
    """

    def __init__(self, args, world_size, rank, device, dtype, models=None):
        self.args = args
        self.world_size = world_size
        self.rank = rank
        self.device = device
        self.dtype = dtype
        self.models = models

    @torch.no_grad()
    def run(self, dataset_pair, pairs=None, device_id="0"):
        """
        Run the full inference pipeline: twoview -> star -> global mapping -> refinement.

        Only executes global mapping and refinement on rank 0.

        Returns:
            Tuple of (pred_dir, timing_dict) or (None, timing_dict)
        """
        args = self.args
        world_size = self.world_size
        rank = self.rank
        device = self.device
        dtype = self.dtype
        models = self.models

        timing = {}
        t_pipeline_start = time.perf_counter()

        # Step 1: Two-view inference
        global_outputs, twoview_timing = run_twoview_inference(
            args,
            dataset_pair,
            world_size,
            rank,
            file_name="twoview_result.pth",
            save_intermediate_results=True,
            device=device,
            preloaded_models=models,
        )
        timing["twoview_inference"] = twoview_timing

        # Step 2: Generate dataset from outputs
        t0 = time.perf_counter()
        dataset = run_star_collection(dataset_pair, global_outputs, args)
        timing["dataset_generation"] = time.perf_counter() - t0

        # Step 3: Star inference
        predictions_dict, star_timing = run_star_inference(
            args,
            dataset,
            world_size,
            rank,
            file_name="star_result.pth",
            save_intermediate_results=True,
            device=device,
            preloaded_models=models,
        )
        timing["star_inference"] = star_timing

        # Release extractor models from GPU — not needed after star inference
        dataset.query_extractors = []
        dataset._extractors_device = None

        # Move predictions_dict tensors to CPU to free GPU memory before postprocessing
        for key, value in predictions_dict.items():
            if isinstance(value, torch.Tensor):
                predictions_dict[key] = value.cpu()
            elif isinstance(value, list):
                predictions_dict[key] = [
                    v.cpu() if isinstance(v, torch.Tensor) else v for v in value
                ]
        torch.cuda.empty_cache()

        # Steps 4-5: Global mapping and refinement (rank 0 only)
        if rank == 0:
            pred_dir, postproc_timing = self.run_postprocessing(
                args,
                predictions_dict,
                dataset_pair,
                dataset,
                pairs=pairs,
                device_id=device_id,
            )
            timing["postprocessing"] = postproc_timing
            timing["total_pipeline"] = time.perf_counter() - t_pipeline_start

            # Print summary
            logger.info("[Profiling] Pipeline Summary:")
            logger.info(
                f"  Two-view (load+infer): model_load={twoview_timing.get('model_loading', 0):.2f}s, "
                f"inference={twoview_timing['total']:.2f}s"
            )
            logger.info(
                f"  Dataset generation:    {timing['dataset_generation']:.2f}s"
            )
            logger.info(
                f"  Star (load+infer):     model_load={star_timing.get('model_loading', 0):.2f}s, "
                f"inference={star_timing['total']:.2f}s"
            )
            logger.info(
                f"  Postprocessing:        {postproc_timing['total']:.2f}s"
            )
            logger.info(
                f"  Total pipeline:        {timing['total_pipeline']:.2f}s"
            )

            # Save per-dataset timing
            timing_path = os.path.join(args.curr_path, "pipeline_timing.pth")
            torch.save(timing, timing_path)
            logger.info(
                f"[Profiling] Per-dataset timing saved to: {timing_path}"
            )

            return pred_dir, timing

        timing["total_pipeline"] = time.perf_counter() - t_pipeline_start
        return None, timing

    @staticmethod
    def run_postprocessing(
        args, predictions_dict, dataset_pair, dataset, pairs=None, device_id="0"
    ):
        """
        Run postprocessing: global mapping and refinement.

        This should only be called on rank 0.

        Returns:
            Tuple of (pred_dir, timing_dict)
        """
        timing = {}
        t_postproc_start = time.perf_counter()
        torch.cuda.empty_cache()

        matching_pairs = pairs if pairs is not None else dataset.pairs

        t0 = time.perf_counter()
        poses_rel, poses_rel_scores = collect_relative_rotations_ministar(
            predictions_dict
        )
        timing["collect_rotations"] = time.perf_counter() - t0

        # Step 4: Global mapping
        t0 = time.perf_counter()
        GlueMapPipeline.restore_image_shape(
            predictions_dict, dataset.images_change, dataset.images_shape_ori
        )

        predictions_dict["image_index_to_star_index"] = (
            dataset.image_index_to_star_index
        )

        global_gluer = GlobalGluer(args)
        global_gluer.sequential_edges = set(
            getattr(dataset_pair, "sequential_edges", [])
        )
        (
            global_rotations,
            global_centers,
            global_intrinsics,
            valid_edges,
            predictions_dict,
        ) = global_gluer.main(
            predictions_dict,
            dataset_pair.intrinsics_mapping,
            dataset_pair.camera_model,
            len(dataset),
        )
        timing["global_mapping"] = time.perf_counter() - t0

        # Override with GT intrinsics if requested (after global mapping)
        if getattr(args, "gt_intrinsics_path", None):
            from gluemap.utils.colmap import extract_gt_intrinsics

            gt_intrinsics = extract_gt_intrinsics(
                args.gt_intrinsics_path,
                dataset_pair.images_list,
                dataset_pair.intrinsics_mapping,
            )
            for cam_id in range(len(global_intrinsics)):
                if (
                    cam_id < len(gt_intrinsics)
                    and gt_intrinsics[cam_id] is not None
                ):
                    global_intrinsics[cam_id] = gt_intrinsics[cam_id]
            logger.info(
                f"Replaced intrinsics with GT from {args.gt_intrinsics_path}"
            )

        virtual_track_preparation = VirtualTrackPreparation()
        virtual_track_preparation.main(
            predictions_dict,
            global_intrinsics,
            dataset_pair.intrinsics_mapping,
            global_rotations,
            global_centers,
        )

        # Write coarse results to COLMAP format
        t0 = time.perf_counter()
        suffix = getattr(args, "output_suffix", "")
        coarse_dir = f"coarse{suffix}"
        logger.info(
            "write_to_colmap_format: %s", args.curr_path + "/" + coarse_dir
        )
        write_to_colmap_format(
            args.curr_path + "/" + coarse_dir,
            dataset_pair.images_shape_ori,
            predictions_dict,
            global_rotations,
            global_centers,
            {},
            global_intrinsics,
            dataset_pair.intrinsics_mapping,
            images_list=dataset_pair.images_list,
            pose_only=True,
            camera_type=dataset_pair.camera_model,
        )
        timing["write_coarse"] = time.perf_counter() - t0

        # Early exit if coarse_only
        if getattr(args, "coarse_only", False):
            logger.info("Coarse only mode: skipping all refinement steps.")
            timing["total"] = time.perf_counter() - t_postproc_start
            return coarse_dir, timing

        t0 = time.perf_counter()
        if not (
            hasattr(args, "force_load") and args.force_load
        ) or not os.path.exists(args.curr_path + "/database_sift.db"):
            prepare_sift_database(
                args.curr_path,
                args.images_path,
                dataset_pair.images_list,
                dataset_pair.intrinsics_mapping,
                matching_pairs,
                None,
                extraction_method="sift",
                camera_model=dataset_pair.camera_model,
                skip_matching=False,
                remove_existing=True,
            )
        timing["sift_database"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        track_snapping = TrackSnapping(snapping_thres=1.0)
        track_snapping.main(
            args.curr_path + "/database_sift.db",
            predictions_dict,
            dataset_pair.images_shape_ori,
            dataset_pair.images_list,
        )
        timing["track_snapping"] = time.perf_counter() - t0

        # Step 5: Refinement
        t0 = time.perf_counter()
        pred_dir, refinement_timing = run_refinement_pipeline(
            args=args,
            predictions_dict=predictions_dict,
            global_rotations=global_rotations,
            global_centers=global_centers,
            global_intrinsics=global_intrinsics,
            dataset_pair=dataset_pair,
            num_images=len(dataset),
            use_triangulation_first=True,
            num_refinement_iterations=getattr(
                args, "num_refinement_iterations", 2
            ),
            track_mode=getattr(args, "track_mode", "SPV"),
        )
        timing["refinement"] = time.perf_counter() - t0
        timing["refinement_detail"] = refinement_timing

        timing["total"] = time.perf_counter() - t_postproc_start

        logger.info("[Profiling] Postprocessing Summary:")
        logger.info(f"  collect_rotations: {timing['collect_rotations']:.2f}s")
        logger.info(f"  global_mapping:    {timing['global_mapping']:.2f}s")
        logger.info(f"  write_coarse:      {timing['write_coarse']:.2f}s")
        logger.info(
            f"  sift_database:     {timing.get('sift_database', 0):.2f}s"
        )
        logger.info(
            f"  track_snapping:    {timing.get('track_snapping', 0):.2f}s"
        )
        logger.info(f"  refinement:        {timing.get('refinement', 0):.2f}s")
        logger.info(f"  total:             {timing['total']:.2f}s")

        return pred_dir, timing

    @staticmethod
    def restore_image_shape(
        predictions_dict, images_change, images_shape_ori, valid_threshold=0.05
    ):
        index = range(len(predictions_dict["indexes"]))
        for idx in index:
            rescale_tracks(predictions_dict, images_change, [idx])
            keep_inframes(predictions_dict, images_shape_ori, [idx])

            if "intrinsics" in predictions_dict:
                members = predictions_dict["indexes"][idx]
                scales_curr = [
                    images_change[members[j]] for j in range(len(members))
                ]
                intrinscs_curr = [
                    predictions_dict["intrinsics"][idx][:, j]
                    for j in range(len(members))
                ]
                intrinscs_curr = torch.stack(
                    rescale_intrinsics(intrinscs_curr, scales_curr), dim=1
                )
                predictions_dict["intrinsics"][idx] = intrinscs_curr.cpu()

        return predictions_dict


# Backward-compatible module-level wrapper functions


def run_inference_pipeline(
    args,
    dataset_pair,
    world_size,
    rank,
    device,
    dtype,
    pairs=None,
    device_id="0",
    models=None,
):
    """Backward-compatible wrapper for GlueMapPipeline.run()."""
    pipeline = GlueMapPipeline(
        args, world_size, rank, device, dtype, models=models
    )
    return pipeline.run(dataset_pair, pairs=pairs, device_id=device_id)


def run_postprocessing_pipeline(
    args, predictions_dict, dataset_pair, dataset, pairs=None, device_id="0"
):
    """Backward-compatible wrapper for GlueMapPipeline.run_postprocessing()."""
    return GlueMapPipeline.run_postprocessing(
        args,
        predictions_dict,
        dataset_pair,
        dataset,
        pairs=pairs,
        device_id=device_id,
    )
