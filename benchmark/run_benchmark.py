"""Main benchmark script for running GLUEMAP on multiple datasets."""

import argparse
import os
import re
from typing import Dict, Any, Tuple, Optional, List

import torch

from gluemap.datasets.twoview_dataset import BaseTwoViewDataset
from gluemap.datasets.sequential_twoview_dataset import SequentialTwoViewDataset
from gluemap.datasets.multi_sequence_twoview_dataset import MultiSequencePairs
from gluemap.utils.gpu import init_distributed
from gluemap.utils.cli import get_args_parser
from gluemap.controllers.salad_retrieval import (
    run_preprocessing_pipeline,
    run_preprocessing_pipeline_multi,
)
from gluemap.controllers.gluemap_impl import run_inference_pipeline
from gluemap.eval.ablations_filtering import run_ablation_inference_pipeline
from gluemap.eval.ablations_backbone import run_backbone_ablation_pipeline
from gluemap.eval.ablations_tracks import run_track_ablation_pipeline
from gluemap.eval.ablations_pure_feedforward import run_direct_inference_pipeline
from gluemap.utils.model_loader import load_all_models

from benchmark.utils import (
    load_config,
    load_datasets_from_config,
    get_output_dir_from_pattern,
    print_summary,
    save_results_csv,
)
from benchmark.evaluate import (
    load_colmap_reconstruction,
    compare_reconstructions,
)

# Config keys that are benchmark-structural (dataset discovery, evaluation settings,
# per-dataset overrides) — excluded when forwarding config values to the args namespace.
_BENCHMARK_STRUCTURAL_KEYS = frozenset({
    "benchmark", "evaluation", "dataset_groups", "experiments",
    "images_pattern", "gt_pattern", "output_pattern", "_base_",
    "images_path", "write_path",  # passed explicitly as function args
    "pipeline",  # old nested style — ignored; use flat keys instead
})

# Defaults for benchmark-only flags not present in get_args_parser().
_BENCHMARK_EXTRA_DEFAULTS: Dict[str, Any] = {
    "is_sequential": False,
    "sample_frequency": 1,
    "multi_sequence": False,
    "subfolder_regex": None,
    "skip_doppelgangers": False,
    "skip_back_and_forth": False,
    "backbone_ablation": False,
    "track_ablation": False,
    "track_modes": ["SPV"],
    "direct_inference": False,
}


def create_args_from_config(
    config: Dict[str, Any],
    images_path: str,
    write_path: str,
    images_list: Optional[List[str]] = None,
) -> "argparse.Namespace":
    """Create an argparse.Namespace from a benchmark config dictionary.

    Starts from cli.py parser defaults, applies benchmark-only extra defaults,
    then overlays all flat config values (excluding benchmark-structural keys).
    Dataset-specific paths are set last.

    Configs should use the same flat key names as configs/example.yaml
    (e.g. path_feedforward, path_retrieval, path_tracker, path_dg, chosen_model, …).

    Args:
        config: Configuration dictionary loaded from YAML.
        images_path: Path to images for this dataset.
        write_path: Path to write results for this dataset.
        images_list: Optional list of image filenames (None = all images).

    Returns:
        argparse.Namespace ready to pass into the pipeline.
    """
    # 1. Start from cli.py defaults — single source of truth for pipeline knobs.
    args = get_args_parser().parse_args([])

    # 2. Apply benchmark-only defaults for flags not in the cli parser.
    for key, value in _BENCHMARK_EXTRA_DEFAULTS.items():
        setattr(args, key, value)

    # 3. Overlay flat config values, skipping benchmark-structural keys.
    for key, value in config.items():
        if key not in _BENCHMARK_STRUCTURAL_KEYS:
            setattr(args, key, value)

    # 4. Dataset-specific overrides (always take precedence over config).
    args.images_path = images_path
    args.write_path = write_path
    args.images_list = images_list

    return args


def _select_pipeline(args):
    """Select the appropriate pipeline function based on ablation flags."""
    if getattr(args, 'direct_inference', False):
        return run_direct_inference_pipeline
    if getattr(args, 'track_ablation', False):
        return run_track_ablation_pipeline
    if getattr(args, 'backbone_ablation', False):
        return run_backbone_ablation_pipeline
    if getattr(args, 'skip_doppelgangers', False) or getattr(args, 'skip_back_and_forth', False):
        return run_ablation_inference_pipeline
    return run_inference_pipeline


def run_single_dataset(
    config: Dict[str, Any],
    dataset_name: str,
    images_path: str,
    gt_path: Optional[str],
    output_path: str,
    world_size: int,
    rank: int,
    device,
    dtype,
    images_list: Optional[List[str]] = None,
    preprocess_path: Optional[str] = None,
    show_viz: bool = False,
    models=None,
) -> Dict[str, Any]:
    """Run the GLUEMAP pipeline on a single dataset and evaluate.

    Args:
        config: Configuration dictionary
        dataset_name: Name of the dataset
        images_path: Path to the images
        gt_path: Path to the ground truth COLMAP reconstruction
        output_path: Path to write results
        world_size: Number of distributed processes
        rank: Current process rank
        device: Torch device
        dtype: Torch dtype
        images_list: Optional list of image filenames to use (None means all images)
        preprocess_path: Optional path for scene-level preprocessing (SALAD features).
            If provided, experiments in the same scene share preprocessed features.

    Returns:
        Dictionary containing evaluation metrics
    """
    print(f"\n{'='*60}")
    print(f"Processing dataset: {dataset_name}")
    print(f"Images: {images_path}")
    print(f"Ground truth: {gt_path}")
    print(f"Output: {output_path}")
    if preprocess_path:
        print(f"Preprocess path: {preprocess_path}")
    if images_list is not None:
        print(f"Using {len(images_list)} specified images")
    print(f"{'='*60}\n")

    # CO3D mode: images are inside an h5py file, extract to /tmp first
    h5py_path_orig = None
    if images_path.endswith('.h5'):
        from gluemap.datasets.co3d_utils import extract_images_from_h5py
        h5py_path_orig = images_path
        images_path = extract_images_from_h5py(images_path, images_list)

    # Create args for this dataset
    args = create_args_from_config(config, images_path, output_path, images_list)
    # Resolve use_gt_intrinsics: set gt_intrinsics_path from gt_path
    if getattr(args, "use_gt_intrinsics", False) and gt_path:
        args.gt_intrinsics_path = gt_path
    # Use preprocess_path for SALAD features if provided, otherwise use write_path
    args.curr_processed = preprocess_path if preprocess_path else args.write_path
    args.curr_path = args.write_path
    args.distributed = world_size > 1
    args.rank = rank
    args.world_size = world_size
    args.gpu = int(os.environ.get("LOCAL_RANK", 0))

    dataset_timing = {}

    # Multi-sequence mode: discover subfolders matching regex, use MultiSequencePairs
    if getattr(args, 'multi_sequence', False) and args.subfolder_regex:
        subfolder_pattern = re.compile(args.subfolder_regex)
        datasets = [x for x in sorted(os.listdir(images_path)) if subfolder_pattern.match(x)]
        print(f"Multi-sequence mode: found {len(datasets)} subfolders matching '{args.subfolder_regex}'")

        (_, _), preproc_timing = run_preprocessing_pipeline_multi(args, world_size, rank, datasets, models=models)
        dataset_timing["preprocessing"] = preproc_timing
        dataset_pair = MultiSequencePairs(args, datasets)
        _run_fn = _select_pipeline(args)
        pred_dir, inference_timing = _run_fn(
            args, dataset_pair, world_size, rank, device, dtype,
            pairs=dataset_pair.pairs, models=models,
        )
        dataset_timing.update(inference_timing)
    else:
        # Standard single-dataset mode
        if getattr(args, 'direct_inference', False):
            # Skip SALAD preprocessing for direct backbone inference
            dataset_timing["preprocessing"] = {"total": 0.0}
        else:
            (_, _), preproc_timing = run_preprocessing_pipeline(args, world_size, rank, models=models)
            dataset_timing["preprocessing"] = preproc_timing

        if args.is_sequential:
            dataset_pair = SequentialTwoViewDataset(args)
        else:
            dataset_pair = BaseTwoViewDataset(args)

        _run_fn = _select_pipeline(args)
        pred_dir, inference_timing = _run_fn(args, dataset_pair, world_size, rank, device, dtype, models=models)
        dataset_timing.update(inference_timing)

    # Evaluate (only on rank 0)
    if rank == 0:
        # Load ground truth reconstruction (shared across all modes)
        gt_reconstruction = None
        if h5py_path_orig is not None:
            from gluemap.datasets.co3d_utils import load_co3d_gt_as_reconstruction
            gt_reconstruction = load_co3d_gt_as_reconstruction(h5py_path_orig, images_list)
        elif gt_path is not None:
            gt_reconstruction = load_colmap_reconstruction(gt_path)

        eval_config = config.get("evaluation", {})
        pose_thresholds = eval_config.get("pose_thresholds", [1, 3, 5])
        error_metric = eval_config.get("error_metric", "pose")
        match_by_basename = eval_config.get("match_by_basename", False) or (h5py_path_orig is not None)
        use_evo = eval_config.get("use_evo", False)
        heatmap_path = "pairwise_error_heatmap.png" if show_viz else None

        # Track ablation returns a dict of {mode: pred_dir}; evaluate each mode
        if isinstance(pred_dir, dict):
            metrics = {}
            for mode, mode_pred_dir in pred_dir.items():
                pred_path = os.path.join(output_path, mode_pred_dir)
                pred_reconstruction = load_colmap_reconstruction(pred_path)
                if gt_reconstruction is not None:
                    mode_metrics = compare_reconstructions(
                        pred_reconstruction,
                        gt_reconstruction,
                        pose_thresholds=pose_thresholds,
                        save_path=heatmap_path,
                        error_metric=error_metric,
                        match_by_basename=match_by_basename,
                        use_evo=use_evo,
                    )
                else:
                    mode_metrics = {"num_images": len(pred_reconstruction.images)}
                # Prefix each metric key with the mode
                for k, v in mode_metrics.items():
                    metrics[f"{mode}/{k}"] = v
            return metrics, dataset_timing
        else:
            pred_path = os.path.join(output_path, pred_dir)
            pred_reconstruction = load_colmap_reconstruction(pred_path)
            if gt_reconstruction is not None:
                metrics = compare_reconstructions(
                    pred_reconstruction,
                    gt_reconstruction,
                    pose_thresholds=pose_thresholds,
                    save_path=heatmap_path,
                    error_metric=error_metric,
                    match_by_basename=match_by_basename,
                    use_evo=use_evo,
                )
            else:
                metrics = {"num_images": len(pred_reconstruction.images)}
            return metrics, dataset_timing

    return {}, dataset_timing


def run_benchmark(
    config_path: str,
    csv_name: str = "benchmark_results.csv",
    experiment_indices: Optional[List[int]] = None,
    use_evo: bool = False,
    show_viz: bool = False,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    """Run benchmark on all datasets specified in the config.

    Args:
        config_path: Path to the YAML config file
        csv_name: Name of the output CSV file

    Returns:
        Tuple of (results dict, failures dict)
    """
    # Load config
    config = load_config(config_path)

    # Initialize distributed
    args = create_args_from_config(config, "", "")
    rank, world_size, device, dtype = init_distributed(args)

    # Pre-load all models once for reuse across datasets
    benchmark_opts = config.get("benchmark", {})
    if benchmark_opts.get("preload_models", True):
        models, device = load_all_models(args)
    else:
        models = None

    # Override use_evo in config if CLI flag is set
    if use_evo:
        config.setdefault("evaluation", {})["use_evo"] = True

    continue_on_error = benchmark_opts.get("continue_on_error", True)

    # Load datasets/experiments from config
    datasets, output_pattern = load_datasets_from_config(config, verbose=(rank == 0))

    # Filter to specific experiment indices if provided
    if experiment_indices is not None:
        datasets = [datasets[i] for i in experiment_indices if i < len(datasets)]

    results = {}
    failures = {}
    all_timing = {}

    for dataset_name, images_path, gt_path, output_path, images_list, preprocess_path in datasets:
        # try:
        if True:
            metrics, dataset_timing = run_single_dataset(
                config,
                dataset_name,
                images_path,
                gt_path,
                output_path,
                world_size,
                rank,
                device,
                dtype,
                images_list,
                preprocess_path,
                show_viz,
                models=models,
            )
            if rank == 0:
                results[dataset_name] = metrics
                all_timing[dataset_name] = dataset_timing
                print(f"\nCompleted {dataset_name}: {metrics}")

        # except Exception as e:
        else:
            if rank == 0:
                failures[dataset_name] = str(e)
                print(f"\nFailed on {dataset_name}: {e}")

            if not continue_on_error:
                raise

    # Output results (rank 0 only)
    if rank == 0:
        print_summary(results, failures)
        output_dir = get_output_dir_from_pattern(output_pattern)
        save_results_csv(results, failures, output_dir, csv_name)

        # # Save timing data
        # timing_path = os.path.join(output_dir, "pipeline_timing.pth")
        # torch.save(all_timing, timing_path)
        # print(f"[Profiling] Timing data saved to: {timing_path}")

    return results, failures


def main():
    parser = argparse.ArgumentParser(
        description="Run GLUEMAP benchmark on multiple datasets"
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML config file",
    )
    parser.add_argument(
        "--csv-name",
        type=str,
        default="benchmark_results.csv",
        help="Name of the output CSV file (default: benchmark_results.csv)",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="-1",
        help="Comma-separated GPU IDs to use (e.g., '0', '0,1,2'). -1 means use all available GPUs (default: -1)",
    )
    parser.add_argument(
        "--experiment-indices",
        type=str,
        default=None,
        help="Comma-separated experiment indices to run (e.g., '0,3,6'). If not set, run all.",
    )
    parser.add_argument(
        "--use-evo",
        action="store_true",
        default=False,
        help="Compute evo trajectory metrics (ATE, ARE, RPE) alongside pairwise AUC",
    )
    parser.add_argument(
        "--show-viz",
        action="store_true",
        default=False,
        help="Save pairwise error heatmap visualizations (off by default)",
    )
    args = parser.parse_args()

    print("args:", args)

    if args.gpus != "-1":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    experiment_indices = None
    if args.experiment_indices is not None:
        experiment_indices = [int(x) for x in args.experiment_indices.split(",")]

    run_benchmark(args.config, args.csv_name, experiment_indices, use_evo=args.use_evo, show_viz=args.show_viz)


if __name__ == "__main__":
    main()
