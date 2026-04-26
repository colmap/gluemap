from gluemap.controllers.gluemap_impl import run_inference_pipeline
from gluemap.controllers.salad_retrieval import (
    run_preprocessing_pipeline,
    run_preprocessing_pipeline_multi,
)
from gluemap.datasets.multi_sequence_twoview_dataset import MultiSequencePairs
from gluemap.datasets.sequential_twoview_dataset import SequentialTwoViewDataset
from gluemap.datasets.twoview_dataset import BaseTwoViewDataset
from gluemap.utils.cli import get_args_parser, parse_args_with_config
from gluemap.utils.gpu import init_distributed


def demo_main():
    parser = get_args_parser()
    parser.add_argument(
        "--is_sequential",
        action="store_true",
        help="whether the images are sequentially ordered",
    )
    parser.add_argument(
        "--sample_frequency",
        type=int,
        default=1,
        help="frequency to sample images if sequential",
    )
    args = parse_args_with_config(parser)

    rank, world_size, device, dtype = init_distributed(args)

    args.curr_processed = args.write_path
    args.curr_path = args.write_path

    (_, _), _ = run_preprocessing_pipeline(args, world_size, rank)

    if "is_sequential" in args and args.is_sequential:
        dataset_pair = SequentialTwoViewDataset(args)
    else:
        dataset_pair = BaseTwoViewDataset(args)

    run_inference_pipeline(
        args, dataset_pair, world_size, rank, device, dtype
    )


def demo_lamar_main():
    import logging
    import os

    logger = logging.getLogger(__name__)

    parser = get_args_parser()
    args = parse_args_with_config(parser)

    rank, world_size, device, dtype = init_distributed(args)

    datasets = [
        x for x in sorted(os.listdir(args.images_path)) if x.startswith("ios")
    ]
    logger.info(f"datasets to process: {datasets}")

    (_, _), _ = run_preprocessing_pipeline_multi(
        args, world_size, rank, datasets
    )

    dataset_pair = MultiSequencePairs(args, datasets)

    run_inference_pipeline(
        args,
        dataset_pair,
        world_size,
        rank,
        device,
        dtype,
        pairs=dataset_pair.pairs,
    )
