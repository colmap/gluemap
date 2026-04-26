"""Multi-GPU benchmark runner that distributes experiments across GPUs.

Each GPU runs subprocess(es) handling a subset of experiments.
Usage:
    python benchmark/run_benchmark_multi.py --config benchmark/configs/imc.yaml --gpus 0,1,2,3
    python benchmark/run_benchmark_multi.py --config benchmark/configs/imc.yaml --gpus 0,1 --procs-per-gpu 2
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict

import torch

from benchmark.utils import load_config, load_datasets_from_config


def main():
    parser = argparse.ArgumentParser(
        description="Run GLUEMAP benchmark across multiple GPUs (one experiment per GPU)"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the YAML config file"
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="-1",
        help="Comma-separated GPU IDs (e.g., '0,1,2,3'). -1 means all available GPUs.",
    )
    parser.add_argument(
        "--procs-per-gpu",
        type=int,
        default=1,
        help="Number of experiments to run simultaneously on each GPU (default: 1)",
    )
    parser.add_argument(
        "--csv-name",
        type=str,
        default="benchmark_results.csv",
        help="Name of the output CSV file (default: benchmark_results.csv)",
    )
    args = parser.parse_args()

    # Determine GPU list
    if args.gpus == "-1":
        num_gpus = torch.cuda.device_count()
        gpu_ids = list(range(num_gpus))
    else:
        gpu_ids = [int(g) for g in args.gpus.split(",")]

    if not gpu_ids:
        print("Error: No GPUs available.")
        sys.exit(1)

    # Load config to count experiments
    config = load_config(args.config)
    datasets, _ = load_datasets_from_config(config, verbose=True)
    num_experiments = len(datasets)

    if num_experiments == 0:
        print("Error: No experiments found in config.")
        sys.exit(1)

    # Total worker slots = num_gpus * procs_per_gpu
    total_slots = len(gpu_ids) * args.procs_per_gpu
    print(
        f"Distributing {num_experiments} experiment(s) across {len(gpu_ids)} GPU(s) "
        f"with {args.procs_per_gpu} proc(s) per GPU ({total_slots} total slots): {gpu_ids}"
    )

    # Round-robin assign experiments to (gpu, slot) workers
    # Each worker is identified by (gpu_id, slot_index) and gets a list of experiment indices
    workers = []
    for slot in range(args.procs_per_gpu):
        for gpu_id in gpu_ids:
            workers.append((gpu_id, slot))

    worker_to_indices = defaultdict(list)
    for i in range(num_experiments):
        worker = workers[i % len(workers)]
        worker_to_indices[worker].append(i)

    # Create a temporary directory for intermediate CSV files
    tmp_dir = tempfile.mkdtemp(prefix="gluemap_benchmark_")
    print(f"Intermediate CSV files will be written to: {tmp_dir}")

    # Spawn one subprocess per worker
    processes = []
    all_csv_paths = []
    for (gpu_id, slot), indices in worker_to_indices.items():
        if not indices:
            continue

        indices_str = ",".join(str(i) for i in indices)
        csv_path = os.path.join(tmp_dir, f"benchmark_results_gpu{gpu_id}_slot{slot}.csv")
        all_csv_paths.append(csv_path)

        cmd = [
            sys.executable,
            "-m", "benchmark.run_benchmark",
            "--config", args.config,
            "--gpus", str(gpu_id),
            "--csv-name", csv_path,
            "--experiment-indices", indices_str,
        ]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        print(f"  GPU {gpu_id} slot {slot}: experiments {indices_str}")
        proc = subprocess.Popen(cmd, env=env)
        processes.append((gpu_id, slot, proc))

    # Wait for all subprocesses
    failed = []
    for gpu_id, slot, proc in processes:
        proc.wait()
        if proc.returncode != 0:
            failed.append((gpu_id, slot))
            print(f"GPU {gpu_id} slot {slot} exited with code {proc.returncode}")

    if failed:
        print(f"\nFailed worker(s): {failed}")
        failed_indices = []
        for gpu_id, slot in failed:
            failed_indices.extend(worker_to_indices[(gpu_id, slot)])
        failed_indices.sort()
        print(f"\nFailed experiments ({len(failed_indices)}):")
        for idx in failed_indices:
            print(f"  [{idx}] {datasets[idx][0]}")
        failed_file = args.csv_name.replace(".csv", "_failed.txt")
        with open(failed_file, "w") as f:
            for idx in failed_indices:
                f.write(f"{datasets[idx][0]}\n")
        print(f"Failed experiments written to: {failed_file}")
        print(f"\nKeeping temporary directory for debugging: {tmp_dir}")
    else:
        print(f"\nAll {len(processes)} subprocess(es) completed successfully.")
        # Merge per-worker CSV files into a combined result
        merge_csv_results(config, all_csv_paths, args.csv_name)
        # Clean up temporary directory only on full success
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"Cleaned up temporary directory: {tmp_dir}")


def merge_csv_results(config, csv_paths, output_csv_name):
    """Merge per-worker CSV files into a single combined CSV.

    Args:
        config: Loaded YAML config dict.
        csv_paths: List of absolute paths to per-worker CSV files.
        output_csv_name: Filename for the merged CSV (written to output_dir).
    """
    import csv
    from benchmark.utils import get_output_dir_from_pattern, load_datasets_from_config

    _, output_pattern = load_datasets_from_config(config, verbose=False)
    output_dir = get_output_dir_from_pattern(output_pattern)

    all_rows = []
    header = None

    for csv_path in csv_paths:
        if not os.path.exists(csv_path):
            continue
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)
            if rows:
                if header is None:
                    header = rows[0]
                all_rows.extend(rows[1:])

    if header and all_rows:
        combined_path = os.path.join(output_dir, output_csv_name)
        os.makedirs(output_dir, exist_ok=True)
        with open(combined_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(sorted(all_rows))
        print(f"Combined results saved to: {combined_path}")


if __name__ == "__main__":
    main()
