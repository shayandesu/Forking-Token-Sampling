import os
from argparse import Namespace, ArgumentParser
from pathlib import Path

import ipdb
import jsonlines

from gradient_vendi import GradientVendi

def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--dataset_filename", type=str)
    parser.add_argument("--gradient_storage_dir", type=str)

    args = parser.parse_args()
    args.dataset_filename = Path(args.dataset_filename)
    args.gradient_storage_dir = Path(args.gradient_storage_dir)

    return args


if __name__ == "__main__":
    args = parse_args()

    # load gradients
    sample_ids, sample_gradients = GradientVendi.load_all_gradients(args.gradient_storage_dir)

    # check if all gradients for the dataset are loaded
    with jsonlines.open(args.dataset_filename) as f:
        dataset_sample_ids = set(s['id'] for s in list(f))

    assert dataset_sample_ids == set(sample_ids), "dataset_sample_ids != set(sample_ids)."

    g_vendi = GradientVendi.compute_gradient_vendi(sample_gradients)
    print(f"G-Vendi for {args.dataset_filename}: {g_vendi}")

# python compute_g-vendi.py --dataset_filename=./data/datasets/seed.jsonl --gradient_storage=./data/gradient_storage/train--qwen2.5-0.5b-instruct

