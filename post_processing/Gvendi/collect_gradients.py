import os
import sys
from argparse import Namespace, ArgumentParser
from pathlib import Path
from typing import Tuple, Dict, List

import ipdb
import jsonlines
from transformers import AutoTokenizer, AutoModelForCausalLM

# from gradient_computer import GradientComputer


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--dataset_filename", type=str)

    parser.add_argument("--device_split_size", type=int, default=8)

    args = parser.parse_args()
    args.dataset_filename = Path(args.dataset_filename)
    args.save_directory = (Path("./data/gradient_storage") /
                           f"{args.dataset_filename.stem}--{args.model_name_or_path.split('/')[-1].lower()}")
    os.makedirs(args.save_directory, exist_ok=True)

    print(f"Saving gradients to {args.save_directory}.")

    # Detect the start index of sample whose gradient is not yet computed
    args.start_idx, args.end_idx = find_start_and_end_idx(args)

    return args


def find_start_and_end_idx(args: Namespace) -> Tuple:
    # get cuda visible device
    cuda_visible_device = int(os.getenv("CUDA_VISIBLE_DEVICES"))

    # get the original start idx for this device
    with jsonlines.open(args.dataset_filename) as f:
        dataset_sample_ids = [s["id"] for s in f]

    args.per_device_size = int(len(dataset_sample_ids) / args.device_split_size) + 1
    start_idx_for_this_device = list(range(0, len(dataset_sample_ids), args.per_device_size))[cuda_visible_device]
    end_idx_for_this_device = start_idx_for_this_device + args.per_device_size

    # get all generation filenames correspond to current device
    precomputed_id_filenames = list(args.save_directory.glob("*.txt"))
    precomputed_start_indices = [int(filename.stem) for filename in precomputed_id_filenames]
    precomputed_start_indices = [
        idx for idx in precomputed_start_indices
        if start_idx_for_this_device <= idx < end_idx_for_this_device
    ]

    if len(precomputed_start_indices) == 0:
        start_idx = start_idx_for_this_device
    else:
        # get the last filename
        last_filename = args.save_directory / f"{max(precomputed_start_indices)}.txt"

        # get the ids of samples in `last_filename`
        with jsonlines.open(last_filename) as f:
            last_ids = [s["id"] for s in f]

        # get the index of last sample in the dataset
        last_ids_indices_in_dataset = []
        for sample_id in last_ids:
            last_ids_indices_in_dataset.append(dataset_sample_ids.index(sample_id))

        last_idx = max(last_ids_indices_in_dataset)
        print(f"Detected Last Pre-computed Gradient Index: {last_idx}.")

        start_idx = last_idx + 1

    print(f"Processing from Index {start_idx} to Index {end_idx_for_this_device}")

    return start_idx, end_idx_for_this_device


def get_model_and_tokenizer(model_name_or_path: str):
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype="auto").cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    return model, tokenizer


def get_dataset(filename: str, start_idx: int, end_idx: int) -> List[Dict]:
    with jsonlines.open(filename) as f:
        samples = list(f)[start_idx:end_idx]

    if len(samples) > 0:
        assert all(key in samples[0] for key in ["prompt", "completion", "id"]), \
            "Datapoint does not include `prompt`, `completion`, or `id`."
    else:
        print(f"List is empty, exiting.")
        sys.exit(0)

    return samples


if __name__ == "__main__":
    args = parse_args()

    # load models and tokenizer
    model, tokenizer = get_model_and_tokenizer(args.model_name_or_path)

    # load dataset
    samples = get_dataset(args.dataset_filename, args.start_idx, args.end_idx)

    # compute gradients and store them
    collector = GradientComputer(model_name=args.model_name_or_path, model=model, tokenizer=tokenizer)
    collector.compute_project_store_gradients(samples, args.save_directory, args.start_idx)





