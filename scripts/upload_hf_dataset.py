#!/usr/bin/env python3
"""Script to upload the VisTA-SR dataset to Hugging Face Hub (heesup/VisTA-SR)."""

import argparse
import os
from huggingface_hub import HfApi, create_repo


def parse_args():
    parser = argparse.ArgumentParser(description="Upload VisTA-SR dataset to Hugging Face")
    parser.add_argument("--dataset-dir", type=str, default="/home/lion397/data/datasets/GEMINI/Training_T4_1_2_3", help="Path to local dataset directory")
    parser.add_argument("--repo-id", type=str, default="heesup/VisTA-SR", help="Hugging Face Dataset repository ID")
    parser.add_argument("--token", type=str, default="", help="Hugging Face write token (or run `huggingface-cli login`)")
    return parser.parse_args()


def upload_dataset():
    args = parse_args()
    api = HfApi()

    token = args.token or os.getenv("HF_TOKEN")
    if not token:
        token_path = os.path.expanduser("~/.cache/huggingface/token")
        if os.path.exists(token_path):
            with open(token_path, "r") as f:
                token = f.read().strip()

    print(f"Creating Hugging Face dataset repository if not existing: {args.repo_id}")
    try:
        create_repo(repo_id=args.repo_id, repo_type="dataset", token=token, exist_ok=True)
    except Exception as e:
        print(f"Repo status: {e}")

    print(f"Uploading dataset directory '{args.dataset_dir}' to 'https://huggingface.co/datasets/{args.repo_id}'...")
    api.upload_folder(
        folder_path=args.dataset_dir,
        repo_id=args.repo_id,
        repo_type="dataset",
        token=token
    )
    print("Upload completed successfully!")


if __name__ == "__main__":
    upload_dataset()
