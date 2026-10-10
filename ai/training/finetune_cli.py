#!/usr/bin/env python3
"""
Command-line fine-tuning — the same engine as the /train web page.
Use this for a personal dataset without keeping the web app running.

Usage:
    python -m ai.training.finetune_cli samples.json --name my_handwriting --epochs 30
    python -m ai.training.finetune_cli samples.json --name my_handwriting --extra-chars "&@+=" --epochs 30 --base default
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ai.training.finetune import TrainableHandwritingModel, load_training_samples, fit, build_alphabet, load_profile_alphabet
from ai.training.devices import normalize_device

PROFILES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "profiles")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("samples_json", help="Path to a JSON file exported from webapp/capture_tool/index.html")
    p.add_argument("--name", required=True, help="Profile name to save as (used later on the Write & Export page)")
    p.add_argument("--base", default="default", help="Base profile to fine-tune from (default: 'default')")
    p.add_argument("--extra-chars", default="", help="New characters to add to the alphabet, e.g. '&@+='")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--device", default="auto", help="Training device: auto (prefer GPU), cpu, cuda, or cuda:N (visible GPU index)")
    p.add_argument("--profiles-dir", default=PROFILES_DIR, help="Where profiles live (default: profiles)")
    args = p.parse_args()
    try:
        args.device = normalize_device(args.device)
    except ValueError as error:
        p.error(str(error))
    for name in (args.name, args.base):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", name):
            p.error("Profile names must contain only letters, digits, underscores or hyphens (1–80 characters).")
    if args.epochs < 1 or not 0 < args.lr <= 0.01:
        p.error("Use positive epochs and a learning rate between 0 and 0.01.")
    out_path = os.path.join(args.profiles_dir, f"{args.name}.npz")
    if os.path.exists(out_path):
        p.error("That profile already exists. Choose a new name to preserve it.")

    samples = load_training_samples(args.samples_json)
    print(f"Loaded {len(samples)} samples from {args.samples_json}")
    if len(samples) == 0:
        print("No usable samples — check the capture tool export.", file=sys.stderr)
        sys.exit(1)

    base_weights = os.path.join(args.profiles_dir, f"{args.base}.npz")
    if not os.path.exists(base_weights):
        print(f"Base profile not found: {base_weights}", file=sys.stderr)
        sys.exit(1)

    alphabet = build_alphabet(args.extra_chars, load_profile_alphabet(base_weights), samples)

    try:
        model = TrainableHandwritingModel(alphabet=alphabet, weights_path=base_weights, device=args.device)
    except (ImportError, ValueError) as error:
        p.error(str(error))
    print(f"Training device: {model.device}")
    print(f"Alphabet size: {len(alphabet)} ({len(alphabet) - 73} new char(s))" if len(alphabet) > 73 else "Alphabet unchanged (73 chars)")
    fit(model, samples, epochs=args.epochs, lr=args.lr, log_every=max(1, args.epochs // 20))

    os.makedirs(args.profiles_dir, exist_ok=True)
    model.export_to_npz(out_path)
    print(f"Saved profile '{args.name}' -> {out_path}")
    print("It'll now appear in the Write & Export page's profile dropdown (restart the web app if it's already running).")


if __name__ == "__main__":
    main()
