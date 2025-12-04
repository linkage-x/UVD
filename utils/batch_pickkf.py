#!/usr/bin/env python3
"""Batch process all episodes under a root directory using utils.pickkf."""

import argparse
import pickle
import time
from pathlib import Path

from utils.pickkf import (
    AVAILABLE_PREPROCESSORS,
    pick_keyframes,
)


def find_episode_dirs(root_dir: Path):
    return sorted(
        [p for p in root_dir.iterdir() if p.is_dir() and p.name.startswith("episode_")]
    )


def main():
    parser = argparse.ArgumentParser(description="Batch run pickkf over all episodes.")
    parser.add_argument(
        "--root_dir",
        default="/Data/1114_left_fr3_insert_pinboard_Generalize_82ep",
        help="Directory containing episode_xxxx folders.",
    )
    parser.add_argument(
        "--preprocessor",
        default="vip",
        help=f"Preprocessor name (available: {', '.join(AVAILABLE_PREPROCESSORS)}).",
    )
    parser.add_argument(
        "--cam_name",
        default="third_person_cam_color",
        help="Camera key under colors to use as UVD input.",
    )
    parser.add_argument(
        "--skip_steps",
        type=int,
        default=1,
        help="Skip interval when reading episode data.",
    )
    parser.add_argument(
        "--append_original_data",
        dest="append_original_data",
        action="store_true",
        default=True,
        help="Append is_keyframe flag back to data.json (default: True).",
    )
    parser.add_argument(
        "--no-append_original_data",
        dest="append_original_data",
        action="store_false",
        help="Do not modify the original data.json.",
    )
    parser.add_argument(
        "--output_dir",
        help="Optional directory to save per-episode pickle outputs.",
    )
    args = parser.parse_args()

    root_dir = Path(args.root_dir).expanduser().resolve()
    if not root_dir.exists():
        raise FileNotFoundError(f"Root dir not found: {root_dir}")

    episodes = find_episode_dirs(root_dir)
    if not episodes:
        raise RuntimeError(f"No episode_* directories found under {root_dir}")

    print(f"Found {len(episodes)} episodes under {root_dir}")
    print(f"Available preprocessors: {', '.join(AVAILABLE_PREPROCESSORS)}")

    if args.output_dir:
        out_dir = Path(args.output_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = None

    successes = 0
    failures = 0
    total_start = time.perf_counter()
    for ep_dir in episodes:
        ep_start = time.perf_counter()
        ep_name = ep_dir.name
        print(f"\nProcessing {ep_name} ...")
        try:
            episode_data, kf_indices, kf_frame_ids = pick_keyframes(
                data_dir=str(ep_dir),
                preprocessor=args.preprocessor,
                skip_steps=args.skip_steps,
                cam_name=args.cam_name,
                append_original_data=args.append_original_data,
            )
            elapsed = time.perf_counter() - ep_start
            print(
                f"{ep_name}: {len(kf_frame_ids)} keyframes, "
                f"episode_data_len={len(episode_data)}, time={elapsed:.2f}s"
            )
            print(f"Keyframe frame_ids: {kf_frame_ids}")

            if out_dir:
                out_path = out_dir / f"{ep_name}_data.pkl"
                with out_path.open("wb") as f:
                    pickle.dump(episode_data, f)
                print(f"Saved {ep_name} data to {out_path}")
            successes += 1
        except Exception as exc:
            failures += 1
            print(f"Failed {ep_name}: {exc}")

    total_elapsed = time.perf_counter() - total_start
    print(
        f"\nDone. Success: {successes}, Failures: {failures}, "
        f"Total episodes: {len(episodes)}, Total time: {total_elapsed:.2f}s"
    )


if __name__ == "__main__":
    main()
