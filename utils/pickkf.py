#!/usr/bin/env python3
"""CLI helper to pick UVD keyframes from an episode and emit processed episode data."""

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch

import uvd
from utils.reader import RerunEpisodeReader


DEFAULT_DATA_DIR = "/Data/1114_left_fr3_insert_pinboard_Generalize_82ep/episode_0001"
AVAILABLE_PREPROCESSORS = ["vip", "r3m", "liv", "clip", "dino-v2", "vc-1", "resnet"]
def _append_keyframe_flags(episode_dir: Path, keyframe_frame_ids):
    """Append is_keyframe flag into the original data.json."""
    json_path = episode_dir / "data.json"
    with json_path.open("r", encoding="utf-8") as f:
        json_file = json.load(f)
    kf_set = set(int(i) for i in keyframe_frame_ids)
    for item in json_file.get("data", []):
        frame_idx = item.get("idx")
        item["is_keyframe"] = frame_idx in kf_set
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(json_file, f, ensure_ascii=False, indent=2)


def _parse_episode_path(episode_path: str) -> Tuple[Path, int]:
    episode_dir = Path(episode_path).expanduser().resolve()
    if not episode_dir.exists():
        raise FileNotFoundError(f"Episode directory not found: {episode_dir}")
    json_path = episode_dir / "data.json"
    if not json_path.exists():
        raise FileNotFoundError(f"No data.json found in {episode_dir}")
    try:
        episode_idx = int(episode_dir.name.split("_")[-1])
    except ValueError as exc:
        raise ValueError(f"Cannot parse episode index from {episode_dir.name}") from exc
    return episode_dir, episode_idx


def _load_frames(episode_dir: Path, cam_name: str) -> Tuple[np.ndarray, list]:
    """Read specified camera frames and their original indices."""
    json_path = episode_dir / "data.json"
    with json_path.open("r", encoding="utf-8") as f:
        json_file = json.load(f)

    frames = []
    frame_ids = []
    for item in json_file.get("data", []):
        color_meta = (item.get("colors") or {}).get(cam_name)
        if not color_meta:
            continue
        img_path = episode_dir / color_meta["path"]
        if not img_path.exists():
            continue
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frames.append(image)
        frame_ids.append(item.get("idx", len(frame_ids)))

    if len(frames) == 0:
        raise RuntimeError(f"No valid {cam_name} frames found.")

    return np.stack(frames, axis=0), frame_ids


def pick_keyframes(
    data_dir: str = DEFAULT_DATA_DIR,
    preprocessor: str = "vip",
    skip_steps: int = 1,
    cam_name: str = "third_person_cam_color",
    append_original_data: bool = True,
):
    episode_dir, episode_idx = _parse_episode_path(data_dir)
    frames, frame_ids = _load_frames(episode_dir, cam_name)
    normalized_preprocessor = preprocessor.lower().replace("-", "")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    keyframe_indices = uvd.get_uvd_subgoals(
        frames,
        normalized_preprocessor,
        device=device,
        return_indices=True,
    )
    keyframe_indices = np.array(keyframe_indices, dtype=int)
    keyframe_frame_ids = [frame_ids[int(idx)] for idx in keyframe_indices]

    reader = RerunEpisodeReader(task_dir=str(episode_dir.parent))
    episode_data = reader.return_episode_data(
        episode_idx,
        skip_steps_nums=skip_steps,
        keyframe_indices=keyframe_frame_ids,
    )

    if append_original_data:
        _append_keyframe_flags(episode_dir, keyframe_frame_ids)

    return episode_data, keyframe_indices.tolist(), keyframe_frame_ids


def main():
    parser = argparse.ArgumentParser(description="Pick UVD keyframes and emit episode data.")
    parser.add_argument(
        "--data_dir",
        default=DEFAULT_DATA_DIR,
        help="Path to episode_xxxx directory (must contain data.json).",
    )
    parser.add_argument(
        "--preprocessor",
        default="vip",
        help="Preprocessor name for UVD (e.g., vip, r3m, liv, clip, dino-v2, vc-1, resnet).",
    )
    parser.add_argument(
        "--skip_steps",
        type=int,
        default=1,
        help="Skip interval when reading episode data.",
    )
    parser.add_argument(
        "--cam_name",
        default="third_person_cam_color",
        help="Camera key under colors to use as UVD input.",
    )
    parser.add_argument(
        "--output",
        help="Optional path to save full episode_data via pickle.",
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
    args = parser.parse_args()

    np.set_printoptions(threshold=20, edgeitems=2)
    print(f"Available preprocessors: {', '.join(AVAILABLE_PREPROCESSORS)}")
    normalized_pre = args.preprocessor.lower().replace("-", "")
    if normalized_pre not in [p.replace("-", "") for p in AVAILABLE_PREPROCESSORS]:
        print(f"Warning: '{args.preprocessor}' not in known list, proceeding anyway.")

    start_time = time.perf_counter()
    episode_data, kf_indices, kf_frame_ids = pick_keyframes(
        data_dir=args.data_dir,
        preprocessor=args.preprocessor,
        skip_steps=args.skip_steps,
        cam_name=args.cam_name,
        append_original_data=args.append_original_data,
    )
    elapsed = time.perf_counter() - start_time

    print(f"Keyframe frame_ids (original episode idx): {kf_frame_ids}")
    print(f"Episode data length (after skip): {len(episode_data)}")
    print(f"Processing time: {elapsed:.2f}s")

    if args.output:
        out_path = Path(args.output).expanduser().resolve()
        with out_path.open("wb") as f:
            pickle.dump(episode_data, f)
        print(f"Saved full episode_data to {out_path}")


if __name__ == "__main__":
    main()
