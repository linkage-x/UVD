import json
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch

import uvd
from utils.reader import RerunEpisodeReader


def _detect_default_episode_path() -> str:
    """Try a few common locations for a sample episode."""
    candidates = [
        Path("/Data/1114_left_fr3_insert_pinboard_Generalize_82ep/episode_0001"),
    ]
    for cand in candidates:
        if (cand / "data.json").exists():
            return str(cand)
    return ""


DEFAULT_EPISODE_PATH = _detect_default_episode_path()


def _parse_episode_path(episode_path: str) -> tuple[Path, int]:
    """Validate and normalize the provided episode directory."""
    if not episode_path:
        raise gr.Error("Please provide an episode directory containing data.json.")
    episode_dir = Path(episode_path).expanduser().resolve()
    if not episode_dir.exists():
        raise gr.Error(f"Episode directory not found: {episode_dir}. Is the dataset mounted?")
    if not (episode_dir / "data.json").exists():
        raise gr.Error(f"No data.json found in {episode_dir}")
    try:
        episode_idx = int(episode_dir.name.split("_")[-1])
    except ValueError as exc:
        raise gr.Error(f"Cannot parse episode index from {episode_dir.name}") from exc
    return episode_dir, episode_idx


def _load_third_person_frames(episode_dir: Path) -> tuple[np.ndarray, list[int]]:
    """Read third_person_cam_color frames and their original indices."""
    json_path = episode_dir / "data.json"
    with json_path.open("r", encoding="utf-8") as f:
        json_file = json.load(f)

    frames = []
    frame_ids = []
    for item in json_file.get("data", []):
        color_meta = (item.get("colors") or {}).get("third_person_cam_color")
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
        raise gr.Error("No valid third_person_cam_color frames found.")

    return np.stack(frames, axis=0), frame_ids


def proc_episode(episode_path: str, preprocessor_name: str):
    episode_dir, episode_idx = _parse_episode_path(episode_path)
    frames, frame_ids = _load_third_person_frames(episode_dir)
    normalized_preprocessor = preprocessor_name.lower().replace("-", "")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    keyframe_indices = uvd.get_uvd_subgoals(
        frames,
        normalized_preprocessor,
        device=device,
        return_indices=True,
    )
    keyframe_indices = np.array(keyframe_indices, dtype=int)
    keyframe_frame_ids = [frame_ids[int(idx)] for idx in keyframe_indices]

    # Attach keyframe markers in the reader output for downstream consumers.
    reader = RerunEpisodeReader(task_dir=str(episode_dir.parent))
    reader.return_episode_data(
        episode_idx,
        skip_steps_nums=1,
        keyframe_indices=keyframe_frame_ids,
    )

    subgoals = frames[keyframe_indices]
    gallery = [
        (img, f"Keyframe {fid} (#{i+1})")
        for i, (img, fid) in enumerate(zip(subgoals, keyframe_frame_ids))
    ]
    indices_str = ", ".join(str(idx) for idx in keyframe_frame_ids)
    return gallery, indices_str


with gr.Blocks() as demo:
    with gr.Row():
        episode_path = gr.Textbox(
            label="Episode directory",
            value=DEFAULT_EPISODE_PATH,
            placeholder="Path to episode_xxxx (must contain data.json)",
            scale=3,
        )
        preprocessor_name = gr.Dropdown(
            ["VIP", "R3M", "LIV", "CLIP", "DINO-v2", "VC-1", "ResNet"],
            label="Preprocessor",
            value="VIP",
            scale=1,
        )
    with gr.Row():
        output_gallery = gr.Gallery(label="UVD Keyframes", height=224, preview=True, scale=3)
        output_indices = gr.Textbox(label="Keyframe indices", interactive=False, scale=1)
    with gr.Row():
        submit = gr.Button("Submit")
        clr = gr.ClearButton(components=[episode_path, output_gallery, output_indices])

    submit.click(proc_episode, inputs=[episode_path, preprocessor_name], outputs=[output_gallery, output_indices])


demo.queue().launch(share=True, show_error=True)
