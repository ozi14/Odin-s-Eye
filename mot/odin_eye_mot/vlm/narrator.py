"""
narrator.py — Qwen3-VL Scene Narration Engine

Periodically invokes a Vision-Language Model (Qwen3-VL-4B-Instruct) to
generate structured natural-language descriptions of tracked pedestrians
and overall scene dynamics.

Supports two backends:
  1. MLX  — for Apple Silicon (M4 Max), via mlx-vlm
  2. HuggingFace Transformers — for CUDA GPUs (Colab A100)

Narration modes:
  - scene_summary   : crowd density, movement patterns, notable events
  - person_describe : clothing, accessories, posture for a specific track ID
  - interaction     : detect and describe person-to-person interactions
  - anomaly         : flag unusual behaviour vs crowd flow

Output: structured JSON dict with fields:
  {
    "mode": "scene_summary",
    "frame_id": 42,
    "timestamp_s": 1.4,
    "scene": { ... },
    "persons": { "T001": { ... }, ... },
    "flagged_events": [ ... ]
  }

Usage:
    narrator = Narrator(backend='mlx')  # or 'transformers'
    result = narrator.narrate(frame_bgr, tracks, mode='scene_summary',
                              frame_id=42, fps=30.0)
"""

from __future__ import annotations

import json
import re
import textwrap
import numpy as np
import cv2
from typing import List, Optional, Dict, Any

# ─────────────────────────────────────────────────────────────────────
# Prompt templates
# ─────────────────────────────────────────────────────────────────────

_PROMPT_SCENE = textwrap.dedent("""\
    You are analysing a surveillance video frame with tracked pedestrians.
    Bounding boxes and track IDs are drawn on the image.

    Provide a JSON response with this exact structure:
    {
      "crowd_density": "<sparse|moderate|dense|very_dense>",
      "crowd_count": <integer estimate>,
      "movement_patterns": "<description of overall flow and direction>",
      "notable_events": ["<event 1>", "<event 2>"],
      "scene_description": "<1-2 sentence overall description>"
    }
    Respond with JSON only. No explanation outside the JSON.
""")

_PROMPT_PERSON = textwrap.dedent("""\
    You are analysing a surveillance video frame.
    The person with track ID {track_id} is highlighted with a brighter bounding box.

    Provide a JSON response describing ONLY that person:
    {{
      "track_id": "{track_id}",
      "clothing_top": "<colour and type, e.g. red jacket>",
      "clothing_bottom": "<colour and type, e.g. blue jeans>",
      "accessories": ["<item 1>", "<item 2>"],
      "posture": "<walking|running|standing|sitting|other>",
      "movement_direction": "<towards camera|away|left|right|stationary>",
      "notable_features": "<any distinguishing characteristics>"
    }}
    Respond with JSON only.
""")

_PROMPT_INTERACTION = textwrap.dedent("""\
    You are analysing a surveillance video frame with multiple tracked pedestrians.

    Identify any interactions between tracked people. Provide JSON:
    {
      "interactions": [
        {
          "persons": ["T001", "T002"],
          "type": "<conversation|walking_together|confrontation|crowd_merge|other>",
          "description": "<brief description>"
        }
      ],
      "interaction_count": <integer>
    }
    If no interactions, return {"interactions": [], "interaction_count": 0}.
    Respond with JSON only.
""")

_PROMPT_ANOMALY = textwrap.dedent("""\
    You are a security analyst reviewing a surveillance frame.
    Most pedestrians are moving normally. Identify anyone behaving unusually.

    Provide JSON:
    {
      "anomalies": [
        {
          "track_id": "<ID or 'unknown'>",
          "behaviour": "<description of unusual behaviour>",
          "severity": "<low|medium|high>"
        }
      ],
      "anomaly_count": <integer>,
      "overall_risk": "<normal|elevated|high>"
    }
    If nothing unusual, return {"anomalies": [], "anomaly_count": 0, "overall_risk": "normal"}.
    Respond with JSON only.
""")


_MODE_PROMPTS = {
    'scene_summary': _PROMPT_SCENE,
    'person_describe': _PROMPT_PERSON,
    'interaction': _PROMPT_INTERACTION,
    'anomaly': _PROMPT_ANOMALY,
}


# ─────────────────────────────────────────────────────────────────────
# Overlay drawing
# ─────────────────────────────────────────────────────────────────────

def _draw_overlay(frame_bgr: np.ndarray,
                  tracks,
                  highlight_id: Optional[int] = None) -> np.ndarray:
    """
    Draw bounding boxes + track IDs on a copy of the frame.

    Args:
        tracks:        list of Track objects (with .bbox_xyxy, .track_id)
        highlight_id:  if set, draw this track with a distinct colour
    """
    img = frame_bgr.copy()
    for t in tracks:
        if t.bbox_xyxy is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in t.bbox_xyxy]
        if t.track_id == highlight_id:
            color = (0, 255, 255)   # cyan highlight
            thickness = 3
        else:
            color = (0, 200, 0)     # green
            thickness = 2

        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        label = f"T{t.track_id:03d}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return img


def _frame_to_pil(frame_bgr: np.ndarray):
    from PIL import Image
    return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))


# ─────────────────────────────────────────────────────────────────────
# JSON extraction helper
# ─────────────────────────────────────────────────────────────────────

def _extract_json(text: str) -> Dict[str, Any]:
    """Extract the first valid JSON object from model output text."""
    # Try direct parse first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Find {...} block
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return {"raw_text": text, "parse_error": True}


# ─────────────────────────────────────────────────────────────────────
# MLX backend
# ─────────────────────────────────────────────────────────────────────

class _MLXBackend:
    """Qwen3-VL via mlx-vlm on Apple Silicon."""

    MODEL_ID = "mlx-community/Qwen3-VL-4B-Instruct-4bit"

    def __init__(self, model_id: Optional[str] = None):
        from mlx_vlm import load, generate
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import load_config

        mid = model_id or self.MODEL_ID
        print(f"Loading VLM via MLX: {mid}")
        self.model, self.processor = load(mid)
        self.config = load_config(mid)
        self._generate = generate
        self._apply_chat_template = apply_chat_template
        print("MLX VLM ready.")

    def generate(self, pil_image, prompt: str, max_tokens: int = 512) -> str:
        formatted = self._apply_chat_template(
            self.processor, self.config, prompt,
            num_images=1, add_generation_prompt=True)
        output = self._generate(
            self.model, self.processor, formatted,
            image=pil_image, max_tokens=max_tokens, verbose=False)
        # mlx-vlm may return a GenerationResult object instead of str
        if hasattr(output, 'text'):
            return output.text
        return str(output)


# ─────────────────────────────────────────────────────────────────────
# HuggingFace Transformers backend
# ─────────────────────────────────────────────────────────────────────

class _TransformersBackend:
    """Qwen3-VL via HuggingFace Transformers on CUDA."""

    MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"

    def __init__(self, model_id: Optional[str] = None, device: str = 'cuda'):
        import torch
        from transformers import AutoProcessor, AutoModelForVision2Seq

        mid = model_id or self.MODEL_ID
        print(f"Loading VLM via Transformers: {mid} on {device}")
        self.processor = AutoProcessor.from_pretrained(mid, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            mid, trust_remote_code=True, torch_dtype=torch.float16).to(device)
        self.model.eval()
        self.device = device
        print("Transformers VLM ready.")

    def generate(self, pil_image, prompt: str, max_tokens: int = 512) -> str:
        import torch
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": pil_image},
                {"type": "text",  "text": prompt},
            ]}
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text], images=[pil_image], return_tensors="pt",
            padding=True).to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_tokens)
        decoded = self.processor.batch_decode(
            out[:, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True)[0]
        return decoded.strip()


# ─────────────────────────────────────────────────────────────────────
# Main Narrator
# ─────────────────────────────────────────────────────────────────────

class Narrator:
    """
    VLM-backed scene narrator for ByteTrack output.

    Args:
        backend:        'mlx' (Apple Silicon) or 'transformers' (CUDA)
        model_id:       override default model ID
        narrate_every:  emit scene_summary every N frames
        device:         for transformers backend
    """

    def __init__(
        self,
        backend:      str = 'mlx',
        model_id:     Optional[str] = None,
        narrate_every: int = 30,
        device:       str = 'cuda',
    ):
        self.narrate_every = narrate_every
        self._last_narrate_frame = -narrate_every

        if backend == 'mlx':
            self._vlm = _MLXBackend(model_id)
        elif backend == 'transformers':
            self._vlm = _TransformersBackend(model_id, device)
        else:
            raise ValueError(f"Unknown backend: {backend!r}. Use 'mlx' or 'transformers'.")

    def should_narrate(self, frame_id: int) -> bool:
        return (frame_id - self._last_narrate_frame) >= self.narrate_every

    def narrate(
        self,
        frame_bgr:    np.ndarray,
        tracks,
        mode:         str = 'scene_summary',
        frame_id:     int = 0,
        fps:          float = 30.0,
        highlight_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate a structured narration for the given frame and tracks.

        Args:
            frame_bgr:    current video frame (BGR)
            tracks:       list of active Track objects
            mode:         one of 'scene_summary', 'person_describe',
                          'interaction', 'anomaly'
            frame_id:     frame index
            fps:          frames per second (for timestamp)
            highlight_id: track_id to highlight (for person_describe mode)

        Returns:
            dict with narration results
        """
        if mode not in _MODE_PROMPTS:
            raise ValueError(f"Unknown mode {mode!r}. "
                             f"Choose from {list(_MODE_PROMPTS)}")

        self._last_narrate_frame = frame_id

        # Build annotated frame
        overlay = _draw_overlay(frame_bgr, tracks, highlight_id)
        pil_img = _frame_to_pil(overlay)

        # Build prompt
        prompt = _MODE_PROMPTS[mode]
        if mode == 'person_describe' and highlight_id is not None:
            prompt = prompt.format(track_id=f"T{highlight_id:03d}")

        # Generate
        raw = self._vlm.generate(pil_img, prompt, max_tokens=512)
        parsed = _extract_json(raw)

        result = {
            "mode":         mode,
            "frame_id":     frame_id,
            "timestamp_s":  round(frame_id / max(fps, 1.0), 3),
            "track_count":  len(tracks),
            "narration":    parsed,
        }
        return result

    def narrate_batch(
        self,
        frames_and_tracks: List[Dict],
        fps: float = 30.0,
    ) -> List[Dict]:
        """
        Convenience wrapper for processing multiple frames.

        Each entry in frames_and_tracks should be:
        {
          "frame_bgr": np.ndarray,
          "tracks":    List[Track],
          "frame_id":  int,
          "mode":      str (optional, default 'scene_summary')
        }
        """
        results = []
        for item in frames_and_tracks:
            res = self.narrate(
                frame_bgr=item['frame_bgr'],
                tracks=item['tracks'],
                mode=item.get('mode', 'scene_summary'),
                frame_id=item['frame_id'],
                fps=fps,
            )
            results.append(res)
        return results
