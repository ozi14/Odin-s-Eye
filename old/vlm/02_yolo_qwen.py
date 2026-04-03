"""
02_yolo_qwen.py
Detects persons using a custom YOLOv11 model, crops the bounding boxes, 
and passes each crop individually to the 4-bit quantized Qwen VLM 
for structured JSON description generation.
Designed for local inference on Apple M4 Max using MLX.
"""

import os
import time
import json
import cv2
from PIL import Image
from ultralytics import YOLO
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Hardcoded to Camera 1, Frame 00000000 from the WILDTRACK val set
TEST_IMAGE_PATH = os.path.join(BASE_DIR, "datasets", "wildtrack", "images", "val", "cam1_00001850.jpg")

# Models
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "models", "yolo_wildtrack_weights", "best.pt")
VLM_MODEL_ID = os.path.join(BASE_DIR, "vlm", "model")

# Output for intermediate crops
CROPS_DIR = os.path.join(BASE_DIR, "vlm", "crops")

def setup_directories():
    if not os.path.exists(CROPS_DIR):
        os.makedirs(CROPS_DIR)
        print(f"Created directory for crops: {CROPS_DIR}")

def detect_and_crop():
    if not os.path.exists(TEST_IMAGE_PATH):
        raise FileNotFoundError(f"Test image not found at {TEST_IMAGE_PATH}")
    if not os.path.exists(YOLO_MODEL_PATH):
        raise FileNotFoundError(f"YOLO model not found at {YOLO_MODEL_PATH}")

    print("=" * 50)
    print("Step 1: Running YOLO Detector")
    print("=" * 50)
    
    detector = YOLO(YOLO_MODEL_PATH)
    
    # Run inference
    start_time = time.time()
    results = detector(TEST_IMAGE_PATH, conf=0.5) # Confidence threshold of 0.5
    yolo_time = time.time() - start_time
    print(f"⏱️ YOLO Inference Latency: {yolo_time:.4f} seconds")

    # The result contains boxes
    result = results[0]
    boxes = result.boxes
    
    image = cv2.imread(TEST_IMAGE_PATH)
    if image is None:
        raise ValueError("Failed to load image with CV2")
        
    cropped_paths = []
    
    for idx, box in enumerate(boxes):
        # Extract coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        
        # We only care about class 0 (person) if it's the standard COCO mapping, 
        # or the single class if the custom model is person-only.
        # Crop the image
        crop_img = image[y1:y2, x1:x2]
        
        # Save the crop
        crop_path = os.path.join(CROPS_DIR, f"crop_{idx:03d}.jpg")
        cv2.imwrite(crop_path, crop_img)
        cropped_paths.append((idx, crop_path, (x1, y1, x2, y2), conf))
        print(f"Saved crop {idx} with confidence {conf:.2f} to {crop_path}")

    return cropped_paths

def describe_crops(cropped_info):
    if not cropped_info:
        print("No crops to process.")
        return

    print("\n" + "=" * 50)
    print(f"Step 2: Loading VLM ({VLM_MODEL_ID}) via MLX")
    print("=" * 50)
    
    model, processor = load(VLM_MODEL_ID)
    config = load_config(VLM_MODEL_ID)

    # VLM Prompt specifically for a cropped person
    system_prompt = (
        "You are an automated visual analysis component. Return ONLY a valid JSON object. "
        "No conversational text, no markdown backticks, no explanations.\n"
        "'Occlusion' means the person is blocked from view by objects or other people. "
        "Occlusion levels must be: '0' (fully visible in the crop), '1' (if object blocked in any way-- heavily or lightly)"
    )

    print("\n" + "=" * 50)
    print("Step 3: Running VLM Inference on Crops")
    print("=" * 50)

    final_results = []
    total_vlm_latency = 0

    for idx, crop_path, bbox, conf in cropped_info:
        x1, y1, x2, y2 = bbox
        user_prompt = (
            f"Examine this cropped image of a single person. They were detected in the original 1920x1080 camera frame at pixel coordinates: X1={x1}, Y1={y1}, X2={x2}, Y2={y2}.\n"
            "Provide a structured description of this individual.\n"
            "1. Appearance (clothing colors, gender, build, accessories)\n"
            "2. Action (e.g., standing, walking, sitting)\n"
            "3. Direction of movement (e.g., moving left, moving right, moving away, stationary)\n"
            "4. Occlusion\n\n"
            "Output a single JSON object matching this exact format:\n"
            "  {\n"
            "    \"appearance\": \"...\",\n"
            "    \"action\": \"...\",\n"
            "    \"direction_of_movement\": \"...\",\n"
            "    \"occlusion\": \"...\"\n"
            "  }"
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": user_prompt}
            ]}
        ]

        prompt = apply_chat_template(processor, config, messages, num_images=1)
        
        print(f"\nProcessing Crop {idx} (BBox: {bbox})...")
        start_time = time.time()
        
        response = generate(
            model, 
            processor, 
            prompt=prompt, 
            image=[crop_path], 
            verbose=False,
            max_tokens=512,
            temperature=0.1,             
            repetition_penalty=1.05      
        )
        
        latency = time.time() - start_time
        total_vlm_latency += latency
        
        print(f"⏱️ Generation Time: {latency:.2f}s")
        print("VLM Output:")
        print(response.text)
        
        final_results.append({
            "person_idx": idx,
            "bbox_coords": bbox,
            "yolo_confidence": conf,
            "vlm_raw_output": response.text.strip()
        })
        
    print("\n" + "=" * 50)
    print(f"Total VLM Latency for {len(cropped_info)} crops: {total_vlm_latency:.2f} seconds")
    print(f"Average latency per crop: {total_vlm_latency/len(cropped_info):.2f} seconds")

if __name__ == "__main__":
    setup_directories()
    try:
        cropped_info = detect_and_crop()
        describe_crops(cropped_info)
    except Exception as e:
        print(f"❌ Error: {e}")
