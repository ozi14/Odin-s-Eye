"""
04_yolo_qwen_padded.py
Detects persons using a custom YOLOv11 model and extracts crops WITH 50px spatial padding.
It draws a red bounding box around the focus person inside the padded crop so the VLM
knows exactly who to describe while still seeing the surrounding occlusion context.
"""

import os
import time
import json
import cv2
from ultralytics import YOLO
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_IMAGE_PATH = os.path.join(BASE_DIR, "datasets", "wildtrack", "images", "val", "cam1_00001850.jpg")
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "models", "yolo_wildtrack_weights", "best.pt")
VLM_MODEL_ID = os.path.join(BASE_DIR, "vlm", "model")

# Output for padded crops
CROPS_DIR = os.path.join(BASE_DIR, "vlm", "padded_crops")

def setup_directories():
    if not os.path.exists(CROPS_DIR):
        os.makedirs(CROPS_DIR)
        print(f"Created directory for padded crops: {CROPS_DIR}")

def detect_and_crop(padding=50):
    if not os.path.exists(TEST_IMAGE_PATH):
        raise FileNotFoundError(f"Test image not found at {TEST_IMAGE_PATH}")

    print("=" * 50)
    print(f"Step 1: Running YOLO Detector & Extracting Padded Crops (Pad={padding}px)")
    print("=" * 50)
    
    detector = YOLO(YOLO_MODEL_PATH)
    results = detector(TEST_IMAGE_PATH, conf=0.5)
    
    boxes = results[0].boxes
    image = cv2.imread(TEST_IMAGE_PATH)
    H, W = image.shape[:2]
    
    cropped_paths = []
    
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        
        # Calculate padded borders
        px1 = max(0, x1 - padding)
        py1 = max(0, y1 - padding)
        px2 = min(W, x2 + padding)
        py2 = min(H, y2 + padding)
        
        # Crop the padded area
        crop_img = image[py1:py2, px1:px2].copy()
        
        # Draw a red bounding box around the target person so the VLM knows who to focus on
        # Calculate the person's coordinates relative to the crop
        bx1 = x1 - px1
        by1 = y1 - py1
        bx2 = x2 - px1
        by2 = y2 - py1
        
        cv2.rectangle(crop_img, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
        
        crop_path = os.path.join(CROPS_DIR, f"padded_crop_{idx:03d}.jpg")
        cv2.imwrite(crop_path, crop_img)
        cropped_paths.append((idx, crop_path, (x1, y1, x2, y2), conf))
        print(f"Saved padded crop {idx} to {crop_path}")

    return cropped_paths

def describe_crops(cropped_info):
    if not cropped_info:
        return

    print("\n" + "=" * 50)
    print(f"Step 2: Loading VLM ({VLM_MODEL_ID}) via MLX")
    print("=" * 50)
    
    model, processor = load(VLM_MODEL_ID)
    config = load_config(VLM_MODEL_ID)

    system_prompt = (
        "You are an automated visual analysis component. Return ONLY a valid JSON object. "
        "No conversational text, no markdown backticks, no explanations.\n"
        "'Occlusion' means the person is blocked from view by objects or other people. "
        "Occlusion levels must be: '0' (fully visible in the crop), '1' (if physically blocked in any way-- heavily or lightly by objects/people in the padding area)."
    )

    print("\n" + "=" * 50)
    print("Step 3: Running VLM Inference on Padded Crops")
    print("=" * 50)

    total_vlm_latency = 0

    for idx, crop_path, bbox, conf in cropped_info:
        x1, y1, x2, y2 = bbox
        user_prompt = (
            f"Examine this expanded image crop. The primary subject you must describe is highlighted inside the red bounding box.\n"
            f"Look around the red box to see if the subject is occluded by anything in the surrounding context.\n"
            f"They were detected in the original camera frame at pixel coordinates: X1={x1}, Y1={y1}, X2={x2}, Y2={y2}.\n"
            "Provide a structured description of the person IN THE RED BOX:\n"
            "1. Appearance (clothing colors, gender, build, accessories)\n"
            "2. Action (e.g., standing, walking, sitting)\n"
            "3. Direction of movement (e.g., moving left, moving right, moving away, stationary)\n"
            "4. Occlusion (Output '0' or '1' based on the system definition)\n\n"
            "Output a single JSON object matching this exact format:\n"
            "  {\n"
            "    \"appearance\": \"...\",\n"
            "    \"action\": \"...\",\n"
            "    \"gender\": \"...\",\n"
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
        
        print(f"\nProcessing Padded Crop {idx} (Original BBox: {bbox})...")
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
        
    print("\n" + "=" * 50)
    print(f"Total VLM Latency for {len(cropped_info)} crops: {total_vlm_latency:.2f} seconds")
    print(f"Average latency per crop: {total_vlm_latency/len(cropped_info):.2f} seconds")

if __name__ == "__main__":
    setup_directories()
    try:
        cropped_info = detect_and_crop(padding=50) # 50px spatial padding
        describe_crops(cropped_info)
    except Exception as e:
        print(f"❌ Error: {e}")
