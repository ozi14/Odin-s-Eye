"""
03_yolo_vlm_whole_frame.py
Detects persons using YOLOv11, draws labeled bounding boxes directly on the frame,
and passes the single annotated image to the Qwen VLM to generate a combined JSON 
array for all detected people.
"""

import os
import time
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
ANNOTATED_IMAGE_PATH = os.path.join(BASE_DIR, "vlm", "annotated_frame.jpg")

def detect_and_annotate():
    if not os.path.exists(TEST_IMAGE_PATH):
        raise FileNotFoundError(f"Test image not found at {TEST_IMAGE_PATH}")

    print("=" * 50)
    print("Step 1: Running YOLO & Annotating Frame")
    print("=" * 50)
    
    detector = YOLO(YOLO_MODEL_PATH)
    
    start_time = time.time()
    results = detector(TEST_IMAGE_PATH, conf=0.5)
    print(f"⏱️ YOLO Inference Latency: {time.time() - start_time:.4f} seconds")

    boxes = results[0].boxes
    image = cv2.imread(TEST_IMAGE_PATH)
    
    if image is None:
        raise ValueError("Failed to load image with CV2")
        
    detected_count = len(boxes)
    print(f"Detected {detected_count} persons. Drawing boxes...")
    
    for idx, box in enumerate(boxes):
        # 1-indexed ID for VLM prompt clarity
        person_id = idx + 1 
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        
        # Draw bounding box (Bright Green or Red for VLM visibility)
        color = (0, 0, 255) # Red in BGR
        thickness = 3
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        # Draw Label ID with a background for contrast
        label = f"[{person_id}]"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        text_thickness = 2
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
        
        # Draw dark background rectangle for text
        cv2.rectangle(image, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 0, 0), -1)
        
        # Put white text
        cv2.putText(image, label, (x1, y1 - 5), font, font_scale, (255, 255, 255), text_thickness)
        
    cv2.imwrite(ANNOTATED_IMAGE_PATH, image)
    print(f"Annotated frame saved to {ANNOTATED_IMAGE_PATH}")
    
    return detected_count

def describe_annotated_frame(detected_count):
    if detected_count == 0:
        print("No persons detected to describe.")
        return

    print("\n" + "=" * 50)
    print("Step 2: Loading VLM and Parsing Annotated Frame")
    print("=" * 50)
    
    model, processor = load(VLM_MODEL_ID)
    config = load_config(VLM_MODEL_ID)

    system_prompt = (
        "You are an automated visual analysis component. Return ONLY a valid JSON object. "
        "No conversational text, no markdown backticks, no explanations.\n"
        "CRITICAL DEFINITION: 'Occlusion' means the person is blocked from view by objects (poles, plants, containers) or by other bounding boxes (other people standing in front of them). "
        "Occlusion levels must be: 'none' (fully visible), 'partial' (e.g., legs or arms blocked), or 'heavy' (only head/upper torso visible)."
    )
    
    user_prompt = (
        f"Examine the image which contains {detected_count} people highlighted with red bounding boxes.\n"
        "Each valid person has a prominent ID label like [1], [2], etc., directly above their bounding box.\n"
        "Go through EVERY annotated box, find the person inside, and describe them.\n\n"
        "For each detected person, provide:\n"
        "1. Appearance (clothing colors, presence of backpack/bag, gender, build, accessories)\n"
        "2. Action (e.g., standing, walking, sitting)\n"
        "3. Occlusion (is another person or object blocking them? Output 'none', 'partial', or 'heavy' based on the system definition)\n\n"
        "Output a single JSON array of objects where each object matches this format:\n"
        "[\n"
        "  {\n"
        "    \"person_id\": 1,\n"
        "    \"appearance\": \"...\",\n"
        "    \"gender\": \"...\",\n"
        "    \"action\": \"...\",\n"
        "    \"direction_of_movement\": \"...\",\n"
        "    \"occlusion\": \"...\"\n"
        "  },\n"
        "  ...\n"
        "]\n\n"
        "Make sure to include all labeled people. Do NOT hallucinate people without an ID label."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": user_prompt}
        ]}
    ]

    prompt = apply_chat_template(processor, config, messages, num_images=1)
    
    print(f"\n🚀 Running Inference on {ANNOTATED_IMAGE_PATH}...")
    start_time = time.time()
    
    # We increase max_tokens because we might have multiple people being generated
    response = generate(
        model, 
        processor, 
        prompt=prompt, 
        image=[ANNOTATED_IMAGE_PATH], 
        verbose=False,
        max_tokens=4096, 
        temperature=0.1,             
        repetition_penalty=1.05      
    )
    
    latency = time.time() - start_time
    
    print("\n" + "=" * 50)
    print("VLM RAW OUTPUT:")
    print("=" * 50)
    print(response.text)
    print("=" * 50)
    print(f"⏱️ VLM Total Inference Latency: {latency:.2f} seconds for {detected_count} people")
    print(f"Average latency per person: {latency/detected_count:.2f} seconds")

if __name__ == "__main__":
    try:
        detected_count = detect_and_annotate()
        describe_annotated_frame(detected_count)
    except Exception as e:
        print(f"❌ Error: {e}")
