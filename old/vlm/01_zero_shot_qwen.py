"""
01_zero_shot_qwen.py
Tests the 4-bit quantized Qwen VLM on a sample WILDTRACK image to extract structured JSON.
Designed for local inference on Apple M4 Max using MLX.
"""

import os
import time
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Hardcoded to Camera 1, Frame 00000000 from the WILDTRACK val set you just created
TEST_IMAGE = os.path.join(BASE_DIR, "datasets", "wildtrack", "images", "val", "cam1_00001800.jpg")

# --- Model ID ---

MODEL_ID = os.path.join(BASE_DIR, "vlm", "model")

def test_zero_shot_narration():
    if not os.path.exists(TEST_IMAGE):
        print(f"❌ Error: Test image not found at {TEST_IMAGE}")
        print("Please ensure you have run the WILDTRACK preparation script.")
        return

    print("=" * 50)
    print(f"Loading {MODEL_ID} via MLX...")
    print("This might take a moment to download weights if it's the first run.")
    print("=" * 50)
    
    model, processor = load(MODEL_ID)
    config = load_config(MODEL_ID)

    # 1. Strict JSON Prompt Design
    system_prompt = (
        "You are an automated visual analysis component. Return ONLY a valid JSON array. "
        "No conversational text, no markdown backticks, no explanations."
    )
    
    user_prompt = (
        "Examine the image and list all clearly visible people. For each person, provide:\n"
        "1. Appearance (clothing colors, gender, build, accessories)\n"
        "2. Action (e.g., standing, walking)\n"
        "3. Position (spatial location)\n\n"
        "CRITICAL INSTRUCTION: Only describe actual people. Do NOT invent or repeat individuals. "
        "When finished with all visible people, immediately close the array with ] and STOP generating.\n\n"
        "Output an array of JSON objects matching this exact format:\n"
        "[\n"
        "  {\n"
        "    \"person_idx\": 1,\n"
        "    \"appearance\": \"...\",\n"
        "    \"action\": \"...\",\n"
        "    \"position\": \"...\"\n"
        "  }\n"
        "]"
    )

    # 2. Format Chat Template
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": user_prompt}
        ]}
    ]

    print(f"Applying chat template and injecting image: {os.path.basename(TEST_IMAGE)}...")
    prompt = apply_chat_template(processor, config, messages, num_images=1)

    # 3. Model Inference (Temperature 0 for structured data)
    print(f"\n🚀 Starting Inference on M4 Max (MPS)...")
    start_time = time.time()
    
    response = generate(
        model, 
        processor, 
        prompt=prompt, 
        image=[TEST_IMAGE], 
        verbose=False,
        max_tokens=2048,
        temperature=0.1,             # Tiny variance to break loops
        repetition_penalty=1.05      # Penalize exact repetition sequences
    )
    
    latency = time.time() - start_time

    # 4. Results
    print("\n" + "=" * 50)
    print("VLM RAW OUTPUT:")
    print("=" * 50)
    print(response.text)
    print("=" * 50)
    print(f"⏱️ Inference Latency: {latency:.2f} seconds")

if __name__ == "__main__":
    test_zero_shot_narration()
