"""
AI Proxy Generator - Creates a neutral/ungraded version of reference images.

Uses OpenRouter API to bypass geo-restrictions on image generation.
"""

import os
from dotenv import load_dotenv
from PIL import Image
import typing
import io
import base64
import requests
import json
import time
import re

# Load environment variables
load_dotenv()

# OpenRouter configuration 
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = "google/gemini-3-pro-preview"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class GeminiProxyGenerator:
    """
    Handles the generation of a 'Neutral Proxy' image using OpenRouter API.
    Bypasses Google's geo-restrictions by proxying the request.
    """

    def __init__(self, api_key: typing.Optional[str] = None):
        self.api_key = api_key or OPENROUTER_API_KEY
        self.model = OPENROUTER_MODEL
        
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found.")

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:5173",
            "X-Title": "HCGE Proxy Generator"
        }
        
    def generate_proxy(self, image_path: str, output_path: str) -> bool:
        """
        Generate a neutral proxy from the input reference image.

        Args:
            image_path (str): Path to the REFERENCE image to neutralize.
            output_path (str): Path to save the generated proxy.

        Returns:
            bool: True if successful, False otherwise.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        print(f"[PROXY] Processing REFERENCE image: {image_path}")
        
        try:
            # Load and prepare reference image
            img = Image.open(image_path)
            
            # Helper to convert image to base64
            def image_to_base64(pil_img):
                if pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')
                buffer = io.BytesIO()
                pil_img.save(buffer, format='JPEG', quality=95)
                return base64.b64encode(buffer.getvalue()).decode('utf-8')

            ref_b64 = image_to_base64(img)
            
            prompt = (
                "You are a master digital imaging technician (DIT). I am providing a heavily color-graded reference image. "
                "Your task is to generate a RAW, UNGRADED 'DIRECT-FROM-SENSOR' version of this exact frame. "
                "AGGRESSIVELY STRIP: "
                "1. All stylistic color tints (e.g., intense green or orange-teal casts). "
                "2. All contrast enhancement, S-curves, and black-level crushing. "
                "3. All LUT-based transformations and creative color grades. "
                "4. All saturation boosts; make it look almost monochrome/gray but with true underlying colors. "
                "\n"
                "The output MUST be: "
                "- Extremely flat and desaturated (like 'LOG' footage or raw sensor data). "
                "- Perfectly neutral in white balance. "
                "- Standard dynamic range with zero stylistic clipping. "
                "- IDENTICAL in composition, subjects, and framing. "
                "\n"
                "Output the modified image."
            )

            # Build OpenRouter request with modalities for image output
            payload = {
                "model": self.model,
                "modalities": ["image", "text"],  # CRITICAL: Required for image generation
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{ref_b64}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 4096,  # Increase for image content
                "temperature": 0.3
            }

            # Retry logic
            max_retries = 3
            response = None
            for attempt in range(max_retries):
                try:
                    print(f"[PROXY] Sending to OpenRouter ({self.model}) - Attempt {attempt+1}/{max_retries}...")
                    response = requests.post(
                        OPENROUTER_URL,
                        headers=self.headers,
                        json=payload,
                        timeout=120
                    )
                    
                    if response.status_code == 429:
                        print(f"[PROXY] Rate limited (429). Waiting 5s...")
                        time.sleep(5)
                        continue
                        
                    response.raise_for_status()
                    break 
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    print(f"[PROXY] Request failed: {e}. Retrying...")
                    time.sleep(2)

            if not response:
                return False

            result = response.json()
            print(f"[PROXY] Full response keys: {result.keys()}")
            
            if 'choices' in result and len(result['choices']) > 0:
                message = result['choices'][0]['message']
                print(f"[PROXY] Message object keys: {message.keys()}")
                print(f"[PROXY] Message object: {json.dumps(message, indent=2)[:500]}...") # Print first 500 chars
                # Check for 'images' field (specific to OpenRouter multimodal output)
                if 'images' in message and message['images']:
                    print(f"[PROXY] Found {len(message['images'])} images in 'images' field")
                    image_data = message['images'][0]
                    print(f"[PROXY] Image data type: {type(image_data)}")
                    
                    if isinstance(image_data, dict):
                         # Handle dict format. Keys found: ['type', 'image_url', 'index']
                         print(f"[PROXY] Image keys: {image_data.keys()}")
                         
                         if 'image_url' in image_data:
                             # OpenAI-style structure
                             img_val = image_data['image_url']
                             if isinstance(img_val, dict) and 'url' in img_val:
                                 image_data = img_val['url']
                             elif isinstance(img_val, str):
                                 image_data = img_val
                         elif 'url' in image_data:
                             image_data = image_data['url']
                         elif 'b64_json' in image_data:
                             image_data = image_data['b64_json']
                         elif 'base64' in image_data:
                             image_data = image_data['base64']

                    if isinstance(image_data, str):
                        # It might be a base64 string or a URL
                        if image_data.startswith('http'):
                             try:
                                 img_resp = requests.get(image_data)
                                 if img_resp.status_code == 200:
                                     with open(output_path, 'wb') as f:
                                         f.write(img_resp.content)
                                     print(f"[PROXY] Downloaded image from URL in 'images' field")
                                     return True
                             except Exception as e:
                                 print(f"[PROXY] Failed to download image from URL: {e}")
                        
                        else:
                             # Assume base64 or Data URL
                             try:
                                 b64_str = image_data
                                 if 'base64,' in b64_str:
                                     b64_str = b64_str.split('base64,')[1]
                                     
                                 # strict=False allows missing padding
                                 img_bytes = base64.b64decode(b64_str)
                                 with open(output_path, 'wb') as f:
                                     f.write(img_bytes)
                                 print(f"[PROXY] Decoded base64 from 'images' field")
                                 return True
                             except Exception as e:
                                 print(f"[PROXY] Failed to decode base64 from 'images': {e}")
                
                content = message.get('content', '')
                
                # OpenRouter returns images in content array with type "image_url"
                # Or sometimes as a direct base64 data URL in text
                if isinstance(content, list):
                    # Content is an array of parts
                    for part in content:
                        if part.get('type') == 'image_url':
                            image_url = part.get('image_url', {}).get('url', '')
                            if image_url.startswith('data:image'):
                                # Extract base64 from data URL
                                b64_data = image_url.split(',', 1)[1]
                                img_data = base64.b64decode(b64_data)
                                with open(output_path, 'wb') as f:
                                    f.write(img_data)
                                print(f"[PROXY] Extracted image from content array, saved to {output_path}")
                                return True
                elif isinstance(content, str):
                    print(f"[PROXY] Response content length: {len(content)}")
                    
                    if "I cannot" in content or "I can't" in content:
                        print(f"[PROXY] Model refused: {content[:100]}...")
                        return False
                    
                    # Check for data URL pattern (data:image/png;base64,...)
                    data_url_match = re.search(r'data:image/[a-z]+;base64,([A-Za-z0-9+/=]+)', content)
                    if data_url_match:
                        b64_data = data_url_match.group(1)
                        img_data = base64.b64decode(b64_data)
                        with open(output_path, 'wb') as f:
                            f.write(img_data)
                        print(f"[PROXY] Extracted data URL image, saved to {output_path}")
                        return True
                        
                    # Try to find markdown image URL
                    url_match = re.search(r'\!\[.*?\]\((.*?)\)', content)
                    if url_match:
                        image_url = url_match.group(1)
                        print(f"[PROXY] Found markdown image URL: {image_url}")
                        if image_url.startswith('data:'):
                            b64_data = image_url.split(',', 1)[1]
                            img_data = base64.b64decode(b64_data)
                        else:
                            img_resp = requests.get(image_url)
                            if img_resp.status_code == 200:
                                img_data = img_resp.content
                            else:
                                print(f"[PROXY] Failed to download image from URL")
                                return False
                        with open(output_path, 'wb') as f:
                            f.write(img_data)
                        print(f"[PROXY] Proxy saved to {output_path}")
                        return True
                    
                    # Try raw base64 decode if content is long enough
                    clean_content = content.replace("```", "").strip()
                    if len(clean_content) > 1000:
                        try:
                            img_data = base64.b64decode(clean_content)
                            with open(output_path, 'wb') as f:
                                f.write(img_data)
                            print(f"[PROXY] Decoded raw base64, saved to {output_path}")
                            return True
                        except:
                            pass
                    
                    print("[PROXY] No valid image found in response content.")
                    print(f"[PROXY] Content preview: {content[:300]}...")
                    return False

            else:
                print(f"[PROXY] Unexpected response format: {result}")
                return False

        except Exception as e:
            print(f"[PROXY] OpenRouter Error: {e}")
            import traceback
            traceback.print_exc()
            return False
