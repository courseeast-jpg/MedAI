"""List available Gemini models"""
import google.generativeai as genai

API_KEY = "AQ.Ab8RN6KCZdzZbOEMh8hxpPBjHjYS5xJXiUv3ZfiSr-uUY7C3gg"
genai.configure(api_key=API_KEY)

print("Available Gemini models:")
print("=" * 50)

for model in genai.list_models():
    if 'generateContent' in model.supported_generation_methods:
        print(f"✓ {model.name}")
        print(f"  Display name: {model.display_name}")
        print()

print("\nNow testing gemini-2.5-flash...")
try:
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content("Say hello")
    print(f"✓ SUCCESS: {response.text}")
except Exception as e:
    print(f"✗ FAILED: {e}")
