"""
Test script to determine best JSON format for Gemini extraction
"""
import google.generativeai as genai
import json

# Use your API key
API_KEY = "AQ.Ab8RN6KCZdzZbOEMh8hxpPBjHjYS5xJXiUv3ZfiSr-uUY7C3gg"
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash-exp")

# Sample Russian text from the PDF
sample_text = """
РУКОВОДСТВО ПО ПИТАНИЮ 4D — ВЕРСИЯ 2
IBS · Дивертикулит · Оксалаты · Кристаллурия

Продукт IBS Div Ox Cryst
Карамель 0 0 0 0
Курица варёная 0 1 0 0
Морковь сырая 0 0 1 0
"""

# Ask Gemini for advice
prompt = """You are helping design a medical data extraction system.

I need to extract structured data from Russian medical/dietary documents and return it as JSON.

The documents contain:
- Medical conditions (IBS, diverticulitis, kidney stones)
- Food items with ratings
- Dietary recommendations

What is the SIMPLEST, most RELIABLE JSON format you can consistently generate without syntax errors?

Please provide:
1. A simple JSON schema you can reliably produce
2. An example extraction from this sample text:

""" + sample_text

response = model.generate_content(prompt)
print("=== GEMINI'S RECOMMENDATION ===")
print(response.text)
print("\n=== Testing the recommended format ===")

# Now test extraction with Gemini's own recommendation
test_prompt = """Extract medical entities from this Russian text. Translate to English.

Text:
""" + sample_text + """

Return ONLY valid JSON in the format YOU just recommended."""

test_response = model.generate_content(test_prompt)
print("\nGemini's extraction:")
print(test_response.text)

# Try to parse it
try:
    parsed = json.loads(test_response.text.strip().replace("```json", "").replace("```", ""))
    print("\n✓ SUCCESS - JSON parsed correctly!")
    print(f"Parsed data: {json.dumps(parsed, indent=2)}")
except Exception as e:
    print(f"\n✗ FAILED - JSON parse error: {e}")
