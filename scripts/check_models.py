# Utility script om te verifieren welke Google Gemini modellen beschikbaar zijn voor de geconfigureerde API key.

import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

print("🔍 Beschikbare modellen voor jouw API-key:")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"--> Naam: {m.name}")
except Exception as e:
    print(f"❌ Fout bij het ophalen van modellen: {e}")