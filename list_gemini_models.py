#!/usr/bin/env python3
"""List available Gemini models"""

import os
import google.generativeai as genai

# Set API key
os.environ["GOOGLE_GEMINI_API_KEY"] = os.getenv("GOOGLE_GEMINI_API_KEY", "YOUR_API_KEY_HERE")

print("=" * 60)
print("Available Gemini Models")
print("=" * 60)

try:
    genai.configure(api_key=os.environ["GOOGLE_GEMINI_API_KEY"])

    # List available models (returns iterator)
    models = list(genai.list_models())

    print(f"\nFound {len(models)} models:\n")

    # Filter for generateContent capable models
    generate_models = [m for m in models if "generateContent" in m.supported_generation_methods]

    for model in generate_models[:20]:  # Show first 20
        print(f"  - {model.name}")
        if model.display_name:
            print(f"    Display: {model.display_name}")

    print("\n" + "=" * 60)
    print("Recommended models for MCP server:")
    print("=" * 60)

    # Find best models
    gemini_25 = [m for m in generate_models if "2.5" in m.name and "pro" in m.name]
    gemini_20 = [m for m in generate_models if "2.0" in m.name and ("flash" in m.name.lower() or "exp" in m.name.lower())]
    gemini_15 = [m for m in generate_models if "1.5" in m.name and "pro" in m.name]

    if gemini_25:
        print(f"\nGemini 2.5 Pro: {gemini_25[0].name}")

    if gemini_20:
        print(f"Gemini 2.0 Flash: {gemini_20[0].name}")

    if gemini_15:
        print(f"Gemini 1.5 Pro: {gemini_15[0].name}")

except Exception as e:
    print(f"\nError: {e}")
    print("\nPossible issues:")
    print("1. API key invalid or quota exceeded")
    print("2. Network connectivity issue")
    print("3. API endpoint changed")
