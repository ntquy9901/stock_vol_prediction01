#!/usr/bin/env python3
"""Test with Gemini 2.0 Flash (better free tier)"""

import os
import sys
import asyncio

# Set API key
os.environ["GOOGLE_GEMINI_API_KEY"] = os.getenv("GOOGLE_GEMINI_API_KEY", "YOUR_API_KEY_HERE")

print("=" * 60)
print("TEST: Gemini 2.0 Flash (Free Tier)")
print("=" * 60)

try:
    import google.generativeai as genai

    genai.configure(api_key=os.environ["GOOGLE_GEMINI_API_KEY"])

    # Try with Gemini 2.0 Flash (free experiment)
    model = genai.GenerativeModel("gemini-2.0-flash-exp")

    print("\n[1] Testing connection with gemini-2.0-flash-exp...")
    response = model.generate_content("Say 'OK' if you receive this")
    print(f"    OK - Response: {response.text[:100]}")

    print("\n[2] Testing adversarial review...")
    test_code = """def foo():
    return 1 / 0  # Bug
"""

    prompt = f"""Review this code and find issues:
```
{test_code}
```

Return JSON format:
{{
  "findings": [
    {{"severity": "HIGH|MEDIUM|LOW", "issue": "description"}}
  ]
}}"""

    response = model.generate_content(prompt)
    print(f"    OK - Review completed")
    print(f"    Response preview: {response.text[:200]}...")

    print("\n" + "=" * 60)
    print("SUCCESS!")
    print("=" * 60)
    print("Gemini 2.0 Flash works!")
    print("\nRecommendations:")
    print("1. Use gemini-2.0-flash-exp for testing (better free tier)")
    print("2. Upgrade to paid tier for gemini-2.5-pro if needed")
    print("3. Update MCP server to default to gemini-2.0-flash-exp")

except Exception as e:
    print(f"\nFAILED: {e}")
    print("\nTroubleshooting:")
    print("1. Check if API key is valid")
    print("2. Check quota at: https://makersuite.google.com/app/apikey")
    print("3. Wait for quota reset (usually daily for free tier)")
