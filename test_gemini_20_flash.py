#!/usr/bin/env python3
"""Test with Gemini 2.0 Flash (correct model name)"""

import os
import sys
import asyncio

# Set API key
os.environ["GOOGLE_GEMINI_API_KEY"] = os.getenv("GOOGLE_GEMINI_API_KEY", "YOUR_API_KEY_HERE")

print("=" * 60)
print("TEST: Gemini 2.0 Flash")
print("=" * 60)

try:
    import google.generativeai as genai

    genai.configure(api_key=os.environ["GOOGLE_GEMINI_API_KEY"])

    # Use correct model name (without 'models/' prefix)
    model = genai.GenerativeModel("gemini-2.0-flash")

    print("\n[1] Testing connection with gemini-2.0-flash...")
    response = model.generate_content("Say 'OK' if you receive this")
    print(f"    OK - Response: {response.text[:100]}")

    print("\n[2] Testing adversarial review...")
    test_code = """def process_payment(user_id, amount):
    # No input validation!
    query = f"UPDATE accounts SET balance = balance - {amount} WHERE user_id = {user_id}"
    db.execute(query)
    return True
"""

    prompt = f"""Review this code for security issues. Return JSON:
{{
  "findings": [
    {{"severity": "HIGH|MEDIUM|LOW", "category": "Security|Code Quality", "issue": "description", "how_to_fix": "fix"}}
  ]
}}

Code:
```
{test_code}
```"""

    response = model.generate_content(prompt)
    print(f"    OK - Review completed")
    print(f"    Response:\n{response.text[:500]}...")

    print("\n[3] Testing with MCP server class...")
    from gemini_mcp_server import GeminiReviewer

    reviewer = GeminiReviewer(model="gemini-2.0-flash")

    async def test_review():
        result = await reviewer.adversarial_review(
            content=test_code,
            content_type="code",
            also_consider=["Security", "SQL Injection"]
        )
        return result

    result = asyncio.run(test_review())

    if "findings" in result:
        findings = result["findings"]
        print(f"    OK - Found {len(findings)} findings")

        print("\n    Findings:")
        for i, finding in enumerate(findings, 1):
            severity = finding.get("severity", "MEDIUM")
            category = finding.get("category", "General")
            desc = finding.get("description", finding.get("issue", "No description"))
            print(f"    {i}. [{severity}] {category}: {desc[:80]}...")

            if i >= 3:  # Show first 3
                print("    ...")
                break
    else:
        print(f"    Response: {result}")

    print("\n" + "=" * 60)
    print("SUCCESS!")
    print("=" * 60)
    print("Gemini 2.0 Flash works with MCP server!")
    print("\nRecommended configuration:")
    print("  - Update .mcp.json to use: gemini-2.0-flash")
    print("  - Or update MCP server default model")

except Exception as e:
    print(f"\nFAILED: {e}")
    import traceback
    traceback.print_exc()
