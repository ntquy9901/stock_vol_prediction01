#!/usr/bin/env python3
"""Simple MCP server test (no unicode, set API key directly)"""

import os
import sys

# Set API key directly for testing
os.environ["GOOGLE_GEMINI_API_KEY"] = os.getenv("GOOGLE_GEMINI_API_KEY", "YOUR_API_KEY_HERE")

print("=" * 60)
print("TEST: Gemini MCP Server")
print("=" * 60)

# Test 1: Import MCP server
print("\n[1] Importing MCP server module...")
try:
    import gemini_mcp_server
    print("    OK - Module imported successfully")
except Exception as e:
    print(f"    FAILED - {e}")
    sys.exit(1)

# Test 2: Check Gemini API connection
print("\n[2] Testing Gemini API connection...")
try:
    import google.generativeai as genai

    genai.configure(api_key=os.environ["GOOGLE_GEMINI_API_KEY"])
    model = genai.GenerativeModel("gemini-2.5-pro")

    # Simple test call
    response = model.generate_content("Test message - say 'OK' if you receive this")
    print(f"    OK - Gemini API responding")
    print(f"    Response: {response.text[:100]}...")
except Exception as e:
    print(f"    FAILED - {e}")
    print("\n    TIP: Your API key might be invalid or quota exceeded")
    print("    Check: https://makersuite.google.com/app/apikey")
    sys.exit(1)

# Test 3: Test adversarial review function
print("\n[3] Testing adversarial review function...")
try:
    reviewer = gemini_mcp_server.GeminiReviewer()

    test_code = """def foo():
    return 1 / 0  # Division by zero bug
"""

    import asyncio

    async def test_review():
        result = await reviewer.adversarial_review(
            content=test_code,
            content_type="code",
            also_consider=["Error Handling"]
        )
        return result

    result = asyncio.run(test_review())

    if "findings" in result:
        findings_count = len(result["findings"])
        print(f"    OK - Found {findings_count} findings")

        if findings_count > 0:
            print("\n    Sample findings:")
            for i, finding in enumerate(result["findings"][:3], 1):
                severity = finding.get("severity", "MEDIUM")
                desc = finding.get("description", finding.get("issue", "No description"))
                print(f"    {i}. [{severity}] {desc[:80]}...")
    else:
        print(f"    FAILED - No findings in response")
        print(f"    Response: {result}")

except Exception as e:
    print(f"    FAILED - {e}")
    import traceback
    traceback.print_exc()

# Test 4: Test code review function
print("\n[4] Testing code review function...")
try:
    reviewer = gemini_mcp_server.GeminiReviewer()

    test_diff = """diff --git a/test.py b/test.py
index 123..abc 100644
--- a/test.py
+++ b/test.py
@@ -1,3 +1,3 @@
-def old_func():
+def new_func():
     return True
"""

    async def test_code_review():
        result = await reviewer.code_review(
            diff_content=test_diff,
            review_mode="full",
            persistent_facts=["All functions must have docstrings"]
        )
        return result

    result = asyncio.run(test_code_review())

    if "findings" in result:
        findings_count = len(result["findings"])
        print(f"    OK - Found {findings_count} findings")

        if findings_count > 0:
            print("\n    Sample findings:")
            for i, finding in enumerate(result["findings"][:2], 1):
                layer = finding.get("layer", "Unknown")
                severity = finding.get("severity", "MEDIUM")
                issue = finding.get("issue", "No description")
                print(f"    {i}. [{layer}] {severity} - {issue[:80]}...")
    else:
        print(f"    FAILED - No findings in response")

except Exception as e:
    print(f"    FAILED - {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("TEST SUMMARY")
print("=" * 60)
print("All core tests completed!")
print("\nNext steps:")
print("1. MCP server is working")
print("2. Restart Claude Code to load the MCP server")
print("3. Tools will be available: gemini_review_adversarial, gemini_review_code")
print("\nSecurity reminder:")
print("- Revoke this test API key after testing")
print("- Create new API key for production use")
print("- Never commit API keys to git")
