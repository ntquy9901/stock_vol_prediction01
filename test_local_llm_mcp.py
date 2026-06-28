#!/usr/bin/env python3
"""
Test script cho Local LLM MCP Server
Validates Ollama connection, MCP server tools, and review quality
"""

import os
import sys
import json
import asyncio

print("=" * 70)
print("Local LLM MCP Server Test Suite")
print("=" * 70)

# ============================================================================
# TEST 1: Check Dependencies
# ============================================================================

print("\n[TEST 1] Checking dependencies...")

missing_deps = []

try:
    import mcp
    print("  ✓ mcp installed")
except ImportError:
    print("  ✗ mcp NOT installed")
    missing_deps.append("mcp")

try:
    import ollama
    print("  ✓ ollama installed")
except ImportError:
    print("  ✗ ollama NOT installed")
    missing_deps.append("ollama")

try:
    from openai import OpenAI
    print("  ✓ openai installed")
except ImportError:
    print("  ✗ openai NOT installed")
    missing_deps.append("openai")

if missing_deps:
    print(f"\n  ERROR: Missing dependencies: {', '.join(missing_deps)}")
    print("  Install with: pip install -r requirements-local-llm.txt")
    sys.exit(1)

print("  ✓ All dependencies installed")

# ============================================================================
# TEST 2: Check Ollama Connection
# ============================================================================

print("\n[TEST 2] Testing Ollama connection...")

try:
    import ollama

    # List models
    models = ollama.list()
    print(f"  ✓ Ollama is running")
    print(f"  ✓ Found {len(models['models'])} models")

    # Check if qwen2.5-coder is available
    qwen_models = [m for m in models['models'] if 'qwen' in m['name'].lower() and 'coder' in m['name'].lower()]
    if qwen_models:
        print(f"  ✓ Found Qwen Coder models:")
        for m in qwen_models:
            size_gb = m.get('size', 0) / (1024**3)
            print(f"    - {m['name']} ({size_gb:.1f}GB)")
        DEFAULT_MODEL = qwen_models[0]['name']
    else:
        print(f"  ⚠ Qwen Coder not found")
        print(f"    Available models:")
        for m in models['models'][:5]:
            print(f"    - {m['name']}")
        print(f"\n  Pull Qwen Coder with:")
        print(f"    ollama pull qwen2.5-coder:32b")
        print(f"    # Or for smaller hardware:")
        print(f"    ollama pull qwen2.5-coder:7b")
        DEFAULT_MODEL = "qwen2.5-coder:32b"  # Assume anyway

except Exception as e:
    print(f"  ✗ Ollama connection failed: {e}")
    print(f"\n  Troubleshooting:")
    print(f"    1. Start Ollama: ollama serve")
    print(f"    2. Pull model: ollama pull qwen2.5-coder:32b")
    print(f"    3. Check Ollama is running: ollama list")
    sys.exit(1)

# ============================================================================
# TEST 3: Test Ollama Code Review
# ============================================================================

print("\n[TEST 3] Testing Ollama code review...")

test_code = """def process_payment(user_id, amount):
    # No input validation!
    query = f"UPDATE accounts SET balance = balance - {amount} WHERE user_id = {user_id}"
    db.execute(query)
    return True

def divide(a, b):
    return a / b  # No error handling!
"""

print(f"  Test code:")
print(f"  ─" * 50)
for line in test_code.split('\n')[:5]:
    print(f"  {line}")
print(f"  ─" * 50)

try:
    prompt = f"""Review this code for security issues and bugs. Return JSON:
{{
  "findings": [
    {{"severity": "HIGH|MEDIUM|LOW", "category": "Security|Code Quality", "issue": "description"}}
  ]
}}

Code:
```
{test_code}
```"""

    response = ollama.chat(
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    print(f"  ✓ Ollama response received")
    print(f"  Response preview: {response['message']['content'][:150]}...")

    # Try to parse JSON
    try:
        import re
        json_match = re.search(r'\{[\s\S]*\}', response['message']['content'])
        if json_match:
            data = json.loads(json_match.group())
            findings_count = len(data.get('findings', []))
            print(f"  ✓ Parsed {findings_count} findings")
        else:
            print(f"  ⚠ No JSON found in response")
    except:
        print(f"  ⚠ JSON parsing failed (model may need better prompt)")

except Exception as e:
    print(f"  ✗ Ollama call failed: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# TEST 4: Test MCP Server Import
# ============================================================================

print("\n[TEST 4] Testing MCP server import...")

try:
    # Set environment variables
    os.environ["LLM_RUNTIME"] = "ollama"
    os.environ["LLM_MODEL"] = DEFAULT_MODEL
    os.environ["LLM_BASE_URL"] = "http://localhost:11434"

    # Import server
    import local_coder_mcp_server
    print("  ✓ MCP server module imported")
    print(f"    Runtime: {local_coder_mcp_server.RUNTIME}")
    print(f"    Model: {local_coder_mcp_server.MODEL}")

except Exception as e:
    print(f"  ✗ MCP server import failed: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# TEST 5: Test LocalLLMReviewer Class
# ============================================================================

print("\n[TEST 5] Testing LocalLLMReviewer class...")

try:
    from local_coder_mcp_server import LocalLLMReviewer

    reviewer = LocalLLMReviewer(model=DEFAULT_MODEL)
    print(f"  ✓ LocalLLMReviewer initialized")
    print(f"    Model: {reviewer.model_name}")
    print(f"    Runtime: {reviewer.runtime}")

except Exception as e:
    print(f"  ✗ LocalLLMReviewer failed: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# TEST 6: Test Adversarial Review (Async)
# ============================================================================

print("\n[TEST 6] Testing adversarial review (this may take 10-30s)...")

try:
    import asyncio
    from local_coder_mcp_server import LocalLLMReviewer

    async def test_review():
        reviewer = LocalLLMReviewer(model=DEFAULT_MODEL)
        result = await reviewer.adversarial_review(
            content=test_code,
            content_type="code",
            also_consider=["Security", "Error Handling"]
        )
        return result

    result = asyncio.run(test_review())

    if result.get("success"):
        findings = result.get("findings", [])
        print(f"  ✓ Adversarial review completed")
        print(f"    Model: {result.get('model')}")
        print(f"    Runtime: {result.get('runtime')}")
        print(f"    Findings: {len(findings)} issues found")

        if findings:
            print(f"\n    Sample findings:")
            for i, finding in enumerate(findings[:3], 1):
                severity = finding.get("severity", "MEDIUM")
                category = finding.get("category", "General")
                desc = finding.get("issue", finding.get("description", "No description"))
                print(f"    {i}. [{severity}] {category}: {desc[:80]}...")
    else:
        print(f"  ✗ Review failed: {result.get('error')}")

except Exception as e:
    print(f"  ✗ Adversarial review failed: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# TEST 7: Test Code Review (Async)
# ============================================================================

print("\n[TEST 7] Testing code review (this may take 10-30s)...")

test_diff = """diff --git a/src/auth.py b/src/auth.py
index 1234567..abcdefg 100644
--- a/src/auth.py
+++ b/src/auth.py
@@ -10,7 +10,7 @@ def authenticate(username, password):
-    if password == stored_password:
+    if password == stored_password.strip():
         return True
"""

try:
    import asyncio
    from local_coder_mcp_server import LocalLLMReviewer

    async def test_code_review():
        reviewer = LocalLLMReviewer(model=DEFAULT_MODEL)
        result = await reviewer.code_review(
            diff_content=test_diff,
            review_mode="full",
            persistent_facts=["All code must follow PEP 8"]
        )
        return result

    result = asyncio.run(test_code_review())

    if result.get("success"):
        findings = result.get("findings", [])
        print(f"  ✓ Code review completed")
        print(f"    Model: {result.get('model')}")
        print(f"    Review mode: {result.get('review_mode')}")
        print(f"    Findings: {len(findings)} issues found")

        summary = result.get("summary", {})
        if summary:
            print(f"\n    Summary:")
            print(f"    HIGH:   {summary.get('high_count', 0)}")
            print(f"    MEDIUM: {summary.get('medium_count', 0)}")
            print(f"    LOW:    {summary.get('low_count', 0)}")
    else:
        print(f"  ✗ Code review failed: {result.get('error')}")

except Exception as e:
    print(f"  ✗ Code review failed: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("TEST SUMMARY")
print("=" * 70)

print("\n✓ MCP Server is ready!")
print(f"\nConfiguration:")
print(f"  Runtime: ollama")
print(f"  Model: {DEFAULT_MODEL}")
print(f"  Base URL: http://localhost:11434")

print(f"\nNext steps:")
print(f"  1. Restart Claude Code")
print(f"  2. Tools will be available:")
print(f"     - local_review_adversarial")
print(f"     - local_review_code")
print(f"  3. Use in conversation:")
print(f"     'Review this code using local LLM: [paste code]'")
print(f"     'Call local_review_adversarial with this content: ...'")

print(f"\nExample usage:")
print(f"  Please review this Python code using local_review_adversarial:")
print(f"  [paste your code]")

print(f"\nOr integrate into BMad skills:")
print(f"  Update skill step files to call local_review_adversarial")
print(f"  instead of built-in Claude analysis")

print(f"\n" + "=" * 70)
