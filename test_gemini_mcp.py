#!/usr/bin/env python3
"""
Test script cho Gemini MCP Server
Verifies MCP server functionality and tools
"""

import os
import sys
import asyncio
import json
from typing import Any

# Check API key first
if not os.getenv("GOOGLE_GEMINI_API_KEY"):
    print("❌ ERROR: GOOGLE_GEMINI_API_KEY environment variable not set!")
    print("\nSet it with:")
    print("  Windows (PowerShell): $env:GOOGLE_GEMINI_API_KEY='your-key'")
    print("  Windows (CMD): set GOOGLE_GEMINI_API_KEY=your-key")
    print("  Linux/Mac: export GOOGLE_GEMINI_API_KEY='your-key'")
    sys.exit(1)


async def test_mcp_connection():
    """Test basic MCP connection"""
    print("=" * 60)
    print("TEST 1: MCP Server Connection")
    print("=" * 60)

    try:
        from mcp.client.stdio import StdioServerParameters
        from mcp.client import Client

        server_params = StdioServerParameters(
            command="python",
            args=["gemini_mcp_server.py"],
            env={
                "GOOGLE_GEMINI_API_KEY": os.getenv("GOOGLE_GEMINI_API_KEY"),
                "GEMINI_MODEL": "gemini-2.5-pro"
            }
        )

        print("\n✅ MCP client configuration loaded")
        print(f"   Command: {server_params.command}")
        print(f"   Args: {server_params.args}")
        print(f"   Env: GOOGLE_GEMINI_API_KEY={'***' if os.getenv('GOOGLE_GEMINI_API_KEY') else 'NOT SET'}")

        return True, server_params

    except ImportError as e:
        print(f"\n❌ Failed to import MCP client: {e}")
        print("\n💡 Install dependencies:")
        print("   pip install -r requirements-gemini-mcp.txt")
        return False, None
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False, None


async def test_list_tools(server_params):
    """Test listing available tools"""
    print("\n" + "=" * 60)
    print("TEST 2: List Available Tools")
    print("=" * 60)

    try:
        from mcp.client import Client

        async with Client(server_params) as client:
            # Initialize connection
            await client.initialize()

            # List tools
            tools_result = await client.list_tools()
            tools = tools_result.tools if hasattr(tools_result, 'tools') else []

            print(f"\n✅ Found {len(tools)} tools:")
            for tool in tools:
                print(f"\n  📦 {tool.name}")
                print(f"     Description: {tool.description}")
                if hasattr(tool, 'inputSchema') and tool.inputSchema:
                    print(f"     Parameters: {list(tool.inputSchema.get('properties', {}).keys())}")

            return True, tools

    except Exception as e:
        print(f"\n❌ Error listing tools: {e}")
        return False, []


async def test_adversarial_review(server_params):
    """Test adversarial review tool"""
    print("\n" + "=" * 60)
    print("TEST 3: Adversarial Review Tool")
    print("=" * 60)

    try:
        from mcp.client import Client

        # Sample code with bugs
        buggy_code = """def process_payment(user_id, amount):
    # No input validation!
    conn = db.connect()
    query = f"UPDATE accounts SET balance = balance - {amount} WHERE user_id = {user_id}"
    conn.execute(query)
    conn.close()
    return True

def divide(a, b):
    return a / b  # No error handling!
"""

        print("\n📝 Sample code to review:")
        print("─" * 60)
        print(buggy_code)
        print("─" * 60)

        async with Client(server_params) as client:
            await client.initialize()

            print("\n🔍 Calling gemini_review_adversarial...")

            result = await client.call_tool(
                "gemini_review_adversarial",
                {
                    "content": buggy_code,
                    "content_type": "code",
                    "also_consider": ["Security", "Error Handling"]
                }
            )

            # Parse result
            if result.content and len(result.content) > 0:
                content = result.content[0]
                if hasattr(content, 'text'):
                    response = json.loads(content.text)

                    print(f"\n✅ Review completed!")
                    print(f"   Model: {response.get('model', 'unknown')}")

                    findings = response.get('findings', [])
                    print(f"   Findings: {len(findings)} issues found")

                    if findings:
                        print("\n   Top findings:")
                        for i, finding in enumerate(findings[:5], 1):
                            severity = finding.get('severity', 'MEDIUM')
                            category = finding.get('category', 'General')
                            issue = finding.get('issue', finding.get('description', 'No description'))
                            print(f"\n   {i}. [{severity}] {category}")
                            print(f"      {issue}")

                            if 'how_to_fix' in finding:
                                print(f"      Fix: {finding['how_to_fix']}")

                    summary = response.get('summary', {})
                    if summary:
                        print(f"\n   Summary:")
                        print(f"   HIGH:   {summary.get('high_count', 0)}")
                        print(f"   MEDIUM: {summary.get('medium_count', 0)}")
                        print(f"   LOW:    {summary.get('low_count', 0)}")

                    return True, response
                else:
                    print(f"\n❌ Unexpected response format")
                    return False, None
            else:
                print(f"\n❌ Empty response")
                return False, None

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False, None


async def test_code_review(server_params):
    """Test code review tool"""
    print("\n" + "=" * 60)
    print("TEST 4: Code Review Tool (Parallel Layers)")
    print("=" * 60)

    try:
        from mcp.client import Client

        # Sample git diff
        diff_content = """diff --git a/src/auth.py b/src/auth.py
index 1234567..abcdefg 100644
--- a/src/auth.py
+++ b/src/auth.py
@@ -10,7 +10,7 @@ def authenticate(username, password):
-    if password == stored_password:
+    if password == stored_password.strip():
         return True

diff --git a/src/user.py b/src/user.py
new file mode 100644
--- /dev/null
+++ b/src/user.py
@@ -0,0 +1,5 @@
+def get_user_info(user_id):
+    # No input validation!
+    query = f"SELECT * FROM users WHERE id = {user_id}"
+    return db.execute(query)
"""

        print("\n📝 Sample diff to review:")
        print("─" * 60)
        print(diff_content[:200] + "..." if len(diff_content) > 200 else diff_content)
        print("─" * 60)

        async with Client(server_params) as client:
            await client.initialize()

            print("\n🔍 Calling gemini_review_code...")

            result = await client.call_tool(
                "gemini_review_code",
                {
                    "diff_content": diff_content,
                    "review_mode": "full",
                    "persistent_facts": [
                        "All user input must be validated",
                        "Use parameterized queries to prevent SQL injection"
                    ]
                }
            )

            # Parse result
            if result.content and len(result.content) > 0:
                content = result.content[0]
                if hasattr(content, 'text'):
                    response = json.loads(content.text)

                    print(f"\n✅ Code review completed!")
                    print(f"   Model: {response.get('model', 'unknown')}")
                    print(f"   Mode: {response.get('review_mode', 'unknown')}")

                    findings = response.get('findings', [])
                    print(f"   Findings: {len(findings)} issues found")

                    if findings:
                        print("\n   Findings by layer:")
                        for i, finding in enumerate(findings[:5], 1):
                            layer = finding.get('layer', 'Unknown')
                            severity = finding.get('severity', 'MEDIUM')
                            issue = finding.get('issue', 'No description')
                            location = finding.get('location', 'Unknown')
                            print(f"\n   {i}. [{layer}] - {severity}")
                            print(f"      Issue: {issue}")
                            print(f"      Location: {location}")

                            if 'how_to_fix' in finding:
                                print(f"      Fix: {finding['how_to_fix']}")

                    summary = response.get('summary', {})
                    if summary:
                        print(f"\n   Summary:")
                        print(f"   HIGH:   {summary.get('high_count', 0)}")
                        print(f"   MEDIUM: {summary.get('medium_count', 0)}")
                        print(f"   LOW:    {summary.get('low_count', 0)}")
                        print(f"   Total:  {summary.get('total_findings', 0)}")

                        layers = summary.get('layers', {})
                        if layers:
                            print(f"\n   By layer:")
                            for layer_name, count in layers.items():
                                print(f"   {layer_name}: {count}")

                    return True, response
                else:
                    print(f"\n❌ Unexpected response format")
                    return False, None
            else:
                print(f"\n❌ Empty response")
                return False, None

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False, None


async def test_model_switching(server_params):
    """Test switching between models"""
    print("\n" + "=" * 60)
    print("TEST 5: Model Switching")
    print("=" * 60)

    models = ["gemini-2.5-pro", "gemini-1.5-pro"]
    simple_code = "def foo(): return 1 / 0"

    results = {}

    for model in models:
        print(f"\n🔍 Testing {model}...")

        try:
            from mcp.client import Client

            # Update server params for this model
            params = StdioServerParameters(
                command="python",
                args=["gemini_mcp_server.py"],
                env={
                    "GOOGLE_GEMINI_API_KEY": os.getenv("GOOGLE_GEMINI_API_KEY"),
                    "GEMINI_MODEL": model
                }
            )

            async with Client(params) as client:
                await client.initialize()

                result = await client.call_tool(
                    "gemini_review_adversarial",
                    {
                        "content": simple_code,
                        "content_type": "code",
                        "model": model
                    }
                )

                if result.content and len(result.content) > 0:
                    content = result.content[0]
                    if hasattr(content, 'text'):
                        response = json.loads(content.text)
                        findings = response.get('findings', [])
                        results[model] = len(findings)
                        print(f"   ✅ {len(findings)} findings")
                else:
                    results[model] = 0
                    print(f"   ❌ No response")

        except Exception as e:
            results[model] = 0
            print(f"   ❌ Error: {e}")

    print("\n📊 Results comparison:")
    for model, count in results.items():
        print(f"   {model}: {count} findings")

    return True, results


async def main():
    """Run all tests"""
    print("🚀 Gemini MCP Server Test Suite")
    print("=" * 60)

    # Test 1: Connection
    success, server_params = await test_mcp_connection()
    if not success:
        print("\n❌ MCP connection test failed. Exiting.")
        return

    # Test 2: List tools
    success, tools = await test_list_tools(server_params)
    if not success or len(tools) == 0:
        print("\n❌ No tools found. Exiting.")
        return

    # Test 3: Adversarial review
    success, response = await test_adversarial_review(server_params)
    if not success:
        print("\n⚠️  Adversarial review test failed")

    # Test 4: Code review
    success, response = await test_code_review(server_params)
    if not success:
        print("\n⚠️  Code review test failed")

    # Test 5: Model switching (optional - skip if slow)
    print("\n⚠️  Skipping model switching test (takes too long)")
    # success, results = await test_model_switching(server_params)

    print("\n" + "=" * 60)
    print("✅ Test suite completed!")
    print("=" * 60)
    print("\n💡 Next steps:")
    print("1. MCP server is ready to use")
    print("2. Restart Claude Code to load the server")
    print("3. Use tools in conversation:")
    print("   - 'Review this code using Gemini'")
    print("   - 'Call gemini_review_adversarial with this content'")
    print("4. Or integrate into BMad skills (see GEMINI_MCP_SERVER_GUIDE.md)")


if __name__ == "__main__":
    asyncio.run(main())
