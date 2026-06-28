#!/usr/bin/env python3
"""
Ví dụ sử dụng Gemini Review Wrapper
Example: How to use Gemini Review Wrapper for BMad tasks
"""

import os
import json
from gemini_review_wrapper import GeminiReviewer

# Example 1: Adversarial review cho code file
def example_adversarial_review():
    """Ví dụ adversarial review cho Python code"""

    # Sample code with intentional bugs
    buggy_code = """
def process_payment(user_id, amount):
    # Process payment without validation
    conn = db.connect()
    query = f"UPDATE accounts SET balance = balance - {amount} WHERE user_id = {user_id}"
    conn.execute(query)
    conn.close()
    return True

def calculate_discount(price, user_level):
    if user_level == 'gold':
        return price * 0.9
    elif user_level == 'silver':
        return price * 0.95
    # Forgot to handle other levels - returns None!
"""

    print("=" * 60)
    print("EXAMPLE 1: Adversarial Code Review")
    print("=" * 60)

    try:
        reviewer = GeminiReviewer()
        result = reviewer.review_adversarial(
            content=buggy_code,
            content_type="code",
            also_consider=["Security vulnerabilities", "Missing error handling"]
        )

        print(f"\nModel: {result.get('model', 'unknown')}")
        print(f"Findings: {len(result.get('findings', []))} issues found\n")

        for i, finding in enumerate(result.get('findings', []), 1):
            severity = finding.get('severity', 'MEDIUM')
            desc = finding.get('description', 'No description')
            print(f"{i}. [{severity}] {desc}")

        if 'usage' in result:
            print(f"\nTokens used: {result['usage']}")

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure GOOGLE_GEMINI_API_KEY is set")


# Example 2: Code review cho git diff
def example_code_review():
    """Ví dụ code review cho git diff"""

    # Sample git diff
    diff_content = """
diff --git a/src/auth.py b/src/auth.py
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

    print("\n" + "=" * 60)
    print("EXAMPLE 2: Git Diff Code Review")
    print("=" * 60)

    try:
        reviewer = GeminiReviewer()
        result = reviewer.review_code(
            diff_content=diff_content,
            review_mode="full",
            persistent_facts=[
                "All user input must be validated",
                "Use parameterized queries to prevent SQL injection",
                "All functions must have docstrings"
            ]
        )

        print(f"\nReview Mode: {result.get('review_mode', 'unknown')}")

        findings = result.get('findings', [])
        if findings:
            print(f"\n{len(findings)} findings:\n")
            for i, finding in enumerate(findings, 1):
                layer = finding.get('layer', 'Unknown')
                severity = finding.get('severity', 'MEDIUM')
                issue = finding.get('issue', 'No issue description')
                location = finding.get('location', 'Unknown location')
                fix = finding.get('how_to_fix', 'No fix suggestion')

                print(f"{i}. [{layer}] - {severity}")
                print(f"   Issue: {issue}")
                print(f"   Location: {location}")
                print(f"   Fix: {fix}\n")

        summary = result.get('summary', {})
        if summary:
            print("Summary:")
            print(f"  HIGH:   {summary.get('high_count', 0)}")
            print(f"  MEDIUM: {summary.get('medium_count', 0)}")
            print(f"  LOW:    {summary.get('low_count', 0)}")
            print(f"  Total:  {summary.get('total_findings', 0)}")

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure GOOGLE_GEMINI_API_KEY is set")


# Example 3: Review actual file từ project
def example_review_file():
    """Ví dụ review actual file"""

    print("\n" + "=" * 60)
    print("EXAMPLE 3: Review Actual File")
    print("=" * 60)

    # Review chính file gemini_review_wrapper.py này
    current_file = __file__

    try:
        with open(current_file, 'r', encoding='utf-8') as f:
            code = f.read()

        reviewer = GeminiReviewer()
        result = reviewer.review_adversarial(
            content=code,
            content_type="code",
            also_consider=[
                "Code quality",
                "Best practices",
                "Python conventions (PEP 8)",
                "Error handling"
            ]
        )

        print(f"\nReviewed: {current_file}")
        print(f"Findings: {len(result.get('findings', []))} issues\n")

        for finding in result.get('findings', [])[:5]:  # Show top 5
            print(f"- [{finding['severity']}] {finding['description']}")

    except Exception as e:
        print(f"Error: {e}")


# Example 4: So sánh outputs
def example_model_comparison():
    """Ví dụ so sánh giữa các Gemini models"""

    print("\n" + "=" * 60)
    print("EXAMPLE 4: Model Comparison")
    print("=" * 60)

    test_code = """
def divide(a, b):
    return a / b
"""

    models = ["gemini-2.5-pro", "gemini-1.5-pro"]

    for model in models:
        try:
            print(f"\nTesting {model}...")
            reviewer = GeminiReviewer(model=model)
            result = reviewer.review_adversarial(
                content=test_code,
                content_type="code"
            )

            findings_count = len(result.get('findings', []))
            print(f"  Findings: {findings_count}")

            if 'usage' in result:
                usage = result['usage']
                if usage:
                    print(f"  Usage: {usage}")

        except Exception as e:
            print(f"  Error: {e}")


def main():
    """Run all examples"""

    # Check API key
    if not os.getenv("GOOGLE_GEMINI_API_KEY"):
        print("❌ ERROR: GOOGLE_GEMINI_API_KEY environment variable not set!")
        print("\nSet it with:")
        print("  Windows (PowerShell): $env:GOOGLE_GEMINI_API_KEY='your-key'")
        print("  Windows (CMD): set GOOGLE_GEMINI_API_KEY=your-key")
        print("  Linux/Mac: export GOOGLE_GEMINI_API_KEY='your-key'")
        return

    print("🚀 Gemini Review Wrapper Examples")
    print("Make sure you've installed dependencies:")
    print("  pip install -r requirements-gemini.txt")
    print()

    # Run examples
    example_adversarial_review()
    example_code_review()
    example_review_file()
    example_model_comparison()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
