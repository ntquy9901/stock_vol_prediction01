#!/usr/bin/env python3
"""
Gemini API Wrapper for BMad Review Skills
Sử dụng Gemini model thay vì Claude cho code review
"""

import os
import sys
import json
import google.generativeai as genai
from typing import Dict, List, Optional

class GeminiReviewer:
    """Gemini-based code reviewer tương thích với BMad review workflow"""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-pro"):
        """
        Khởi tạo Gemini reviewer

        Args:
            api_key: Gemini API key (mặc định lấy từ GOOGLE_GEMINI_API_KEY env var)
            model: Gemini model name (default: gemini-2.5-pro)
        """
        self.api_key = api_key or os.getenv("GOOGLE_GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Cần cung cấp Gemini API key qua parameter hoặc GOOGLE_GEMINI_API_KEY env var")

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model)
        self.model_name = model

    def review_adversarial(
        self,
        content: str,
        content_type: str = "code",
        also_consider: Optional[List[str]] = None
    ) -> Dict[str, any]:
        """
        Thực hiện adversarial review sử dụng Gemini

        Args:
            content: Nội dung cần review
            content_type: Loại nội dung (code, diff, spec, doc)
            also_consider: Các aspect thêm cần xem xét

        Returns:
            Dict với findings và metadata
        """
        # Build prompt theo BMad adversarial review pattern
        prompt = self._build_adversarial_prompt(content, content_type, also_consider)

        try:
            response = self.model.generate_content(prompt)
            findings = self._parse_findings(response.text)

            return {
                "model": self.model_name,
                "findings": findings,
                "raw_response": response.text,
                "usage": response.usage_metadata.to_dict() if hasattr(response, 'usage_metadata') else {}
            }
        except Exception as e:
            return {
                "error": str(e),
                "model": self.model_name
            }

    def review_code(
        self,
        diff_content: str,
        review_mode: str = "full",
        persistent_facts: Optional[List[str]] = None
    ) -> Dict[str, any]:
        """
        Code review chuyên sâu cho code changes

        Args:
            diff_content: Git diff hoặc code changes
            review_mode: "full" hoặc "no-spec"
            persistent_facts: Các facts cần giữ trong tâm

        Returns:
            Dict với review results
        """
        prompt = self._build_code_review_prompt(diff_content, review_mode, persistent_facts)

        try:
            response = self.model.generate_content(prompt)
            parsed = self._parse_code_review(response.text)

            return {
                "model": self.model_name,
                "review_mode": review_mode,
                "findings": parsed.get("findings", []),
                "triage": parsed.get("triage", {}),
                "raw_response": response.text,
                "usage": response.usage_metadata.to_dict() if hasattr(response, 'usage_metadata') else {}
            }
        except Exception as e:
            return {
                "error": str(e),
                "model": self.model_name
            }

    def _build_adversarial_prompt(
        self,
        content: str,
        content_type: str,
        also_consider: Optional[List[str]]
    ) -> str:
        """Build prompt cho adversarial review"""
        prompt = f"""# Adversarial Review Task

You are a cynical, jaded reviewer with zero patience for sloppy work. Review the following {content_type} with extreme skepticism.

## Content to Review:
```
{content}
```

"""
        if also_consider:
            prompt += f"\n## Also Consider:\n" + "\n".join(f"- {c}" for c in also_consider)

        prompt += """

## Instructions:
1. Assume problems exist - this content was submitted by a clueless weasel
2. Find at least TEN issues to fix or improve
3. Look for what's MISSING, not just what's WRONG
4. Be skeptical of everything
5. Use precise, professional tone - no profanity or personal attacks

## Output Format:
Return findings as a Markdown list with descriptions only. Each finding should be:
- Specific and actionable
- Include severity (HIGH/MEDIUM/LOW)
- Include the exact issue location
- Suggest how to fix it

Begin your review now:"""
        return prompt

    def _build_code_review_prompt(
        self,
        diff_content: str,
        review_mode: str,
        persistent_facts: Optional[List[str]]
    ) -> str:
        """Build prompt cho code review"""
        prompt = f"""# Code Review Task

You are an elite code reviewer performing adversarial analysis.

## Code Changes to Review:
```
{diff_content}
```

"""

        if persistent_facts:
            prompt += "\n## Persistent Facts (Context):\n"
            prompt += "\n".join(f"- {fact}" for fact in persistent_facts) + "\n"

        prompt += f"""

## Review Mode: {review_mode.upper()}

## Instructions:
Review using these parallel layers:
1. **Blind Hunter**: Logic bugs, edge cases, race conditions
2. **Edge Case Hunter**: Empty/null handling, boundary conditions
3. **Acceptance Auditor**: Matches spec? Testable? Complete?

## Output Format (JSON):
```json
{{
  "findings": [
    {{
      "layer": "Blind Hunter|Edge Case Hunter|Acceptance Auditor",
      "severity": "HIGH|MEDIUM|LOW",
      "issue": "Brief description",
      "location": "file:line",
      "how_to_fix": "Specific fix instructions"
    }}
  ],
  "summary": {{
    "high_count": 0,
    "medium_count": 0,
    "low_count": 0,
    "total_findings": 0
  }}
}}
```

Begin your review now:"""
        return prompt

    def _parse_findings(self, response_text: str) -> List[Dict]:
        """Parse findings từ Gemini response"""
        findings = []
        lines = response_text.split('\n')

        current_finding = {}
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Try to parse markdown list items
            if line.startswith('- ') or line.startswith('* '):
                if current_finding:
                    findings.append(current_finding)
                    current_finding = {}

                # Extract severity if present
                severity = "MEDIUM"  # default
                if "HIGH" in line.upper():
                    severity = "HIGH"
                elif "LOW" in line.upper():
                    severity = "LOW"

                current_finding = {
                    "description": line[2:],
                    "severity": severity
                }
            elif current_finding:
                # Append details to current finding
                if "description" in current_finding:
                    current_finding["description"] += " " + line
                else:
                    current_finding["description"] = line

        if current_finding:
            findings.append(current_finding)

        return findings

    def _parse_code_review(self, response_text: str) -> Dict:
        """Parse code review response"""
        # Try to extract JSON from response
        try:
            # Look for JSON block
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                json_str = response_text[start:end].strip()
                return json.loads(json_str)
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                json_str = response_text[start:end].strip()
                return json.loads(json_str)
            else:
                # Try parsing entire response as JSON
                return json.loads(response_text)
        except:
            # Fallback: return as findings list
            return {
                "findings": self._parse_findings(response_text),
                "triage": {},
                "parse_error": "Could not parse JSON, returned as raw findings"
            }


def main():
    """CLI entry point"""
    if len(sys.argv) < 2:
        print("Usage: python gemini_review_wrapper.py <command> [args]")
        print("\nCommands:")
        print("  adversarial <content_file>     - Run adversarial review")
        print("  code <diff_file> [mode]        - Run code review")
        sys.exit(1)

    command = sys.argv[1]
    reviewer = GeminiReviewer()

    if command == "adversarial":
        if len(sys.argv) < 3:
            print("Error: adversarial requires content_file argument")
            sys.exit(1)

        with open(sys.argv[2], 'r', encoding='utf-8') as f:
            content = f.read()

        result = reviewer.review_adversarial(content)
        print(json.dumps(result, indent=2, ensure_ascii=False))

    elif command == "code":
        if len(sys.argv) < 3:
            print("Error: code requires diff_file argument")
            sys.exit(1)

        with open(sys.argv[2], 'r', encoding='utf-8') as f:
            diff = f.read()

        mode = sys.argv[3] if len(sys.argv) > 3 else "full"
        result = reviewer.review_code(diff, review_mode=mode)
        print(json.dumps(result, indent=2, ensure_ascii=False))

    else:
        print(f"Error: Unknown command '{command}'")
        sys.exit(1)


if __name__ == "__main__":
    main()
