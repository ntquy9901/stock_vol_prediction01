#!/usr/bin/env python3
"""
Gemini MCP Server for BMad Review Skills
Provides Gemini model access through Model Context Protocol
"""

import os
import sys
import json
import asyncio
from typing import Any, Optional
from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import google.generativeai as genai

# Environment variables
GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")

# Validate API key on startup
if not GEMINI_API_KEY:
    print("ERROR: GOOGLE_GEMINI_API_KEY environment variable is required", file=sys.stderr)
    sys.exit(1)

# Initialize Gemini
genai.configure(api_key=GEMINI_API_KEY)


class GeminiReviewer:
    """Gemini-based reviewer for MCP tools"""

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model_name = model
        self.model = genai.GenerativeModel(model)

    async def adversarial_review(
        self,
        content: str,
        content_type: str = "code",
        also_consider: Optional[list[str]] = None
    ) -> dict[str, Any]:
        """Perform adversarial review using Gemini"""

        prompt = self._build_adversarial_prompt(content, content_type, also_consider)

        try:
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=8192,
                )
            )

            findings = self._parse_findings(response.text)

            return {
                "model": self.model_name,
                "findings": findings,
                "raw_response": response.text,
                "usage": {
                    "prompt_tokens": response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') else 0,
                }
            }
        except Exception as e:
            return {
                "error": str(e),
                "model": self.model_name
            }

    async def code_review(
        self,
        diff_content: str,
        review_mode: str = "full",
        persistent_facts: Optional[list[str]] = None
    ) -> dict[str, Any]:
        """Perform code review using Gemini"""

        prompt = self._build_code_review_prompt(diff_content, review_mode, persistent_facts)

        try:
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.5,
                    max_output_tokens=8192,
                )
            )

            parsed = self._parse_code_review(response.text)

            return {
                "model": self.model_name,
                "review_mode": review_mode,
                "findings": parsed.get("findings", []),
                "triage": parsed.get("triage", {}),
                "summary": parsed.get("summary", {}),
                "raw_response": response.text
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
        also_consider: Optional[list[str]]
    ) -> str:
        """Build prompt for adversarial review"""
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
Return findings as a JSON object with this exact structure:
```json
{
  "findings": [
    {
      "severity": "HIGH|MEDIUM|LOW",
      "category": "Security|Performance|Code Quality|Design|Documentation|Other",
      "issue": "Brief description of the problem",
      "location": "file:line or specific location",
      "how_to_fix": "Specific actionable fix instructions",
      "why_it_matters": "Why this issue is important"
    }
  ],
  "summary": {
    "total_findings": 0,
    "high_count": 0,
    "medium_count": 0,
    "low_count": 0,
    "categories": {}
  }
}
```

Begin your review now. Return ONLY valid JSON, no markdown formatting:"""

        return prompt

    def _build_code_review_prompt(
        self,
        diff_content: str,
        review_mode: str,
        persistent_facts: Optional[list[str]]
    ) -> str:
        """Build prompt for code review"""
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
1. **Blind Hunter**: Logic bugs, edge cases, race conditions, off-by-one errors
2. **Edge Case Hunter**: Empty/null handling, boundary conditions, malformed input
3. **Acceptance Auditor**: Matches spec? Testable? Complete? Production-ready?

## Output Format (JSON only):
```json
{{
  "findings": [
    {{
      "layer": "Blind Hunter|Edge Case Hunter|Acceptance Auditor",
      "severity": "HIGH|MEDIUM|LOW",
      "issue": "Brief description",
      "location": "file:line or function name",
      "how_to_fix": "Specific fix instructions",
      "why_it_matters": "Impact explanation"
    }}
  ],
  "triage": {{
    "must_fix": [],
    "should_fix": [],
    "nice_to_have": []
  }},
  "summary": {{
    "total_findings": 0,
    "high_count": 0,
    "medium_count": 0,
    "low_count": 0,
    "layers": {{}}
  }}
}}
```

Begin your review now. Return ONLY valid JSON:"""

        return prompt

    def _parse_findings(self, response_text: str) -> list[dict[str, Any]]:
        """Parse findings from Gemini response"""
        try:
            # Try to extract JSON
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                json_str = response_text[start:end].strip()
                data = json.loads(json_str)
                return data.get("findings", [])
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                json_str = response_text[start:end].strip()
                data = json.loads(json_str)
                return data.get("findings", [])
            else:
                data = json.loads(response_text)
                return data.get("findings", [])
        except:
            # Fallback: parse as markdown list
            findings = []
            lines = response_text.split('\n')

            for line in lines:
                line = line.strip()
                if line.startswith(('-', '*')) and len(line) > 2:
                    severity = "MEDIUM"
                    if "HIGH" in line.upper():
                        severity = "HIGH"
                    elif "LOW" in line.upper():
                        severity = "LOW"

                    findings.append({
                        "severity": severity,
                        "description": line[2:].strip()
                    })

            return findings

    def _parse_code_review(self, response_text: str) -> dict[str, Any]:
        """Parse code review response"""
        try:
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
                return json.loads(response_text)
        except:
            return {
                "findings": self._parse_findings(response_text),
                "triage": {},
                "summary": {}
            }


# Create MCP server instance
server = Server("gemini-review-server")
reviewer = GeminiReviewer()


@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    """List available tools"""
    return [
        Tool(
            name="gemini_review_adversarial",
            description="Perform adversarial review using Gemini model. Finds issues in code, documents, or any content with extreme skepticism.",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Content to review (code, document, diff, etc.)"
                    },
                    "content_type": {
                        "type": "string",
                        "description": "Type of content (code, diff, document, spec)",
                        "default": "code",
                        "enum": ["code", "diff", "document", "spec", "config"]
                    },
                    "also_consider": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Additional aspects to consider (e.g., ['Security', 'Performance'])",
                        "default": []
                    },
                    "model": {
                        "type": "string",
                        "description": "Gemini model to use",
                        "default": DEFAULT_MODEL,
                        "enum": ["gemini-2.5-pro", "gemini-2.0-flash-exp", "gemini-1.5-pro"]
                    }
                },
                "required": ["content"]
            }
        ),
        Tool(
            name="gemini_review_code",
            description="Perform comprehensive code review using Gemini model. Uses parallel review layers (Blind Hunter, Edge Case Hunter, Acceptance Auditor).",
            inputSchema={
                "type": "object",
                "properties": {
                    "diff_content": {
                        "type": "string",
                        "description": "Git diff or code changes to review"
                    },
                    "review_mode": {
                        "type": "string",
                        "description": "Review mode",
                        "default": "full",
                        "enum": ["full", "no-spec", "quick"]
                    },
                    "persistent_facts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Context/facts to keep in mind during review",
                        "default": []
                    },
                    "model": {
                        "type": "string",
                        "description": "Gemini model to use",
                        "default": DEFAULT_MODEL,
                        "enum": ["gemini-2.5-pro", "gemini-2.0-flash-exp", "gemini-1.5-pro"]
                    }
                },
                "required": ["diff_content"]
            }
        )
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls"""

    if name == "gemini_review_adversarial":
        content = arguments.get("content", "")
        content_type = arguments.get("content_type", "code")
        also_consider = arguments.get("also_consider", [])
        model = arguments.get("model", DEFAULT_MODEL)

        if not content:
            return [TextContent(
                type="text",
                text="Error: 'content' argument is required"
            )]

        # Update reviewer model if different
        local_reviewer = GeminiReviewer(model=model)
        result = await local_reviewer.adversarial_review(
            content=content,
            content_type=content_type,
            also_consider=also_consider
        )

        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2, ensure_ascii=False)
        )]

    elif name == "gemini_review_code":
        diff_content = arguments.get("diff_content", "")
        review_mode = arguments.get("review_mode", "full")
        persistent_facts = arguments.get("persistent_facts", [])
        model = arguments.get("model", DEFAULT_MODEL)

        if not diff_content:
            return [TextContent(
                type="text",
                text="Error: 'diff_content' argument is required"
            )]

        # Update reviewer model if different
        local_reviewer = GeminiReviewer(model=model)
        result = await local_reviewer.code_review(
            diff_content=diff_content,
            review_mode=review_mode,
            persistent_facts=persistent_facts
        )

        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2, ensure_ascii=False)
        )]

    else:
        return [TextContent(
            type="text",
            text=f"Error: Unknown tool '{name}'"
        )]


async def main():
    """Main entry point"""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="gemini-review-server",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={}
                )
            )
        )


if __name__ == "__main__":
    asyncio.run(main())
