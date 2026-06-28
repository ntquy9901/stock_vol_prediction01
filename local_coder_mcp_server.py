#!/usr/bin/env python3
"""
Local LLM MCP Server for BMad Review Skills
Supports: Ollama, vLLM, OpenAI-compatible APIs (DeepSeek, etc.)
Models: Qwen2.5-Coder, DeepSeek Coder, CodeLlama, etc.
"""

import os
import sys
import json
import asyncio
from typing import Any, Optional
from mcp.server.models import InitializationOptions
from mcp.server import Server, NotificationOptions
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# ============================================================================
# CONFIGURATION
# ============================================================================

# Runtime configuration
RUNTIME = os.getenv("LLM_RUNTIME", "ollama")  # Options: ollama, vllm, openai
MODEL = os.getenv("LLM_MODEL", "qwen2.5-coder:32b")
BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434")
API_KEY = os.getenv("LLM_API_KEY", "dummy")  # For vLLM/OpenAI-compatible

# Model configuration
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "8192"))

# Server info
SERVER_NAME = "local-coder-review-server"
SERVER_VERSION = "1.0.0"

print(f"[{SERVER_NAME}] Starting...", file=sys.stderr)
print(f"[{SERVER_NAME}] Runtime: {RUNTIME}", file=sys.stderr)
print(f"[{SERVER_NAME}] Model: {MODEL}", file=sys.stderr)
print(f"[{SERVER_NAME}] Base URL: {BASE_URL}", file=sys.stderr)


# ============================================================================
# RUNTIME IMPLEMENTATIONS
# ============================================================================

class LocalLLMReviewer:
    """Local LLM reviewer supporting multiple backends"""

    def __init__(self, model: str = MODEL):
        self.model_name = model
        self.runtime = RUNTIME
        self.base_url = BASE_URL

        # Validate runtime
        if self.runtime == "ollama":
            self._init_ollama()
        elif self.runtime in ["vllm", "openai"]:
            self._init_openai()
        else:
            raise ValueError(f"Unknown runtime: {self.runtime}")

    def _init_ollama(self):
        """Initialize Ollama client"""
        try:
            import ollama
            self.ollama_client = ollama
            print(f"[{SERVER_NAME}] ✓ Ollama client initialized", file=sys.stderr)
        except ImportError:
            print(f"[{SERVER_NAME}] ✗ Ollama not installed. Run: pip install ollama", file=sys.stderr)
            raise

    def _init_openai(self):
        """Initialize OpenAI-compatible client (vLLM, DeepSeek, etc.)"""
        try:
            from openai import OpenAI
            self.openai_client = OpenAI(
                base_url=self.base_url,
                api_key=API_KEY
            )
            print(f"[{SERVER_NAME}] ✓ OpenAI-compatible client initialized", file=sys.stderr)
        except ImportError:
            print(f"[{SERVER_NAME}] ✗ OpenAI not installed. Run: pip install openai", file=sys.stderr)
            raise

    async def call_llm(self, prompt: str) -> str:
        """Call LLM with prompt"""
        if self.runtime == "ollama":
            return await self._call_ollama(prompt)
        elif self.runtime in ["vllm", "openai"]:
            return await self._call_openai(prompt)
        else:
            raise ValueError(f"Unknown runtime: {self.runtime}")

    async def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API"""
        try:
            # Ollama doesn't have async support, use thread
            response = await asyncio.to_thread(
                self.ollama_client.chat,
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": TEMPERATURE,
                    "num_predict": MAX_TOKENS
                }
            )
            return response["message"]["content"]
        except Exception as e:
            raise Exception(f"Ollama call failed: {e}")

    async def _call_openai(self, prompt: str) -> str:
        """Call OpenAI-compatible API"""
        try:
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"OpenAI API call failed: {e}")

    async def adversarial_review(
        self,
        content: str,
        content_type: str = "code",
        also_consider: Optional[list[str]] = None
    ) -> dict[str, Any]:
        """Perform adversarial review"""

        prompt = self._build_adversarial_prompt(content, content_type, also_consider)

        try:
            response = await self.call_llm(prompt)
            findings = self._parse_findings(response)

            return {
                "model": self.model_name,
                "runtime": self.runtime,
                "findings": findings,
                "raw_response": response,
                "success": True
            }
        except Exception as e:
            return {
                "model": self.model_name,
                "runtime": self.runtime,
                "error": str(e),
                "success": False
            }

    async def code_review(
        self,
        diff_content: str,
        review_mode: str = "full",
        persistent_facts: Optional[list[str]] = None
    ) -> dict[str, Any]:
        """Perform code review with parallel layers"""

        prompt = self._build_code_review_prompt(diff_content, review_mode, persistent_facts)

        try:
            response = await self.call_llm(prompt)
            parsed = self._parse_code_review(response)

            return {
                "model": self.model_name,
                "runtime": self.runtime,
                "review_mode": review_mode,
                "findings": parsed.get("findings", []),
                "triage": parsed.get("triage", {}),
                "summary": parsed.get("summary", {}),
                "raw_response": response,
                "success": True
            }
        except Exception as e:
            return {
                "model": self.model_name,
                "runtime": self.runtime,
                "error": str(e),
                "success": False
            }

    def _build_adversarial_prompt(
        self,
        content: str,
        content_type: str,
        also_consider: Optional[list[str]]
    ) -> str:
        """Build adversarial review prompt"""
        prompt = f"""# Adversarial Code Review Task

You are a cynical, jaded code reviewer with zero patience for sloppy work. Review the following {content_type} with extreme skepticism.

## Content to Review:
```
{content}
```

"""

        if also_consider:
            prompt += f"\n## Also Consider:\n"
            prompt += "\n".join(f"- {c}" for c in also_consider) + "\n"

        prompt += """

## Instructions:
1. Assume problems exist - this content was submitted by a clueless weasel
2. Find at least FIVE issues to fix or improve
3. Look for what's MISSING, not just what's WRONG
4. Be skeptical of everything
5. Focus on: Security, Performance, Code Quality, Error Handling, Edge Cases
6. Use precise, professional tone

## Output Format (JSON only):
```json
{
  "findings": [
    {
      "severity": "HIGH|MEDIUM|LOW",
      "category": "Security|Performance|Code Quality|Design|Error Handling|Documentation",
      "issue": "Brief description of the problem",
      "location": "file:line or function name",
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

Return ONLY valid JSON, no markdown formatting:"""

        return prompt

    def _build_code_review_prompt(
        self,
        diff_content: str,
        review_mode: str,
        persistent_facts: Optional[list[str]]
    ) -> str:
        """Build code review prompt"""
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

Return ONLY valid JSON:"""

        return prompt

    def _parse_findings(self, response_text: str) -> list[dict[str, Any]]:
        """Parse findings from LLM response"""
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


# ============================================================================
# MCP SERVER
# ============================================================================

server = Server(SERVER_NAME)
reviewer = None  # Will be initialized per request with model override


@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    """List available tools"""
    return [
        Tool(
            name="local_review_adversarial",
            description="Perform adversarial review using local LLM (Ollama/vLLM). Finds issues in code, documents, or any content with extreme skepticism.",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Content to review (code, document, diff, etc.)"
                    },
                    "content_type": {
                        "type": "string",
                        "description": "Type of content",
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
                        "description": "Model override (optional, uses default if not specified)",
                        "default": MODEL
                    }
                },
                "required": ["content"]
            }
        ),
        Tool(
            name="local_review_code",
            description="Perform comprehensive code review using local LLM. Uses parallel review layers (Blind Hunter, Edge Case Hunter, Acceptance Auditor).",
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
                        "description": "Model override (optional)",
                        "default": MODEL
                    }
                },
                "required": ["diff_content"]
            }
        )
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls"""

    global reviewer

    if name == "local_review_adversarial":
        content = arguments.get("content", "")
        content_type = arguments.get("content_type", "code")
        also_consider = arguments.get("also_consider", [])
        model = arguments.get("model", MODEL)

        if not content:
            return [TextContent(
                type="text",
                text="Error: 'content' argument is required"
            )]

        # Initialize reviewer with model override if provided
        local_reviewer = LocalLLMReviewer(model=model)

        result = await local_reviewer.adversarial_review(
            content=content,
            content_type=content_type,
            also_consider=also_consider
        )

        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2, ensure_ascii=False)
        )]

    elif name == "local_review_code":
        diff_content = arguments.get("diff_content", "")
        review_mode = arguments.get("review_mode", "full")
        persistent_facts = arguments.get("persistent_facts", [])
        model = arguments.get("model", MODEL)

        if not diff_content:
            return [TextContent(
                type="text",
                text="Error: 'diff_content' argument is required"
            )]

        # Initialize reviewer with model override if provided
        local_reviewer = LocalLLMReviewer(model=model)

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
                server_name=SERVER_NAME,
                server_version=SERVER_VERSION,
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={}
                )
            )
        )


if __name__ == "__main__":
    print(f"[{SERVER_NAME}] MCP server starting...", file=sys.stderr)
    print(f"[{SERVER_NAME}] Runtime: {RUNTIME}", file=sys.stderr)
    print(f"[{SERVER_NAME}] Model: {MODEL}", file=sys.stderr)
    print(f"[{SERVER_NAME}] Base URL: {BASE_URL}", file=sys.stderr)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n[{SERVER_NAME}] Shutting down...", file=sys.stderr)
    except Exception as e:
        print(f"\n[{SERVER_NAME}] Error: {e}", file=sys.stderr)
        sys.exit(1)
