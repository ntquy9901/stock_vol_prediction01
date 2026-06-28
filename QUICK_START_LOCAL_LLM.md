# Quick Start: Local Open-Source LLM cho Code Review

**Bắt đầu trong 15 phút!**

---

## ⚡ Option 1: Fastest (Ollama - Recommended)

### Bước 1: Install Ollama (2 phút)

```bash
# Linux/Mac
curl -fsSL https://ollama.com/install.sh | sh

# Windows (PowerShell - Run as Administrator)
iwr -useb https://ollama.com/install.ps1 | iex
```

### Bước 2: Pull Model (5-10 phút)

```bash
# Option 1A: Qwen2.5-Coder 32B (Best quality, cần 32GB RAM)
ollama pull qwen2.5-coder:32b

# Option 1B: Qwen2.5-Coder 7B (Faster, cần 16GB RAM)
ollama pull qwen2.5-coder:7b

# Option 1C: DeepSeek Coder V2 (16GB RAM)
ollama pull deepseek-coder-v2:16b
```

### Bước 3: Test Code Review (1 phút)

```bash
# Interactive mode
ollama run qwen2.5-coder:32b

# Trong Ollama prompt, type:
Review this Python code for bugs and security issues:

def process_payment(user_id, amount):
    query = f"UPDATE accounts SET balance = balance - {amount} WHERE user_id = {user_id}"
    db.execute(query)
    return True

# Press Ctrl+D to exit
```

### Bước 4: Python Integration (2 phút)

```python
# install
pip install ollama

# test
import ollama

response = ollama.chat(
    model='qwen2.5-coder:32b',
    messages=[
        {
            'role': 'user',
            'content': '''Review this code for security issues:
def process_payment(user_id, amount):
    query = f"UPDATE accounts SET balance = balance - {amount} WHERE user_id = {user_id}"
    db.execute(query)
    return True'''
        }
    ]
)

print(response['message']['content'])
```

**✅ Done!** Bạn đã có local LLM code review working!

---

## 🔧 Option 2: MCP Integration (30 phút)

### Bước 1: Install Ollama + Model (như trên)

### Bước 2: Clone local-llm-mcp (5 phút)

```bash
git clone https://github.com/NAJEMWEHBE/local-llm-mcp.git
cd local-llm-mcp
pip install -r requirements.txt
```

### Bước 3: Configure (5 phút)

Edit `config.json`:
```json
{
  "runtime": "ollama",
  "model": "qwen2.5-coder:32b",
  "baseUrl": "http://localhost:11434",
  "temperature": 0.3
}
```

### Bước 4: Update .mcp.json (2 phút)

```json
{
  "local-llm-mcp": {
    "command": "python",
    "args": ["D:/local-llm-mcp/main.py"],
    "env": {
      "MODEL": "qwen2.5-coder:32b",
      "RUNTIME": "ollama"
    }
  }
}
```

### Bước 5: Test MCP Server (5 phút)

```bash
# Terminal 1: Start Ollama
ollama run qwen2.5-coder:32b

# Terminal 2: Start MCP server
cd D:/local-llm-mcp
python main.py
```

### Bước 6: Use trong Claude Code

```
Please review this code using local LLM:
[paste code]
```

**✅ Done!** MCP server integrated với Claude Code!

---

## 🚀 Option 3: Custom MCP Server (1-2 hours)

**Dựa trên `gemini_mcp_server.py` code của bạn**

### Bước 1: Create `local_coder_mcp_server.py`

```python
#!/usr/bin/env python3
"""
Local LLM MCP Server cho Code Review
Suports: Ollama, vLLM, any OpenAI-compatible API
"""

import os
import sys
import json
import asyncio
from typing import Any
from mcp.server.models import InitializationOptions
from mcp.server import Server, NotificationOptions
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Configuration
RUNTIME = os.getenv("LLM_RUNTIME", "ollama")  # ollama, vllm, openai
MODEL = os.getenv("LLM_MODEL", "qwen2.5-coder:32b")
BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434")

server = Server("local-coder-review-server")

def call_ollama(prompt: str, model: str) -> str:
    """Call Ollama API"""
    try:
        import ollama
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"]
    except Exception as e:
        return f"Error calling Ollama: {e}"

def call_openai(prompt: str, model: str, base_url: str) -> str:
    """Call OpenAI-compatible API (vLLM, DeepSeek, etc.)"""
    try:
        from openai import OpenAI
        client = OpenAI(base_url=base_url, api_key="dummy")
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error calling API: {e}"

def build_adversarial_prompt(content: str, content_type: str = "code") -> str:
    """Build adversarial review prompt"""
    return f"""# Adversarial Code Review Task

You are a cynical, jaded code reviewer. Review the following {content_type} with extreme skepticism.

## Content to Review:
```
{content}
```

## Instructions:
1. Find at least FIVE issues to fix or improve
2. Look for what's MISSING, not just what's WRONG
3. Focus on: Security, Performance, Code Quality, Error Handling
4. Be specific and actionable

## Output Format (JSON):
```json
{{
  "findings": [
    {{
      "severity": "HIGH|MEDIUM|LOW",
      "category": "Security|Performance|Code Quality|Error Handling",
      "issue": "Brief description",
      "location": "file:line or function name",
      "how_to_fix": "Specific fix instructions",
      "why_it_matters": "Why this is important"
    }}
  ],
  "summary": {{
    "total_findings": 0,
    "high_count": 0,
    "medium_count": 0,
    "low_count": 0
  }}
}}
```

Return ONLY valid JSON, no markdown formatting:"""

@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    """List available tools"""
    return [
        Tool(
            name="local_review_adversarial",
            description="Perform adversarial code review using local LLM",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Code or content to review"
                    },
                    "content_type": {
                        "type": "string",
                        "description": "Type of content",
                        "default": "code",
                        "enum": ["code", "diff", "document", "config"]
                    }
                },
                "required": ["content"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls"""
    
    if name == "local_review_adversarial":
        content = arguments.get("content", "")
        content_type = arguments.get("content_type", "code")
        
        if not content:
            return [TextContent(
                type="text",
                text="Error: 'content' argument is required"
            )]
        
        # Build prompt
        prompt = build_adversarial_prompt(content, content_type)
        
        # Call appropriate runtime
        if RUNTIME == "ollama":
            response = call_ollama(prompt, MODEL)
        elif RUNTIME in ["vllm", "openai"]:
            response = call_openai(prompt, MODEL, BASE_URL)
        else:
            response = f"Error: Unknown runtime '{RUNTIME}'"
        
        return [TextContent(
            type="text",
            text=response
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
                server_name="local-coder-review-server",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={}
                )
            )
        )

if __name__ == "__main__":
    print(f"Starting Local Coder MCP Server...")
    print(f"Runtime: {RUNTIME}")
    print(f"Model: {MODEL}")
    print(f"Base URL: {BASE_URL}")
    asyncio.run(main())
```

### Bước 2: Update .mcp.json

```json
{
  "local-coder-review": {
    "command": "python",
    "args": ["D:/bmad-projects/stock_vol_prediction01/local_coder_mcp_server.py"],
    "env": {
      "LLM_RUNTIME": "ollama",
      "LLM_MODEL": "qwen2.5-coder:32b"
    }
  }
}
```

### Bước 3: Test MCP Server

```bash
# Terminal 1: Start Ollama
ollama run qwen2.5-coder:32b

# Terminal 2: Test MCP server
python local_coder_mcp_server.py
```

### Bước 4: Use trong Claude Code

Restart Claude Code và test:

```
Review this code using local_review_adversarial tool:
[paste code]
```

**✅ Done!** Custom MCP server working!

---

## 📊 Performance Comparison (Local vs API)

### **Test Task:** Review 100 lines of Python code

| Method | Time | Cost | Quality |
|--------|------|------|---------|
| **Ollama (Qwen 32B, 32GB RAM)** | ~8-12s | $0 | 85% Claude |
| **Ollama (Qwen 7B, 16GB RAM)** | ~3-5s | $0 | 75% Claude |
| **vLLM (Qwen 32B, GPU)** | ~2-4s | $0 | 85% Claude |
| **DeepSeek V4 API** | ~5-8s | ~$0.01 | 92% Claude |
| **Claude Sonnet API** | ~3-5s | ~$0.15 | 100% (baseline) |

**Note:** Based on benchmarks, not actual testing on your code

---

## 🎯 Recommendations cho Project của Bạn

**Dựa trên context (stock volatility prediction, ML project):**

### **Best Setup cho Development:**

```bash
# Use Qwen2.5-Coder 32B via Ollama
ollama run qwen2.5-coder:32b
```

**Why:**
- ✅ Good ML/PyTorch knowledge
- ✅ Strong code review (92.7% HumanEval)
- ✅ Runs on 32GB RAM (reasonable hardware)
- ✅ Free (no API costs)

---

### **Best Setup cho Integration:**

```bash
# Use local-llm-mcp with Ollama
git clone https://github.com/NAJEMWEHBE/local-llm-mcp
# Configure with qwen2.5-coder:32b
```

**Why:**
- ✅ Native Claude Code integration
- ✅ Works with BMad workflow
- ✅ Flexible model switching
- ✅ Easy setup

---

### **Best Setup cho Production (if needed):**

```bash
# Use DeepSeek V4 Flash API
# (~$0.43/1M tokens, best quality/price ratio)
```

**Why:**
- ✅ Best benchmark performance
- ✅ Very affordable
- ✅ No hardware requirements
- ✅ Reliable uptime

---

## 🔧 Troubleshooting

### **Issue:** "Model not found"

**Fix:**
```bash
# List available models
ollama list

# Pull model if missing
ollama pull qwen2.5-coder:32b
```

---

### **Issue:** "Out of memory"

**Fix:**
```bash
# Use smaller model
ollama pull qwen2.5-coder:7b

# Or quit other applications
# Or upgrade RAM (recommended: 32GB)
```

---

### **Issue:** "MCP server not starting"

**Fix:**
```bash
# Check Ollama is running
ollama list

# Check MCP server config
cat .mcp.json

# Check MCP server logs
python local_coder_mcp_server.py
```

---

### **Issue:** "Poor review quality"

**Fix:**
```bash
# Try larger model
ollama pull qwen2.5-coder:32b  # instead of 7B

# Or use API for better quality
# DeepSeek V4 Flash: $0.43/1M tokens
```

---

## 📚 Next Steps

1. **Test local model:**
   ```bash
   ollama run qwen2.5-coder:32b
   ```

2. **Integrate with BMad:**
   - Build MCP server (use template above)
   - Update .mcp.json
   - Test with Claude Code

3. **Benchmark quality:**
   - Run same code review với Claude vs Local
   - Compare findings
   - Adjust model choice based on results

4. **Production deployment:**
   - Deploy vLLM server (if GPU available)
   - Or use DeepSeek V4 Flash API (affordable)
   - Add monitoring/logging

---

**Quick Start Version:** 1.0  
**Based on:** [OPEN_SOURCE_CODE_REVIEW_RESEARCH.md](OPEN_SOURCE_CODE_REVIEW_RESEARCH.md)

**Estimated Time to Complete:**
- Option 1 (Ollama only): **15 minutes**
- Option 2 (MCP integration): **30 minutes**
- Option 3 (Custom server): **1-2 hours**

**Questions?**
- See full research report: OPEN_SOURCE_CODE_REVIEW_RESEARCH.md
- Model benchmarks: [SWE-bench](https://www.swebench.com/)
- Deployment guide: [Local LLM Hosting 2025](https://medium.com/@rosgluk/local-llm-hosting-complete-2025-guide-ollama-vllm-localai-jan-lm-studio-more-f98136ce7e4a)
