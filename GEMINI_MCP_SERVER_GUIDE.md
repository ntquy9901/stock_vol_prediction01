# Gemini MCP Server cho BMad Review Skills

## Tổng quan

Gemini MCP Server cho phép các BMad review skills sử dụng Google Gemini model thay vì Claude model thông qua Model Context Protocol (MCP).

## Cài đặt

### Bước 1: Cài đặt dependencies

```bash
pip install -r requirements-gemini-mcp.txt
```

**Dependencies:**
- `mcp>=0.9.0` - MCP Python SDK
- `google-generativeai>=0.8.0` - Google Generative AI SDK

### Bước 2: Cấu hình API Key

Tạo Google Gemini API key tại: https://makersuite.google.com/app/apikey

**Windows (PowerShell):**
```powershell
$env:GOOGLE_GEMINI_API_KEY="your-api-key-here"
```

**Windows (CMD):**
```cmd
set GOOGLE_GEMINI_API_KEY=your-api-key-here
```

**Linux/Mac:**
```bash
export GOOGLE_GEMINI_API_KEY="your-api-key-here"
```

Hoặc thêm vào `.env` file:
```
GOOGLE_GEMINI_API_KEY=your-api-key-here
```

### Bước 3: Kiểm tra kết nối

```bash
# Test MCP server
python gemini_mcp_server.py
```

Nếu thành công, server sẽ start và chờ input từ STDIN (MCP protocol).

## MCP Tools Available

Server cung cấp 2 tools:

### 1. `gemini_review_adversarial`

Adversarial review cho bất kỳ content nào.

**Parameters:**
- `content` (required): Nội dung cần review
- `content_type` (optional): Loại content - "code", "diff", "document", "spec", "config" (default: "code")
- `also_consider` (optional): Array các aspects thêm để xem xét - ["Security", "Performance", etc.] (default: [])
- `model` (optional): Gemini model - "gemini-2.5-pro", "gemini-2.0-flash-exp", "gemini-1.5-pro" (default: "gemini-2.5-pro")

**Returns:** JSON object với findings, summary, metadata.

### 2. `gemini_review_code`

Comprehensive code review với parallel layers.

**Parameters:**
- `diff_content` (required): Git diff hoặc code changes
- `review_mode` (optional): "full", "no-spec", "quick" (default: "full")
- `persistent_facts` (optional): Array các context/facts để giữ trong tâm (default: [])
- `model` (optional): Gemini model (default: "gemini-2.5-pro")

**Returns:** JSON object với findings từ 3 layers (Blind Hunter, Edge Case Hunter, Acceptance Auditor), triage, summary.

## Cách sử dụng

### Cách 1: Trong Claude Code (tự động)

MCP server đã được đăng ký trong `.mcp.json`. Khi bạn start Claude Code, server sẽ tự động load và tools sẽ có sẵn.

Bạn có thể gọi tools trực tiếp:

```
Please review this code using Gemini:
[paste code]
```

Hoặc explicit:
```
Use gemini_review_adversarial tool to review this code with content_type="code" and also_consider=["Security", "Performance"]
```

### Cách 2: Từ bmad-review skills

Các BMad review skills có thể sử dụng MCP tools thay vì built-in review logic.

**Example trong BMad workflow:**
```markdown
## Step 2: Adversarial Analysis

Instead of using built-in Claude analysis, call the MCP tool:

Call tool: `gemini_review_adversarial`
- content: {diff_content}
- content_type: "code"
- also_consider: ["Security", "Performance", "Code Quality"]
```

### Cách 3: Manual test

Tạo test script để test MCP server:

```python
import asyncio
import json
from mcp.client.stdio import StdioServerParameters
from mcp.client import Client

async def test_gemini_review():
    server_params = StdioServerParameters(
        command="python",
        args=["gemini_mcp_server.py"],
        env={"GOOGLE_GEMINI_API_KEY": "your-key-here"}
    )

    async with Client(server_params) as client:
        # List tools
        tools = await client.list_tools()
        print("Available tools:", [tool.name for tool in tools])

        # Call adversarial review
        result = await client.call_tool(
            "gemini_review_adversarial",
            {
                "content": "def foo():\n    return 1/0",
                "content_type": "code",
                "also_consider": ["Security"]
            }
        )
        print("Result:", result)

asyncio.run(test_gemini_review())
```

## Available Models

| Model | Description | Context | Pricing |
|-------|-------------|---------|---------|
| `gemini-2.5-pro` | Mới nhất, mạnh nhất (recommended) | 1M tokens | $1.25/1M input + $5/1M output |
| `gemini-2.0-flash-exp` | Nhanh, free tier experiment | 1M tokens | Free (có limits) |
| `gemini-1.5-pro` | Stable version cũ hơn | 1M tokens | $0.075/1M input + $0.30/1M output |

## Output Format

### Adversarial Review Output

```json
{
  "model": "gemini-2.5-pro",
  "findings": [
    {
      "severity": "HIGH",
      "category": "Security",
      "issue": "SQL injection vulnerability in user_id parameter",
      "location": "src/auth.py:45",
      "how_to_fix": "Use parameterized queries instead of string concatenation",
      "why_it_matters": "Allows attackers to execute arbitrary SQL queries"
    }
  ],
  "summary": {
    "total_findings": 10,
    "high_count": 2,
    "medium_count": 5,
    "low_count": 3,
    "categories": {
      "Security": 3,
      "Performance": 2,
      "Code Quality": 5
    }
  }
}
```

### Code Review Output

```json
{
  "model": "gemini-2.5-pro",
  "review_mode": "full",
  "findings": [
    {
      "layer": "Blind Hunter",
      "severity": "HIGH",
      "issue": "Race condition in balance update",
      "location": "src/payment.py:78",
      "how_to_fix": "Add database transaction lock or use atomic operation",
      "why_it_matters": "Can cause double-spending vulnerability"
    }
  ],
  "triage": {
    "must_fix": ["Finding 1", "Finding 3"],
    "should_fix": ["Finding 2"],
    "nice_to_have": ["Finding 4"]
  },
  "summary": {
    "total_findings": 12,
    "high_count": 3,
    "medium_count": 6,
    "low_count": 3,
    "layers": {
      "Blind Hunter": 5,
      "Edge Case Hunter": 4,
      "Acceptance Auditor": 3
    }
  }
}
```

## Troubleshooting

### MCP server không start

**Error:** `ERROR: GOOGLE_GEMINI_API_KEY environment variable is required`

**Fix:**
```bash
# Check env var
echo $GOOGLE_GEMINI_API_KEY  # Linux/Mac
echo %GOOGLE_GEMINI_API_KEY%  # Windows CMD
echo $env:GOOGLE_GEMINI_API_KEY  # Windows PowerShell

# Set if missing
export GOOGLE_GEMINI_API_KEY="your-key"
```

### Tools không available trong Claude Code

**Check MCP server status:**
```bash
# Test server manually
python gemini_mcp_server.py
```

Nếu server chạy nhưng tools không xuất hiện trong Claude Code:
1. Restart Claude Code
2. Check `.mcp.json` syntax (valid JSON)
3. Check path trong `args` là absolute path

### Quota exceeded

**Error:** `Quota exceeded` hoặc `429 Resource exhausted`

**Fix:**
- Kiểm tra usage tại: https://makersuite.google.com/app/apikey
- Upgrade tier hoặc chờ đến ngày mai (free tier reset mỗi ngày)
- Switch sang `gemini-2.0-flash-exp` (free tier hào phóng hơn)

### Response quality kém

**Tweaks:**

1. **Thử model mạnh hơn:**
```json
{
  "model": "gemini-2.5-pro"  // Thay vì gemini-1.5-pro
}
```

2. **Thêm persistent_facts cho context:**
```json
{
  "persistent_facts": [
    "All code must follow PEP 8",
    "Security is critical - check for SQL injection, XSS, CSRF",
    "Performance matters - optimize hot paths"
  ]
}
```

3. **Tweak prompt trong server code:**
Edit `gemini_mcp_server.py`, function `_build_adversarial_prompt()` hoặc `_build_code_review_prompt()`

## So sánh với Claude-based Review

### Ưu điểm Gemini (via MCP):
- ✅ **Rẻ hơn**: Gemini 2.5 Pro ~60% rẻ hơn Claude Sonnet
- ✅ **Context lớn hơn**: 1M tokens vs 200K tokens
- ✅ **Flexible**: Dễ dàng switch giữa models
- ✅ **Standard protocol**: MCP là standard, portable
- ✅ **Independent**: Không depend vào Anthropic infrastructure

### Nhược điểm:
- ❌ **Output quality**: Claude thường tốt hơn cho complex reasoning
- ❌ **Setup complexity**: Cần MCP server + dependencies
- ❌ **JSON parsing**: Cần robust JSON parsing (Gemini output không consistent như Claude)

## Integration với BMad Workflow

### Method 1: Replace built-in review (aggressive)

Edit skill step files để thay thế built-in review với MCP tool:

**Before:**
```markdown
## Step 2: Adversarial Analysis
Review with extreme skepticism... Find at least ten issues...
```

**After:**
```markdown
## Step 2: Adversarial Analysis

CALL TOOL: `gemini_review_adversarial`
- content: {content}
- content_type: {content_type}
- also_consider: {also_consider}

Parse tool output and present findings to user.
```

### Method 2: Parallel review (conservative)

Chạy cả Claude và Gemini reviews, so sánh results:

```markdown
## Step 2: Adversarial Analysis

2.1: Run Claude-based review (built-in)
2.2: Run Gemini-based review (MCP tool)
2.3: Compare and merge findings
2.4: Present combined results with attribution
```

### Method 3: Fallback model

Sử dụng Gemini khi Claude unavailable hoặc rate-limited:

```markdown
## Step 2: Adversarial Analysis

TRY: Claude-based review
CATCH rate_limit/error:
  CALL TOOL: `gemini_review_adversarial` as fallback
```

## Performance Benchmarks

**Rough timing cho 1000-line code review:**

| Model | Time | Cost (approx) | Quality |
|-------|------|---------------|---------|
| Claude Sonnet 4.6 | ~30s | $0.45 | ⭐⭐⭐⭐⭐ |
| Gemini 2.5 Pro | ~45s | $0.18 | ⭐⭐⭐⭐ |
| Gemini 2.0 Flash | ~15s | FREE | ⭐⭐⭐ |
| Gemini 1.5 Pro | ~25s | $0.01 | ⭐⭐⭐ |

**Recommendation:**
- Production quality: Gemini 2.5 Pro
- Fast iteration: Gemini 2.0 Flash
- Budget-constrained: Gemini 2.0 Flash hoặc 1.5 Pro

## Advanced Usage

### Batch review với concurrency

```python
import asyncio
from mcp.client import Client

async def review_batch(files):
    server_params = StdioServerParameters(...)
    async with Client(server_params) as client:
        tasks = [
            client.call_tool("gemini_review_adversarial", {
                "content": open(f).read(),
                "content_type": "code"
            })
            for f in files
        ]
        results = await asyncio.gather(*tasks)
        return results

results = asyncio.run(review_batch(["file1.py", "file2.py", "file3.py"]))
```

### Custom prompt engineering

Edit `gemini_mcp_server.py` để custom prompts theo project needs:

```python
def _build_adversarial_prompt(self, content, content_type, also_consider):
    prompt = f"""# {project_name} Review Standards

You are reviewing code for {project_name}. Follow these standards:

## Critical Rules:
- All database queries MUST use parameterized queries
- All user input MUST be validated and sanitized
- All functions MUST have error handling

## Content:
{content}

... (rest of prompt)
"""
    return prompt
```

## Security Considerations

⚠️ **Important security notes:**

1. **API Key exposure**: `GOOGLE_GEMINI_API_KEY` trong `.mcp.json` visible cho team members với repo access
   - **Fix**: Use `${GOOGLE_GEMINI_API_KEY}` env var reference, không hardcode key

2. **Code leakage**: Review content sent to Google servers
   - **Check**: Company policy về sending code to external services
   - **Alternative**: Self-host Gemini model nếu có sensitive code

3. **Rate limiting**: Shared API key có thể bị rate limit bởi team members khác
   - **Fix**: Use separate API keys cho mỗi developer hoặc service account

## Monitoring & Observability

### Log API usage

Add logging vào `gemini_mcp_server.py`:

```python
import logging

logging.basicConfig(filename='gemini_mcp.log', level=logging.INFO)
logger = logging.getLogger(__name__)

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict[str, Any]):
    logger.info(f"Tool called: {name}, args: {list(arguments.keys())}")

    result = await actual_review(...)

    logger.info(f"Tokens used: {result.get('usage', {})}")
    return result
```

### Track costs

```python
# Pricing (2026)
GEMINI_PRICING = {
    "gemini-2.5-pro": {"input": 1.25, "output": 5.0},
    "gemini-1.5-pro": {"input": 0.075, "output": 0.30},
    "gemini-2.0-flash-exp": {"input": 0, "output": 0}  # Free
}

def calculate_cost(usage, model):
    input_cost = usage["prompt_tokens"] * GEMINI_PRICING[model]["input"] / 1_000_000
    output_cost = usage.get("completion_tokens", 0) * GEMINI_PRICING[model]["output"] / 1_000_000
    return input_cost + output_cost
```

## Kết luận

Gemini MCP Server cung cấp cách:
- ✅ Standardized để integrate Gemini model vào BMad workflow
- ✅ Flexible để switch giữa models
- ✅ Cost-effective so với Claude
- ✅ Portable qua MCP protocol

**Next steps:**
1. Install dependencies: `pip install -r requirements-gemini-mcp.txt`
2. Set API key: `export GOOGLE_GEMINI_API_KEY="..."`
3. Test MCP server: `python gemini_mcp_server.py`
4. Restart Claude Code
5. Use tools trong conversation hoặc integrate vào skills

---

**Documentation version:** 1.0.0
**Last updated:** 2026-06-21
**Sources:**
- [MCP Python SDK - GitHub](https://github.com/modelcontextprotocol/python-sdk)
- [Getting Started with Local MCP Servers on Claude Desktop](https://support.claude.com/en/articles/10949351-getting-started-with-local-mcp-servers-on-claude-desktop)
- [Practical Guide to MCP in Python](https://dev.to/m_sea_bass/practical-guide-to-mcp-model-context-protocol-in-python-ijd)
