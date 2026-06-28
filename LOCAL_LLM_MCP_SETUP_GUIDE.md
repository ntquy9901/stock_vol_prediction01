# Local LLM MCP Server - Installation & Setup Guide

**Complete guide to integrate local open-source LLMs với BMad workflow**

---

## 📋 Prerequisites

### **Hardware Requirements:**

**Option 1: 16GB RAM (Minimum)**
- Qwen2.5-Coder 7B
- DeepSeek Coder V2 Lite 7B
- Good for: Testing, quick tasks

**Option 2: 32GB RAM (Recommended)** ⭐
- Qwen2.5-Coder 32B
- Good for: Daily development, production

**Option 3: 64GB+ RAM + GPU (Optimal)**
- Full precision models
- vLLM deployment
- Good for: Production, heavy tasks

### **Software Requirements:**
- Python 3.8+
- Ollama (for local models)
- Claude Code (for MCP integration)

---

## 🚀 Installation Steps

### **Step 1: Install Ollama (5 minutes)**

#### **Linux/Mac:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

#### **Windows (PowerShell - Run as Administrator):**
```powershell
iwr -useb https://ollama.com/install.ps1 | iex
```

#### **Verify Installation:**
```bash
ollama --version
# Should show version number
```

---

### **Step 2: Pull Qwen2.5-Coder Model (10-20 minutes)**

#### **Option 2A: Qwen2.5-Coder 32B (Recommended)**
```bash
ollama pull qwen2.5-coder:32b
# Size: ~19GB
# RAM: 32GB
# Performance: 92.7% HumanEval (beats GPT-4)
```

#### **Option 2B: Qwen2.5-Coder 7B (Faster)**
```bash
ollama pull qwen2.5-coder:7b
# Size: ~4GB
# RAM: 16GB
# Performance: 88.4% HumanEval
```

#### **Option 2C: DeepSeek Coder V2 (Alternative)**
```bash
ollama pull deepseek-coder-v2:16b
# Size: ~9GB
# RAM: 16GB
# Performance: Competitive
```

#### **Verify Model:**
```bash
ollama list
# Should show downloaded models
```

---

### **Step 3: Test Ollama Model (2 minutes)**

```bash
# Start interactive mode
ollama run qwen2.5-coder:32b

# Test prompt (inside Ollama):
Review this Python code for bugs:

def process_payment(user_id, amount):
    query = f"UPDATE accounts SET balance = balance - {amount} WHERE user_id = {user_id}"
    db.execute(query)
    return True

# Press Ctrl+D to exit
```

**Expected output:** Model should identify SQL injection vulnerability and lack of input validation.

---

### **Step 4: Install Python Dependencies (2 minutes)**

```bash
# Install MCP server dependencies
pip install -r requirements-local-llm.txt
```

**Dependencies:**
- `mcp>=0.9.0` - MCP SDK
- `ollama>=0.1.0` - Ollama client
- `openai>=1.0.0` - OpenAI-compatible client (for vLLM, DeepSeek API)
- `asyncio>=3.4.3` - Async support

---

### **Step 5: Configure MCP Server (2 minutes)**

Check `.mcp.json` configuration:

```json
{
  "local-coder-review": {
    "command": "python",
    "args": [
      "D:\\bmad-projects\\stock_vol_prediction01\\local_coder_mcp_server.py"
    ],
    "env": {
      "LLM_RUNTIME": "ollama",
      "LLM_MODEL": "qwen2.5-coder:32b",
      "LLM_BASE_URL": "http://localhost:11434",
      "LLM_TEMPERATURE": "0.3",
      "LLM_MAX_TOKENS": "8192"
    }
  }
}
```

**Customize if needed:**
- Change `LLM_MODEL` to use different model (e.g., `qwen2.5-coder:7b`)
- Change `LLM_TEMPERATURE` (0.0-1.0, lower = more deterministic)
- Change `LLM_MAX_TOKENS` (max response length)

---

### **Step 6: Test MCP Server (5 minutes)**

```bash
# Terminal 1: Start Ollama (if not already running)
ollama serve

# Terminal 2: Run test suite
python test_local_llm_mcp.py
```

**Expected output:**
```
======================================================================
Local LLM MCP Server Test Suite
======================================================================

[TEST 1] Checking dependencies...
  ✓ mcp installed
  ✓ ollama installed
  ✓ openai installed

[TEST 2] Testing Ollama connection...
  ✓ Ollama is running
  ✓ Found X models
  ✓ Found Qwen Coder models:
    - qwen2.5-coder:32b (19.2GB)

[TEST 3] Testing Ollama code review...
  ✓ Ollama response received

[TEST 6] Testing adversarial review (this may take 10-30s)...
  ✓ Adversarial review completed
    Findings: 5 issues found

✓ MCP Server is ready!
```

---

### **Step 7: Restart Claude Code (1 minute)**

1. **Close Claude Code completely** (ensure no processes running)
2. **Restart Claude Code**
3. **Open your project**

**Verify MCP server loaded:**
- MCP tools should be available
- Check Claude Code settings → MCP servers

---

## 🎯 Usage

### **Option 1: Direct in Claude Code (Simplest)**

```
Please review this code using local LLM:

def calculate_volatility(prices):
    # No input validation!
    return prices.std() * (252 ** 0.5)
```

**Or explicit tool call:**
```
Call local_review_adversarial with:
- content: "def calculate_volatility(prices): return prices.std() * (252 ** 0.5)"
- content_type: "code"
- also_consider: ["Error Handling", "Input Validation"]
```

---

### **Option 2: Integrate vào BMad Skills**

#### **Method A: Replace Built-in Review (Aggressive)**

Edit skill step file (e.g., `.claude/skills/bmad-code-review/steps/step-02-review.md`):

**Before:**
```markdown
## Step 2: Adversarial Analysis

Review with extreme skepticism... Find at least ten issues...
```

**After:**
```markdown
## Step 2: Adversarial Analysis

CALL TOOL: local_review_adversarial
- content: {content}
- content_type: {content_type}
- also_consider: {also_consider}

Parse tool output and present findings to user.
```

---

#### **Method B: Parallel Review (Conservative)**

```markdown
## Step 2: Adversarial Analysis

2.1: Run Claude-based review (built-in)
2.2: Run local LLM review (local_review_adversarial)
2.3: Compare and merge findings
2.4: Present combined results with attribution
```

---

#### **Method C: Fallback Model**

```markdown
## Step 2: Adversarial Analysis

TRY: Claude-based review
CATCH rate_limit/error:
  CALL TOOL: local_review_adversarial as fallback
```

---

## 🔧 Advanced Configuration

### **Switch Between Models**

Edit `.mcp.json`:

```json
{
  "env": {
    "LLM_MODEL": "qwen2.5-coder:7b"  // Use 7B model
  }
}
```

**Available models:**
- `qwen2.5-coder:32b` (best quality, 32GB RAM)
- `qwen2.5-coder:7b` (faster, 16GB RAM)
- `deepseek-coder-v2:16b` (alternative)
- `codellama:13b` (older model)

---

### **Use vLLM Instead of Ollama (Production)**

#### **Install vLLM:**
```bash
pip install vllm
```

#### **Start vLLM Server:**
```bash
vllm serve Qwen/Qwen2.5-Coder-32B-Instruct --port 8000
```

#### **Update .mcp.json:**
```json
{
  "env": {
    "LLM_RUNTIME": "vllm",
    "LLM_MODEL": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "LLM_BASE_URL": "http://localhost:8000/v1"
  }
}
```

---

### **Use DeepSeek API (Cloud Alternative)**

#### **Update .mcp.json:**
```json
{
  "env": {
    "LLM_RUNTIME": "openai",
    "LLM_MODEL": "deepseek-chat",
    "LLM_BASE_URL": "https://api.deepseek.com",
    "LLM_API_KEY": "your-deepseek-api-key"
  }
}
```

**Pricing:** ~$0.43/1M tokens (V4 Flash)

---

## 📊 Performance Optimization

### **Reduce Inference Time**

**Method 1: Use Smaller Model**
```bash
# Use 7B instead of 32B
ollama pull qwen2.5-coder:7b
```

**Method 2: Adjust Max Tokens**
```json
{
  "env": {
    "LLM_MAX_TOKENS": "4096"  // Instead of 8192
  }
}
```

**Method 3: Use GPU (if available)**
```bash
# Ollama will automatically use GPU if available
# Check GPU usage:
nvidia-smi
```

---

### **Improve Review Quality**

**Method 1: Lower Temperature**
```json
{
  "env": {
    "LLM_TEMPERATURE": "0.1"  // More deterministic
  }
}
```

**Method 2: Use Larger Model**
```bash
ollama pull qwen2.5-coder:32b
```

**Method 3: Add Persistent Facts**
```
Call local_review_code with:
- persistent_facts: [
    "All code must follow PEP 8",
    "Security is critical - check for SQL injection, XSS, CSRF",
    "Performance matters - optimize hot paths"
  ]
```

---

## 🐛 Troubleshooting

### **Issue: "MCP server not starting"**

**Symptoms:**
- Tools not available in Claude Code
- Error: "MCP server failed to start"

**Solutions:**
```bash
# 1. Check Ollama is running
ollama list

# 2. Check MCP server config
cat .mcp.json

# 3. Test MCP server manually
python local_coder_mcp_server.py

# 4. Check Claude Code logs
# Settings → Developer → View Logs
```

---

### **Issue: "Model not found"**

**Symptoms:**
- Error: "model 'qwen2.5-coder:32b' not found"

**Solutions:**
```bash
# 1. List available models
ollama list

# 2. Pull correct model
ollama pull qwen2.5-coder:32b

# 3. Or use smaller model
ollama pull qwen2.5-coder:7b

# 4. Update .mcp.json with correct model name
```

---

### **Issue: "Out of memory"**

**Symptoms:**
- System slows down
- Error: "out of memory"

**Solutions:**
```bash
# 1. Use smaller model
ollama pull qwen2.5-coder:7b

# 2. Close other applications
# 3. Upgrade RAM (recommended: 32GB)
# 4. Use GPU (offloads CPU)
```

---

### **Issue: "Poor review quality"**

**Symptoms:**
- Model misses obvious bugs
- Generic responses

**Solutions:**
```bash
# 1. Use larger model
ollama pull qwen2.5-coder:32b

# 2. Lower temperature
# Edit .mcp.json: "LLM_TEMPERATURE": "0.1"

# 3. Improve prompts
# Add persistent_facts with context

# 4. Use DeepSeek API instead
# Better quality, affordable
```

---

## 📈 Benchmarking

### **Test Script**

```python
import time
import ollama

code_to_review = """def process(data):
    return data.sort()
"""

start = time.time()
response = ollama.chat(
    model='qwen2.5-coder:32b',
    messages=[{'role': 'user', 'content': f'Review: {code_to_review}'}]
)
end = time.time()

print(f"Inference time: {end - start:.2f}s")
print(f"Response: {response['message']['content']}")
```

---

### **Compare Models**

```bash
# Test 7B model
time ollama run qwen2.5-coder:7b "Review: [code]"

# Test 32B model
time ollama run qwen2.5-coder:32b "Review: [code]"
```

**Expected Results:**
- 7B: 3-5 seconds, ~75% Claude quality
- 32B: 8-12 seconds, ~85% Claude quality

---

## 🎓 Best Practices

### **1. Model Selection**

| Use Case | Recommended Model | Why |
|----------|-------------------|-----|
| Daily development | Qwen2.5-Coder 32B | Best quality/price ratio |
| Quick testing | Qwen2.5-Coder 7B | Fast, good enough |
| Production | vLLM + 32B | Reliable, scalable |
| Resource-constrained | 7B or DeepSeek 7B | Runs on 16GB RAM |

---

### **2. Prompt Engineering**

**DO:**
- ✅ Provide context (what the code does)
- ✅ Specify what to look for (security, performance, etc.)
- ✅ Use persistent_facts for standards

**DON'T:**
- ❌ Ask vague questions ("is this good?")
- ❌ Skip context
- ❌ Expect architectural understanding from 7B models

---

### **3. Integration with BMad**

**Recommended Approach:**

1. **Start with manual testing:**
   ```
   Use local LLM to review this code
   ```

2. **Integrate into one skill:**
   - Edit bmad-code-review skill
   - Add local_review_adversarial call

3. **Expand to other skills:**
   - bmad-review-edge-case-hunter
   - bmad-review-adversarial-general

4. **Compare results:**
   - Run same code with Claude vs local
   - Measure quality difference
   - Decide on integration strategy

---

## 📚 Resources

### **Documentation:**
- [Ollama Official Docs](https://ollama.com/docs)
- [Qwen2.5-Coder Paper](https://arxiv.org/abs/2309.15209)
- [vLLM Documentation](https://docs.vllm.ai/)
- [MCP Protocol](https://modelcontextprotocol.io/)

### **Community:**
- [Ollama Discord](https://discord.gg/ollama)
- [Qwen GitHub](https://github.com/QwenLM/Qwen)
- [Local LLM Reddit](https://reddit.com/r/LocalLLaMA)

### **Related:**
- [OPEN_SOURCE_CODE_REVIEW_RESEARCH.md](OPEN_SOURCE_CODE_REVIEW_RESEARCH.md) - Comprehensive model research
- [QUICK_START_LOCAL_LLM.md](QUICK_START_LOCAL_LLM.md) - Quick start guide
- [gemini_mcp_server.py](gemini_mcp_server.py) - Gemini MCP server (reference)

---

## ✅ Checklist

Before using in production:

- [ ] Ollama installed and running
- [ ] Model pulled (test with `ollama run`)
- [ ] Dependencies installed (`pip install -r requirements-local-llm.txt`)
- [ ] MCP server tested (`python test_local_llm_mcp.py`)
- [ ] .mcp.json configured
- [ ] Claude Code restarted
- [ ] Tools available in Claude Code
- [ ] Quality validated (compare with Claude)
- [ ] Integrated into BMad skills (if applicable)
- [ ] Documentation updated for team

---

**Version:** 1.0  
**Last Updated:** 2026-06-21  
**Author:** Claude Code Assistant  

**Support:**
- Issues: Check troubleshooting section
- Questions: See resources section
- BMad Integration: See BMad workflow section

**Next Steps:**
1. Run test suite: `python test_local_llm_mcp.py`
2. Test in Claude Code with sample code
3. Compare quality with Claude
4. Integrate into BMad skills
5. Deploy to team

---

**Happy code reviewing with local LLMs! 🚀**
