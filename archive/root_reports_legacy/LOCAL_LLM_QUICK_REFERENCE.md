# Local LLM MCP Server - Quick Reference Card

**Quick commands and configurations for daily use**

---

## 🚀 Quick Start (3 Commands)

```bash
# 1. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. Pull model (choose one)
ollama pull qwen2.5-coder:32b  # Best quality (32GB RAM)
ollama pull qwen2.5-coder:7b   # Faster (16GB RAM)

# 3. Test
ollama run qwen2.5-coder:32b "Review this code: def foo(): return 1/0"
```

---

## 📝 Daily Usage

### **In Claude Code:**

```
Review this code using local LLM:
[paste code]
```

### **Explicit Tool Call:**

```
Call local_review_adversarial with:
- content: "def process(user_id): db.execute(f'UPDATE...{user_id}')"
- also_consider: ["Security", "SQL Injection"]
```

---

## 🔧 Configuration

### **.mcp.json (Current Config)**

```json
{
  "local-coder-review": {
    "command": "python",
    "args": ["D:\\bmad-projects\\stock_vol_prediction01\\local_coder_mcp_server.py"],
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

### **Switch Models (Edit .mcp.json)**

```json
"LLM_MODEL": "qwen2.5-coder:7b"     // Faster, 16GB RAM
"LLM_MODEL": "qwen2.5-coder:32b"    // Best quality, 32GB RAM
"LLM_MODEL": "deepseek-coder-v2:16b" // Alternative
```

### **Adjust Quality**

```json
"LLM_TEMPERATURE": "0.1"   // More deterministic (better quality)
"LLM_TEMPERATURE": "0.5"   // Balanced
"LLM_TEMPERATURE": "0.7"   // More creative (worse for code review)
```

---

## 🧪 Testing

### **Quick Test (1 minute)**
```bash
python test_local_llm_mcp.py
```

### **Manual Test (2 minutes)**
```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Test
python -c "
import ollama
response = ollama.chat(model='qwen2.5-coder:32b', messages=[
    {'role': 'user', 'content': 'Review this code: def foo(): return 1/0'}
])
print(response['message']['content'])
"
```

---

## 🔍 Troubleshooting

### **Problem:** Tools not available
```bash
# Solution 1: Check Ollama
ollama list

# Solution 2: Restart Claude Code

# Solution 3: Check .mcp.json
cat .mcp.json
```

### **Problem:** "Model not found"
```bash
# Solution: Pull model
ollama pull qwen2.5-coder:32b
```

### **Problem:** "Out of memory"
```bash
# Solution: Use smaller model
ollama pull qwen2.5-coder:7b
```

### **Problem:** Poor quality
```bash
# Solution 1: Use larger model
ollama pull qwen2.5-coder:32b

# Solution 2: Lower temperature
# Edit .mcp.json: "LLM_TEMPERATURE": "0.1"
```

---

## 📊 Performance

| Model | Time | Quality | RAM |
|-------|------|---------|-----|
| qwen2.5-coder:7b | 3-5s | 75% Claude | 16GB |
| qwen2.5-coder:32b | 8-12s | 85% Claude | 32GB |
| DeepSeek API | 5-8s | 92% Claude | - |

---

## 🎯 Model Selection Guide

| Scenario | Model | Why |
|----------|-------|-----|
| **Daily development** | qwen2.5-coder:32b | Best quality/price |
| **Quick testing** | qwen2.5-coder:7b | Fast enough |
| **Production** | DeepSeek V4 API | Best quality |
| **Limited RAM** | qwen2.5-coder:7b | Runs on 16GB |
| **Best quality** | DeepSeek V4-Pro | 92% of Claude |

---

## 🔗 Quick Links

- **Full Setup Guide:** [LOCAL_LLM_MCP_SETUP_GUIDE.md](LOCAL_LLM_MCP_SETUP_GUIDE.md)
- **Quick Start:** [QUICK_START_LOCAL_LLM.md](QUICK_START_LOCAL_LLM.md)
- **Research:** [OPEN_SOURCE_CODE_REVIEW_RESEARCH.md](OPEN_SOURCE_CODE_REVIEW_RESEARCH.md)
- **Ollama Models:** https://ollama.com/library
- **Qwen Models:** https://ollama.com/library/qwen2.5-coder

---

## 💡 Tips

1. **First time?** Start with qwen2.5-coder:7b (faster to pull and test)
2. **Production?** Use qwen2.5-coder:32b or DeepSeek API
3. **GPU available?** vLLM deployment is fastest
4. **Quality issues?** Lower temperature to 0.1
5. **Slow?** Reduce max_tokens to 4096

---

## ⌨️ Keyboard Shortcuts (Claude Code)

```
Ctrl+Shift+P    Command palette
/               Search commands
Ctrl+P          Quick open file
```

---

## 📞 Support

- **Troubleshooting:** See LOCAL_LLM_MCP_SETUP_GUIDE.md
- **Model Research:** See OPEN_SOURCE_CODE_REVIEW_RESEARCH.md
- **Issues:** Check test output first
- **Community:** https://discord.gg/ollama

---

**Print this page for quick reference! 🖨️**

---

**Version:** 1.0  
**Size:** 1 page (print-friendly)  
**Last Updated:** 2026-06-21
