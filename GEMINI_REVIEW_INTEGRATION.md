# Gemini API Integration cho BMad Review Skills

## Tổng quan

Script `gemini_review_wrapper.py` cho phép sử dụng Google Gemini model thay vì Claude model cho các tác vụ review code trong BMad framework.

## Cài đặt

### 1. Cài đặt dependencies

```bash
pip install google-generativeai
```

Hoặc sử dụng file `requirements-gemini.txt`:

```bash
pip install -r requirements-gemini.txt
```

### 2. Cấu hình API Key

Tạo Google Gemini API key tại: https://makersuite.google.com/app/apikey

Sau đó cấu hình environment variable:

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

## Cách sử dụng

### Cách 1: CLI trực tiếp

```bash
# Review adversarial cho document/code
python gemini_review_wrapper.py adversarial path/to/file.txt

# Code review cho git diff
python gemini_review_wrapper.py code path/to/diff.txt full
python gemini_review_wrapper.py code path/to/diff.txt no-spec
```

### Cách 2: Từ Python script

```python
from gemini_review_wrapper import GeminiReviewer

# Khởi tạo reviewer
reviewer = GeminiReviewer(model="gemini-2.5-pro")

# Adversarial review
with open("my_code.py", "r") as f:
    code = f.read()

result = reviewer.review_adversarial(
    content=code,
    content_type="code",
    also_consider=["Security", "Performance", "Maintainability"]
)

print(f"Tìm thấy {len(result['findings'])} issues")
for finding in result['findings']:
    print(f"[{finding['severity']}] {finding['description']}")

# Code review cho diff
with open("changes.diff", "r") as f:
    diff = f.read()

review = reviewer.review_code(
    diff_content=diff,
    review_mode="full",
    persistent_facts=[
        "All code must follow PEP 8 style guide",
        "All functions must have docstrings",
        "Test coverage must be > 85%"
    ]
)

print(f"High: {review['summary']['high_count']}")
print(f"Medium: {review['summary']['medium_count']}")
print(f"Low: {review['summary']['low_count']}")
```

### Cách 3: Tích hợp với BMad workflow

Chỉnh sửa `customize.toml` của bmad-code-review skill:

```toml
[workflow]
# Thêm activation step để gọi Gemini script
activation_steps_prepend = [
  "python {project-root}/gemini_review_wrapper.py code {diff_file} full > gemini_review.json"
]

# Hoặc thay vì dùng built-in review, sử dụng Gemini output
on_complete = "python {project-root}/merge_gemini_review.py"
```

## Available Models

- `gemini-2.5-pro` - Mới nhất, mạnh nhất (recommended)
- `gemini-2.0-flash-exp` - Nhanh hơn, free tier
- `gemini-1.5-pro` - Phiên bản stable cũ hơn

## Output Format

### Adversarial Review Output
```json
{
  "model": "gemini-2.5-pro",
  "findings": [
    {
      "description": "Missing input validation for user_id parameter",
      "severity": "HIGH"
    },
    {
      "description": "Function lacks error handling for database connection failure",
      "severity": "MEDIUM"
    }
  ],
  "raw_response": "...",
  "usage": {...}
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
      "issue": "Race condition in update_balance",
      "location": "src/payment.py:45",
      "how_to_fix": "Add transaction lock or atomic operation"
    }
  ],
  "triage": {
    "high_priority": ["Finding 1", "Finding 3"],
    "medium_priority": ["Finding 2"],
    "low_priority": []
  },
  "summary": {
    "high_count": 2,
    "medium_count": 1,
    "low_count": 0,
    "total_findings": 3
  }
}
```

## So sánh với Claude-based Review

### Ưu điểm Gemini:
- ✅ **Rẻ hơn**: Gemini API có pricing tốt hơn Anthropic API
- ✅ **Nhanh hơn**: Flash model rất nhanh
- ✅ **Flexible**: Dễ dàng switch giữa models
- ✅ **Free tier**: Gemini 2.0 Flash experiment có free tier hào phóng

### Nhược điểm:
- ❌ **Context window nhỏ hơn**: Gemini 1M tokens vs Claude 200K tokens
- ❌ **Output quality**: Claude thường tốt hơn cho complex reasoning
- ❌ **Integration complexity**: Cần thêm wrapper layer

## Pricing (approximate, 2026)

| Model | Input | Output | Context |
|-------|-------|--------|---------|
| Gemini 2.5 Pro | $1.25/1M | $5/1M | 1M tokens |
| Gemini 2.0 Flash | Free (exp) | Paid | 1M tokens |
| Claude Sonnet 4.6 | $3/1M | $15/1M | 200K tokens |

## Troubleshooting

### API Key không hoạt động
```bash
# Test API key
python -c "import google.generativeai as genai; genai.configure(api_key='YOUR_KEY'); print(genai.GenerativeModel('gemini-2.5-pro').generate_content('test').text)"
```

### Quota exceeded
- Kiểm tra usage tại: https://makersuite.google.com/app/apikey
- Upgrade tier hoặc chờ đến ngày mai (free tier reset mỗi ngày)

### Response quality kém
- Thử model mạnh hơn: `gemini-2.5-pro` thay vì `gemini-1.5-pro`
- Tweak prompt trong `_build_adversarial_prompt()` hoặc `_build_code_review_prompt()`
- Thêm nhiều `persistent_facts` để cung cấp context

## Advanced Usage

### Tùy chỉnh prompt

Chỉnh sửa file `gemini_review_wrapper.py`:

```python
def _build_adversarial_prompt(self, content, content_type, also_consider):
    # Thêm custom instructions
    prompt += """

## Additional Rules:
- Focus on security vulnerabilities
- Check for SQL injection, XSS, CSRF
- Verify proper error handling
"""
    return prompt
```

### Batch review nhiều files

```python
import os
from gemini_review_wrapper import GeminiReviewer

reviewer = GeminiReviewer()

for root, dirs, files in os.walk("src/"):
    for file in files:
        if file.endswith(".py"):
            filepath = os.path.join(root, file)
            with open(filepath, "r") as f:
                code = f.read()

            result = reviewer.review_adversarial(code, content_type="code")

            # Save result
            output_file = filepath.replace(".py", "_review.json")
            with open(output_file, "w") as f:
                json.dump(result, f, indent=2)
```

## Tích hợp với Git Hooks

Tạo file `.git/hooks/pre-commit`:

```bash
#!/bin/bash

# Get staged files
STAGED_FILES=$(git diff --cached --name-only --diff-filter=ACM | grep '\.py$')

if [ -n "$STAGED_FILES" ]; then
    echo "Running Gemini code review on staged files..."

    for file in $STAGED_FILES; do
        echo "Reviewing $file..."
        git show ":$file" | python gemini_review_wrapper.py adversarial /dev/stdin > "$file.review.json"

        # Check for HIGH severity issues
        if grep -q '"severity": "HIGH"' "$file.review.json"; then
            echo "❌ HIGH severity issues found in $file. Review details in $file.review.json"
            exit 1
        fi
    done

    echo "✅ Review passed"
fi
```

## Kết luận

Script `gemini_review_wrapper.py` cung cấp cách thay thế Claude model bằng Gemini model cho các tác vụ review code trong BMad framework, với lợi thế về chi phí và tốc độ, trade-off về context window và integration complexity.
