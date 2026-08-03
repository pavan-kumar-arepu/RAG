# ACTION PLAN - Fix Issues & Make Assignment Evidence-Based

**Status**: Pipeline 85% complete. Single blocker: Ollama LLM service.

---

## Problem

The RAG pipeline cannot generate answers because **Ollama is not installed or running**.

- All retrieval components work (FAISS, BM25, hybrid, re-ranking)
- Embedding generation works (all-MiniLM-L6-v2)
- LLM generation fails: Connection refused to localhost:11434

---

## Solution: 3-Step Fix

### Step 1: Install Ollama (5 min)

Ollama provides local LLM inference via HTTP API.

```bash
# macOS: Download and install from https://ollama.ai
# Or via brew:
brew install ollama

# Verify installation
ollama --version
```

### Step 2: Start Ollama Service (in background terminal)

```bash
# Terminal 1: Start the Ollama server
ollama serve

# Keep this running while testing
# Expected output:
#   Serving on 127.0.0.1:11434
```

### Step 3: Pull a Model (first time only)

```bash
# Terminal 2: Download LLM
ollama pull llama2
# or: ollama pull llama3
# or: ollama pull mistral

# This downloads the model to ~/.ollama/models/
# Takes 2-15 min depending on model and internet
```

---

## Then Re-Run Pipeline for Real Results

### Option A: Run Jupyter Notebook (Same as before)
```bash
# In notebook cell, re-execute Phase 11
# Should now see actual LLM answers instead of connection errors
```

### Option B: Run CLI Script (Faster for testing)
```bash
cd /Users/pavankumararepu/BitsPilani/2nd\ Year/AIMLCZG536_LLM/Assignment/Assignment_Solution/Group138_LLM_Assignment_2B

# Interactive Q&A with real LLM
python main.py

# Test with questions like:
# "What is transformer architecture?"
# "How does QLoRA reduce memory?"
# "What are challenges for low-resource NLP?"
```

### Option C: Run Evaluation Script (For metrics)
```bash
# Generate real evaluation metrics
python -c "
import sys; sys.path.insert(0, '.')
from src.evaluation import evaluate_rag
results = evaluate_rag(['outputs/eval_results.json'])
print(results)
"
```

---

## What This Will Fix

**Currently**:
- eval_results.json has SYNTHETIC answers
- Coverage scores are GUESSED
- Latencies don't include LLM time
- Can't prove pipeline works end-to-end

**After Ollama**:
- eval_results.json will have REAL answers from llama2/llama3
- Coverage scores computed from actual LLM output
- Real latencies (retrieval + LLM generation)
- Fully reproducible evaluation

---

## Example: What "Real Evidence" Looks Like

### Before (Synthetic - Current):
```json
{
  "question": "What is transformer architecture?",
  "answer": "[SYNTHETIC] The transformer architecture introduced by Vaswani...",
  "coverage": 0.67,
  "latency_s": 2.5,
  "retrieved_chunks": 3
}
```

### After (Real - With Ollama):
```json
{
  "question": "What is transformer architecture?",
  "answer": "[FROM LLAMA2] Transformers use self-attention to process sequences...",
  "coverage": 0.72,
  "latency_s": 4.2,
  "retrieved_chunks": 3,
  "llm_model": "llama2",
  "generation_time_s": 1.7
}
```

---

## Verification Checklist

After completing the 3 steps above, verify:

- [ ] `ollama serve` running without errors
- [ ] `ollama pull llama2` completed (4GB downloaded)
- [ ] `python main.py` connects to localhost:11434 successfully
- [ ] Test query returns actual answer (not error)
- [ ] Latency includes LLM generation (typically 1-3s per query)
- [ ] eval_results.json updated with real answers
- [ ] EXECUTION_EVIDENCE.md shows all 11 phases as ✓

---

## Timeline

| Step | Time | Status |
|------|------|--------|
| Install Ollama | 5 min | Not started |
| Start ollama serve | 1 min | Blocked |
| Pull llama2 model | 10 min | Blocked |
| Re-run pipeline | 5 min | Blocked |
| Verify results | 2 min | Blocked |
| **Total** | **~25 min** | **Ready to start** |

---

## Expected Outcome

✓ All 13 phases of RAG pipeline working end-to-end  
✓ Real, measured results (not synthetic)  
✓ Evidence-based evaluation metrics  
✓ Reproducible pipeline (anyone can run it)  
✓ No fancy cosmetics, just clear functionality

---

## If Issues Occur

### Ollama won't start
```bash
# Check if port 11434 is in use
lsof -i :11434

# Check logs
tail -f ~/.ollama/logs/
```

### Model download too slow
```bash
# Can use smaller model
ollama pull mistral  # 4.1 GB (faster)
ollama pull neural-chat  # 3.8 GB (faster)

# Or check internet connection
curl -I https://ollama.ai
```

### "Connection refused" error
```bash
# Verify Ollama is running
curl http://localhost:11434/api/tags

# If no response, ollama serve hasn't started
ollama serve
```

---

## What NOT to Do

❌ Don't modify test queries  
❌ Don't try to "fix" synthetic answers  
❌ Don't skip Ollama setup (it's required)  
❌ Don't use different model without testing (may need different prompt format)  

---

## Files to Check After Fix

```bash
# Updated with real results
outputs/eval_results.json

# Execution trace
test_pipeline.py

# This plan
ACTION_PLAN.md

# Updated evidence report
EXECUTION_EVIDENCE.md
```

---

## Summary

**You have**: Working retrieval pipeline (9 phases tested)  
**You need**: Ollama running (< 30 min setup)  
**You get**: Complete evidence-based RAG system that works

Ready to proceed? Start with: `brew install ollama`
