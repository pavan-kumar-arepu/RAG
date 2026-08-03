# Group138 – LLM Assignment 2B | Progress Tracker

**Last updated:** 2026-07-29  
**Status:** 🟡 In Progress — scaffolding complete, execution pending

---

## Overall Status

| Phase | Description | Files Created | Executed | Notes |
|-------|-------------|:-------------:|:--------:|-------|
| 0 | Project setup & structure | ✅ | ✅ | Folders + corpus copied |
| 1 | Requirements | ✅ | ⬜ | `requirements.txt` ready |
| 2 | Document Loading | ✅ | ⬜ | `src/utils.py` |
| 3 | Text Cleaning | ✅ | ⬜ | `src/utils.py` |
| 4 | Chunking (3 strategies) | ✅ | ⬜ | `src/chunking.py` |
| 5 | Embeddings | ✅ | ⬜ | `src/embeddings.py` |
| 6 | FAISS Dense Retrieval | ✅ | ⬜ | `src/retrieval.py` |
| 7 | BM25 Sparse Retrieval | ✅ | ⬜ | `src/retrieval.py` |
| 8 | Hybrid RRF | ✅ | ⬜ | `src/retrieval.py` |
| 9 | Cross-Encoder Re-ranking | ✅ | ⬜ | `src/reranker.py` |
| 10 | Prompt Construction | ✅ | ⬜ | `src/prompt.py` |
| 11 | LLM Generation | ✅ | ⬜ | `main.py` (Ollama / HF) |
| 12 | Evaluation | ✅ | ⬜ | `src/evaluation.py` |
| 13 | Tabular RAG | ✅ | ⬜ | `src/tabular_rag.py` |
| — | Jupyter Notebook | ✅ | ⬜ | `notebooks/assignment_2b.ipynb` |

Legend: ✅ Done | ⬜ Pending | 🔄 In Progress | ❌ Blocked

---

## Project Structure (as created)

```
Group138_LLM_Assignment_2B/
├── data/
│   ├── corpus/                  ← 10 TXT files copied from Assignment 1B
│   └── pdfs/                    ← (create manually for Phase 13)
├── notebooks/
│   └── assignment_2b.ipynb      ← Main notebook (13 phases)
├── src/
│   ├── __init__.py
│   ├── utils.py                 ← Phase 2 & 3 (load + clean)
│   ├── chunking.py              ← Phase 4 (fixed, sliding, semantic)
│   ├── embeddings.py            ← Phase 5 (all-MiniLM-L6-v2)
│   ├── retrieval.py             ← Phase 6–8 (FAISS, BM25, Hybrid RRF)
│   ├── reranker.py              ← Phase 9 (cross-encoder)
│   ├── prompt.py                ← Phase 10 (prompt template)
│   ├── evaluation.py            ← Phase 12 (metrics)
│   └── tabular_rag.py           ← Phase 13 (PDF table RAG)
├── indexes/                     ← FAISS + BM25 saved here after first run
├── evaluation/                  ← chunking_report.json saved here
├── outputs/                     ← eval_results.json + plots saved here
├── reports/                     ← Final report (to be written)
├── main.py                      ← CLI orchestrator (all phases)
├── requirements.txt
├── Group138_LLM_Assignment2B_Implementation_Guide.md
└── ProgressTracker.md           ← This file
```

---

## Next Steps (when you return)

### Step 1 – Create & Activate Virtual Environment
```bash
cd "Group138_LLM_Assignment_2B"
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
pip install -r requirements.txt
```

### Step 2 – Verify corpus is present
```bash
ls data/corpus/    # should list 10 .txt files
```

### Step 3 – Run the notebook cell by cell
Open `notebooks/assignment_2b.ipynb` in VS Code or JupyterLab.  
Run phases 0 → 12 sequentially.  
Phase 13 requires PDFs – skip if not available.

### Step 4 – Configure your LLM (Phase 11)

**Option A – Ollama (recommended for local use)**
```bash
brew install ollama
ollama serve                     # start server in a separate terminal
ollama pull llama3               # or gemma2, mistral, etc.
```
No environment variable changes needed (defaults to `llama3`).

**Option B – HuggingFace**
```bash
export LLM_PROVIDER=hf
export HF_MODEL=google/gemma-2b-it
```
Requires ~5 GB disk space and a GPU / 16 GB RAM.

### Step 5 – Run CLI pipeline
```bash
python main.py
```
First run builds indexes → saves to `indexes/`.  
Subsequent runs load from disk (faster).

### Step 6 – Review outputs
| File | Description |
|------|-------------|
| `evaluation/chunking_report.json` | Chunking strategy comparison |
| `outputs/chunking_comparison.png` | Bar chart of chunking metrics |
| `outputs/eval_results.json` | Per-question RAG evaluation |
| `outputs/eval_metrics.png` | Latency + coverage chart |

### Step 7 – Write Final Report
- Save to `reports/Group138_Assignment2B_Report.pdf`
- Include: chunking comparison table, retrieval metrics, FAISS vs BM25 vs Hybrid comparison, before/after re-ranking, sample Q&A, Tabular RAG demo

---

## Key Design Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Embedding model | `all-MiniLM-L6-v2` | Fast, small (80 MB), 384-dim, excellent zero-shot |
| FAISS index type | `IndexFlatIP` (cosine) | Exact search sufficient for small corpus (10 docs) |
| Hybrid fusion | RRF (k=60) | Standard, no tuning required, robust to score scale |
| Cross-encoder | `ms-marco-MiniLM-L-6-v2` | Fast re-ranker, MSMARCO-trained |
| LLM | Ollama default | No cloud dependency; swap model by env var |
| Chunking default | Semantic (paragraph) | Lowest broken-sentence %; coherent context blocks |

---

## Dependency on Assignment 1B

| 1B Artefact | Used in 2B | How |
|-------------|-----------|-----|
| `03_Domain_Corpus/*.txt` | ✅ | Copied to `data/corpus/` – this IS the knowledge base |
| `04_QLora_Output/final_adapter/` | Optional | Can load fine-tuned model as LLM in Phase 11 |
| `02_Dataset/instruction_dataset.jsonl` | Optional | Can use Q-A pairs as evaluation references |

---

## Blockers / Issues

_None currently. Update this section when you resume work._

---

## Session Log

| Date | Action |
|------|--------|
| 2026-07-29 | Initial scaffolding created: full folder structure, corpus copied, all src/ modules, main.py, notebook, requirements.txt |
| 2026-07-29 | `.venv` created with Python 3.11.4 (`/usr/local/bin/python3.11`) |
| 2026-07-29 | All core packages installed successfully (see Installed Packages below) |
| 2026-07-29 | `faiss-cpu` upgraded to 1.9.0 for NumPy 2.x compatibility |
| 2026-07-29 | `pdfplumber` pinned to 0.10.4 to avoid pypdfium2 source-build issue |
| 2026-07-29 | **Paused** — imports not fully verified yet; resume with verification step |

---

## Installed Packages (as of 2026-07-29)

| Package | Version | Notes |
|---------|---------|-------|
| Python | 3.11.4 | via `/usr/local/bin/python3.11` |
| faiss-cpu | 1.9.0 | pre-built wheel, NumPy 2.x compatible |
| sentence-transformers | 5.6.1 | |
| torch | 2.2.2 | CPU only (macOS Intel) |
| transformers | 5.14.1 | |
| rank-bm25 | 0.2.2 | |
| scikit-learn | 1.9.0 | |
| numpy | 2.4.6 | |
| pandas | 3.0.5 | |
| pdfplumber | 0.10.4 | pinned — avoids pypdfium2 build issue |
| accelerate | 1.14.0 | |
| huggingface-hub | 1.25.1 | |
| rouge-score | 0.1.2 | |
| matplotlib | 3.11.1 | |
| seaborn | 0.13.2 | |
| jupyter | 1.1.1 | |
| bitsandbytes | ❌ skipped | macOS/CPU not supported; Linux+CUDA only |

---

## Resume Checklist

When you come back, run this first to confirm everything works:
```bash
cd "Group138_LLM_Assignment_2B"
.venv/bin/python -c "
import faiss, sentence_transformers, rank_bm25, transformers, torch, pdfplumber
import numpy, pandas, matplotlib
print('faiss:', faiss.__version__)
print('sentence-transformers:', sentence_transformers.__version__)
print('torch:', torch.__version__)
print('All OK')
"
```
If that passes, you're ready to run the notebook.
