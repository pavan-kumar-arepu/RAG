# Group138 -- LLM Assignment 2B Implementation Guide

## Goal

Implement a complete Retrieval-Augmented Generation (RAG) pipeline using
the **same domain corpus** from Assignment 1.

## Existing Folder Structure

``` text
Group138_LLM_Assignment1B/
├── 01_Notebook/
├── 02_Dataset/
├── 03_Domain_Corpus/
└── 04_QLora_Output/

Group138_LLM_Assignment2B/
└── RAG_Assignment_2B_Master_Guide.pdf
```

## Step 0 -- Copy Corpus

Copy the complete folder:

``` text
Group138_LLM_Assignment1B/03_Domain_Corpus/
```

into

``` text
Group138_LLM_Assignment2B/data/corpus/
```

Expected structure:

``` text
Group138_LLM_Assignment2B/
│
├── data/
│   └── corpus/
├── notebooks/
├── src/
├── indexes/
├── evaluation/
├── outputs/
├── reports/
└── main.py
```

------------------------------------------------------------------------

# End-to-End RAG Flow

``` text
Corpus
   │
Cleaning
   │
Chunking
   │
Embeddings
   │
FAISS Index
   │
BM25 Index
   │
Hybrid Retrieval (RRF)
   │
Cross Encoder Re-ranking
   │
Prompt Construction
   │
LLM
   │
Evaluation
   │
Tabular RAG
```

------------------------------------------------------------------------

# Implementation Tasks

## Phase 1 -- Project Setup

-   Create Python virtual environment
-   Install dependencies
-   Create project structure

## Phase 2 -- Document Loading

-   Load all TXT files from `data/corpus`
-   Read UTF-8 safely
-   Store metadata (file name, title)

## Phase 3 -- Cleaning

-   Remove extra spaces
-   Remove page numbers
-   Remove repeated headers/footers
-   Normalise whitespace

Output: - `clean_documents`

## Phase 4 -- Chunking

Implement and compare: 1. Fixed chunking 2. Sliding window chunking 3.
Semantic chunking

Measure: - Number of chunks - Average chunk size - Standard deviation -
Broken sentence %

Select the best approach.

## Phase 5 -- Embeddings

Recommended: - sentence-transformers - all-MiniLM-L6-v2

Generate embedding for every chunk.

## Phase 6 -- Dense Retrieval

-   Build FAISS index
-   Save locally
-   Implement similarity search

## Phase 7 -- Sparse Retrieval

-   Build BM25 index
-   Implement keyword search

## Phase 8 -- Hybrid Retrieval

Combine: - FAISS - BM25

using Reciprocal Rank Fusion (RRF).

## Phase 9 -- Cross Encoder

-   Re-rank retrieved chunks
-   Compare before/after

## Phase 10 -- Prompt Construction

Template:

Context Question Instructions

Use retrieved chunks only.

## Phase 11 -- LLM

Connect to one model: - Llama - Gemma - Mistral - GPT

Generate grounded answers.

## Phase 12 -- Evaluation

Measure: - Latency - Recall - Precision - Coverage - Relevance

## Phase 13 -- Tabular RAG

-   Extract PDF tables
-   Convert rows into chunks
-   Embed
-   Retrieve
-   Answer

------------------------------------------------------------------------

# Suggested Source Files

``` text
src/
    chunking.py
    embeddings.py
    retrieval.py
    reranker.py
    prompt.py
    tabular_rag.py
    evaluation.py
    utils.py
main.py
```

------------------------------------------------------------------------

# Difference from Assignment 1

Assignment 1: - Build instruction dataset - Fine-tune model using
QLoRA - Produce adapted LLM

Assignment 2: - Keep LLM unchanged - Retrieve relevant knowledge -
Inject context into prompt - Generate grounded responses

------------------------------------------------------------------------

# Final Deliverables

-   Source code
-   Jupyter notebook
-   Evaluation results
-   Retrieval benchmarks
-   Chunking comparison
-   FAISS index
-   BM25 index
-   Final report
-   Viva preparation notes
