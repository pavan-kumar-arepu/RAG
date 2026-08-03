# Assignment 1B - Overall Flow Diagram

## Domain LLM Adaptation & Production Optimization

```mermaid
graph TD
    A["🎯 Assignment 1B<br/>Domain LLM Adaptation &<br/>Production Optimization"] -->|SmolLM2-360M| B["⚙️ Environment Setup<br/>Load Libraries & Config"]
    
    B --> C["📊 Part A: Domain Data Collection<br/>& Instruction Dataset"]
    
    C --> C1["Load Raw Corpus<br/>10 Scientific Papers"]
    C1 --> C2["Text Cleaning<br/>Remove noise, non-English"]
    C2 --> C3["Corpus Statistics<br/>Generate word/char counts"]
    C3 --> C4["📈 Baseline Testing<br/>SmolLM2-360M bfloat16"]
    
    C4 --> D["🔧 Part B: QLoRA<br/>Instruction Fine-Tuning"]
    
    D --> D1["Chunk Corpus<br/>300-word chunks"]
    D1 --> D2["Create Instructions<br/>5 Template Types"]
    D2 --> D3["Build JSONL Dataset<br/>80/20 Train/Eval"]
    D3 --> D4["⚡ QLoRA Training<br/>4-bit NF4 + r=16"]
    D4 --> D5["Train with SFT Trainer<br/>3 epochs, batch=16"]
    D5 --> D6["✅ Fine-Tuned Adapter<br/>Evaluate vs Baseline"]
    
    D6 --> E["🚀 Part C: Production<br/>Optimization"]
    
    E --> E1["🎲 Decoding Strategies<br/>5 Methods"]
    E1 --> E1a["Greedy Decode"]
    E1 --> E1b["Beam Search k=4"]
    E1 --> E1c["Top-K Sampling"]
    E1 --> E1d["Top-P Sampling"]
    E1 --> E1e["Temperature Sampling"]
    E1a --> E1_eval["Compare ROUGE-L<br/>& Throughput"]
    E1b --> E1_eval
    E1c --> E1_eval
    E1d --> E1_eval
    E1e --> E1_eval
    
    E1_eval --> E2["⚙️ Speculative Decoding<br/>135M Draft + 360M Target"]
    E2 --> E2_bench["Benchmark: Standard<br/>vs Speculative"]
    E2_bench --> E2_result["📊 Speedup Metrics"]
    
    E2_result --> E3["💾 Quantization Analysis<br/>bfloat16 vs 4-bit"]
    E3 --> E3_vram["VRAM: ~2GB → ~800MB"]
    E3 --> E3_thr["Throughput: Comparison"]
    E3_vram --> E3_cost
    E3_thr --> E3_cost["💰 Production Cost<br/>Per 1M tokens"]
    
    E3_cost --> F["✨ Outputs"]
    F --> F1["instruction_dataset.jsonl"]
    F --> F2["LoRA Adapter<br/>checkpoint-24/48/72"]
    F --> F3["Quantized Model<br/>Benchmarks"]
    
    style A fill:#FFE5B4
    style C fill:#B4D7FF
    style D fill:#D7FFB4
    style E fill:#FFB4D7
    style F fill:#E5D7FF
    style E1_eval fill:#FFFACD
    style E2_result fill:#FFFACD
    style E3_cost fill:#FFFACD
```

## Flow Overview

### Part A: Domain Data Collection & Instruction Dataset (2 marks)
- **Load Raw Corpus**: Collect 10 scientific research papers
- **Text Cleaning**: Remove noise, non-ASCII, and non-English content
- **Corpus Statistics**: Generate word/character/sentence counts
- **Baseline Testing**: Test base model (SmolLM2-360M bfloat16) on domain prompts

### Part B: QLoRA Instruction Fine-Tuning (5 marks)
- **Chunk Corpus**: Split documents into 300-word overlapping chunks
- **Create Instructions**: Generate 5 types of instruction templates
- **Build JSONL Dataset**: Create 80/20 train/eval split
- **QLoRA Training**: Fine-tune using 4-bit NF4 quantization with LoRA (r=16, alpha=32)
- **Train Adapter**: 3 epochs with batch size 16
- **Evaluate**: Compare fine-tuned vs baseline outputs using ROUGE metrics

### Part C: Production Optimization & Economics (8 marks)

#### C1: Decoding Strategies (3 marks)
Compare 5 decoding methods:
- Greedy decoding
- Beam search (k=4)
- Top-K sampling (k=50)
- Top-P sampling (p=0.9)
- Temperature sampling (0.7)

Evaluate throughput and ROUGE-L quality

#### C2: Speculative Decoding (2 marks)
- Draft model: SmolLM2-135M-Instruct
- Target model: SmolLM2-360M-Instruct
- Benchmark standard vs speculative speedup

#### C3: Quantization & Cost Analysis (3 marks)
- VRAM comparison: bfloat16 vs 4-bit NF4
- Throughput analysis
- Production cost per 1M tokens across multiple GPU instances

## Key Outputs
- `instruction_dataset.jsonl`: Training dataset with 80/20 split
- `qlora_output/`: Final adapter weights and checkpoints
- Benchmarks and cost analysis reports
