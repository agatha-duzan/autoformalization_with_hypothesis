# Hypothesis Decomposition and Retrieval for Autoformalization

This repository contains the code and data for a semester project on **autoformalization** of discrete math statements in Lean&nbsp;4. 

The [`project report`](pdfs/Report.pdf) contains a detailed discussion of:

- The autoformalization problem and its challenges
- Related work in theorem proving and retrieval
- Our hypothesis decomposition and retrieval pipeline designs
- Experiments, results, and lessons learned

---

## Key Structure
- **`data/`**  
  Contains the curated datasets of informal–formal pairs, the retrieval corpora, and the code used to create them
  
- **`pdfs/`**  
  Final report and figures. Refer to this for a detailed explanation of the project’s methodology, experiments, and findings

- **`repl/`**  
  A submodule used for the type-checking evaluation

- **`results/`**  
  - `checkpoints/` – Important because the Lean server crashes easily, so we save our progress in evals  
  - `evaluations/` – Detailed evaluation results, including type-checking, TC-BLEU, and BEq scores  
  - `generations/` – Raw outputs from our different methods

## Usage

### 1. Configuration

All key settings—API keys, datasets, generation methods, and paths—are controlled in [`config.py`](./config.py).

- `METHOD = None` :  direct translation baseline
- `METHOD = 'informal_decomp'` : adds the informal hypothesis decomposition step
- `METHOD = 'leandojo'` : proof state-based retrieval
- `METHOD = 'leansearch'` : semantic search retrieval

You can further adjust:

- **`FEWSHOT`**: whether to include few-shot examples in the prompt  
- **`DATASET_NAME`**: Hugging Face dataset for evaluation  
- **`DEFAULT_MODEL`**: base model

### 2. Run the Generation

After editing `config.py` to your liking, you can generate formal statements by running:

```bash
python generation.py
```

### 3. Evaluate

Two evaluation steps are available:

1. **Type Checking & BLEU**  
   ```bash
   python eval.py
   ```

2. **BEq (Bidirectional Equivalence)**  
   ```bash
   python beq_eval_cpu.py
   ```
   Designed to run on CPU but can be intensive, adjust `NB_PROCESS` inside the script depending on your setup.

### 4. Analyze Results

Use the notebook [`results_analysis.ipynb`](./results_analysis.ipynb) to visualize key metrics, method complementarities, and more. 

---

**Happy autoformalizing !**