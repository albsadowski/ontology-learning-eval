# Measuring Information Preservation in Legal Ontology Learning

This repository contains the code and prompts for evaluating information preservation in legal ontology learning, as described in the paper "Measuring Information Preservation in Legal Ontology Learning" by Albert Sadowski and Jarosław A. Chudziak.

## Overview

Current evaluation methods for ontology learning focus on structural correctness but fail to quantify the trade-offs in information accessibility when transforming documents into ontological representations. This project introduces an evaluation methodology that uses baseline LLM performance as a reference point for measuring information preservation during ontological transformation.

## Results

Our evaluation reveals systematic information loss across all ontology learning approaches, with performance degradation ranging from 8.4 to 27.4 percentage points compared to baseline LLM performance on source documents.

| Model | Method | Loss (pp) | Success (%) | Failure (%) |
|-------|--------|-----------|-------------|-------------|
| **GPT 4.1 mini** | LLMs4OL | 17.5 | 64.13 | 16.02 |
| | **NeOn-GPT** | **8.8** | **79.33** | 12.71 |
| | NeOn-CoT | 12.7 | 71.73 | 15.47 |
| **Gemini 2.5 Flash** | LLMs4OL | 27.4 | 57.71 | 16.97 |
| | NeOn-GPT | 11.2 | 81.39 | 8.98 |
| | **NeOn-CoT** | **8.4** | **82.9** | 16.02 |
| **DeepSeek v3** | LLMs4OL | 25.1 | 54.31 | 22.01 |
| | **NeOn-GPT** | **17.6** | **65.82** | 20.08 |
| | NeOn-CoT | 25.3 | 53.3 | 23.55 |
| **Claude Sonnet 4** | LLMs4OL | 24.8 | 45.08 | 19.51 |
| | **NeOn-GPT** | **18.1** | **60.66** | 13.41 |
| | NeOn-CoT | 19.1 | 53.28 | 21.95 |
| **o4 mini** | LLMs4OL | 22.8 | 61.94 | 14.09 |
| | **NeOn-GPT** | **9.6** | **80.06** | 15.44 |
| | NeOn-CoT | 16.3 | 70.9 | 14.77 |
| **Llama 4 Maverick** | **LLMs4OL** | **13.9** | 59.29 | 36.22 |
| | NeOn-GPT | 14.5 | **64.8** | 24.19 |
| | NeOn-CoT | 19.0 | 57.17 | 25.05 |

*Baseline accuracies: GPT 4.1 mini (64.51%), Gemini 2.5 Flash (75.14%), DeepSeek v3 (69.53%), Claude Sonnet 4 (59.80%), o4 mini (70.78%), Llama 4 Maverick (65.53%)*

## Methodology

### Evaluation Framework

Our approach treats baseline LLM performance on source documents as representing the total accessible information from that specific LLM's perspective, then measures how much information remains accessible when the same LLM processes transformed ontological representations.

**Information Loss = Baseline Performance - Transformed Performance**

### Approaches Evaluated

1. **Direct LLM Application** (0-shot baseline)
2. **LLMs4OL**: Parallel task decomposition approach with term typing, taxonomy discovery, and relation extraction
3. **NeOn-GPT**: Structured five-stage pipeline with domain awareness for legal documents
4. **NeOn-CoT**: Simplified NeOn methodology using chain-of-thought prompting

### Dataset

- **MAUD (Merger Agreement Understanding Dataset)** from LegalBench
- 34 multiple-choice tasks focusing on merger agreement analysis
- 2,346 total examples (69 examples per task for balanced evaluation)
- Tasks include material adverse effect definitions, representations and warranties, fiduciary obligations

## Usage

### Basic Execution

```bash
# Run evaluation on all tasks with all methods for a specific model
uv run main.py --dataset test --task all --mode all --strip --model <MODEL_NAME> --sample 69
```

### Parameters

- `--dataset`: MAUD Dataset to use (`train` or `test`)
- `--task`: Specific task ID or `all` for all tasks
- `--mode`: Evaluation mode (`all` for all methods, or specific method)
- `--strip`: Strip whitespace and normalize Turtles
- `--model`: LLM model to use
- `--sample`: Number of samples per task (69 for balanced evaluation)

### Example Commands

```bash
# Evaluate NeOn-GPT approach with Gemini 2.5 Flash
uv run main.py --dataset test --task all --mode neon-gpt --model gemini-2.5-flash --sample 69

# Evaluate specific task (t13 - fiduciary duty standards)
uv run main.py --dataset test --task t13 --mode all --model claude-sonnet-4 --sample 69

# Run baseline evaluation only
uv run main.py --dataset test --task all --mode baseline --model gpt-4.1-mini --sample 69
```

## License

This project is licensed under the [MIT License](LICENSE).
