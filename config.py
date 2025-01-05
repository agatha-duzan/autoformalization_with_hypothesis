import os

# API keys
with open('/home/agatha/Desktop/MA3/sem proj/api_key_nlp_lab.txt', 'r') as file:
    OPENAI_API_KEY = file.read().strip()

# Dataset
DATASET_NAME ='agatha-duzan/number_theory_af' # 'agatha-duzan/advanced_algebra_af' # 'agatha-duzan/number_theory_af'
FEW_SHOT_EXAMPLES_PATH = '/home/agatha/Desktop/MA3/sem proj/autoformalization_with_hypothesis/data/8_shot_proofnet_lean4_decomp.jsonl'
# OLD: '/home/agatha/Desktop/MA3/sem proj/autoformalization_with_hypothesis/data/12_shot_proofnet_lean4.jsonl'

# Model and provider
DEFAULT_MODEL = 'gpt-4o-2024-11-20'
DEFAULT_PROVIDER = 'openai'

# Process
FEWSHOT = True
HYPOTHESIS_DECOMP = None

# Output config
OUTPUT_NAME = 'baseline_leansearch_top5'
RESULTS_DIR = '/home/agatha/Desktop/MA3/sem proj/autoformalization_with_hypothesis/results/generations'
EVAL_RESULTS_DIR = '/home/agatha/Desktop/MA3/sem proj/autoformalization_with_hypothesis/results/evaluations'
CHECKPOINT_DIR = '/home/agatha/Desktop/MA3/sem proj/autoformalization_with_hypothesis/results/checkpoints'