import os

# API keys
with open('/home/agatha/Desktop/MA3/sem proj/api_key_nlp_lab.txt', 'r') as file:
    OPENAI_API_KEY = file.read().strip()

# Dataset
# DATASET_NAME = 'agatha-duzan/mini_number_theory_af' 
DATASET_NAME ='agatha-duzan/number_theory_af'
FEW_SHOT_EXAMPLES_PATH = '/home/agatha/Desktop/MA3/sem proj/autoformalization_with_hypothesis/data/12_shot_proofnet_lean4.jsonl'
# FEW_SHOT_EXAMPLES_PATH = '/home/duzan/autoformalization_with_hypothesis/data/12_shot_proofnet_lean4.jsonl'

# Model and provider
DEFAULT_MODEL = 'gpt-4o'
DEFAULT_PROVIDER = 'openai'

# Process
FEWSHOT = False
HYPOTHESIS_DECOMP = True

# Output config
RESULTS_DIR = '/home/agatha/Desktop/MA3/sem proj/autoformalization_with_hypothesis/results/generations'
OUTPUT_NAME = 'base_hypothesis_decomp'
EVAL_RESULTS_DIR = '/home/agatha/Desktop/MA3/sem proj/autoformalization_with_hypothesis/results/evaluations'