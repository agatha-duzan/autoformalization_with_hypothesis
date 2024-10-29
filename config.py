import os

# API keys
#OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', 'YOUR_OPENAI_API_KEY')
# ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY', 'YOUR_ANTHROPIC_API_KEY')
# COHERE_API_KEY = os.environ.get('COHERE_API_KEY', 'YOUR_COHERE_API_KEY')

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

# Output config
RESULTS_DIR = 'results/generations'
OUTPUT_NAME = 'direct_translation_with_general_fewshot'