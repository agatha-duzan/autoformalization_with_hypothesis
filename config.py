import os

# API keys
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', 'YOUR_OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY', 'YOUR_ANTHROPIC_API_KEY')
COHERE_API_KEY = os.environ.get('COHERE_API_KEY', 'YOUR_COHERE_API_KEY')

# Dataset
DATASET_NAME = 'agatha-duzan/number_theory_af'
FEW_SHOT_EXAMPLES_PATH = '/home/duzan/autoformalization_with_hypothesis/data/12_shot_proofnet_lean4.jsonl'

# Model and provider
DEFAULT_MODEL = 'gpt-3.5-turbo'
DEFAULT_PROVIDER = 'openai'

# Output config
RESULTS_DIR = 'results'
OUTPUT_FILE = 'translation_results.json'
