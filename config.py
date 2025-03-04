# API keys
with open('/home/agatha/Desktop/MA3/sem proj/api_key_nlp_lab.txt', 'r') as file:
    OPENAI_API_KEY = file.read().strip()

# Dataset
DATASET_NAME ='PAug/ProofNetSharp'
FEW_SHOT_EXAMPLES_PATH = '/home/agatha/Desktop/MA3/sem proj/autoformalization_with_hypothesis/data/8_shot_proofnet_lean4_decomp.jsonl'

# Model and provider
DEFAULT_MODEL = 'gpt-4o-2024-11-20'
DEFAULT_PROVIDER = 'openai'

# Process
FEWSHOT = True
METHOD = 'leansearch'

# Output config
OUTPUT_NAME = 'leansearch_top5'
RESULTS_DIR = '/home/agatha/Desktop/MA3/sem proj/autoformalization_with_hypothesis/results/generations'
EVAL_RESULTS_DIR = '/home/agatha/Desktop/MA3/sem proj/autoformalization_with_hypothesis/results/evaluations'
CHECKPOINT_DIR = '/home/agatha/Desktop/MA3/sem proj/autoformalization_with_hypothesis/results/checkpoints'
PREMISES_DIR = '/home/agatha/Desktop/MA3/sem proj/autoformalization_with_hypothesis/results/premises'