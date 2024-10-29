from nltk.translate.bleu_score import sentence_bleu
from utils import clean_theorem_string

def bleu_eval(generated_formal_statement, formal_statement):
    # clean up both statements
    generated = clean_theorem_string(generated_formal_statement)
    reference = clean_theorem_string(formal_statement)
    