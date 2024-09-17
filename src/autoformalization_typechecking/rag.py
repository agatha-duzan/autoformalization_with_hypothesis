import os
import re
from dataclasses import dataclass
from enum import Enum

import jsonlines
import nltk
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from pylatexenc.latex2text import LatexNodes2Text
from rank_bm25 import BM25Okapi
from rich.syntax import Syntax
from thefuzz import fuzz
from word2number import w2n

from autoformalization_typechecking.utils import DATA_DIR, clean_theorem_string, console


class TheoremType(Enum):
    LATEX = 1
    LEAN = 2


@dataclass
class RAGResult:
    rank: int
    score: float
    doc: str


class RAG:
    def __init__(self) -> None:
        nltk.download("stopwords", quiet=True)
        nltk.download("punkt", quiet=True)
        nltk.download("wordnet", quiet=True)

        self._stop_words = set(stopwords.words("english")) | {
            "theorem",
            "lemma",
            "corollary",
            "hypothesis",
            "proof",
            "let",
            "show",
            "assume",
            "suppose",
            "qed",
            "prove",
        }
        # add lemmatised and stemmed versions of stop words
        self._stop_words = self._stop_words | {
            LancasterStemmer().stem(WordNetLemmatizer().lemmatize(word)) for word in self._stop_words
        }

        # initialize mathlib theorems
        with console.status("[bold]Loading Mathlib theorems[/bold]"):
            self._mathlib_theorems = []
            self._load_mathlib_theorems()

        # initialize BM25 index
        with console.status("[bold]Building BM25 index[/bold]"):
            tokenized_corpus = [self._preprocess_lean_theorem(doc) for doc in self._mathlib_theorems]
            self._bm25 = BM25Okapi(tokenized_corpus)

    def _load_mathlib_theorems(self):
        with jsonlines.open(os.path.join(DATA_DIR, "mathlib4_statements.jsonl")) as reader:
            all_mathlib_theorems = list(obj["short_decl"] for obj in reader if obj["statement_kind"] == "theorem")

        # remove exact duplicates
        visited = set()
        self._mathlib_theorems = []
        for thm in all_mathlib_theorems:
            normalized_thm = clean_theorem_string(thm)
            if normalized_thm not in visited:
                visited.add(normalized_thm)
                self._mathlib_theorems.append(thm)

    @staticmethod
    def _latex_to_text(latex_str: str):
        latex_str = latex_str.replace("\\", " \\ \\")  # add space before backslashes to separate symbols
        return LatexNodes2Text(strict_latex_spaces="based-on-source").latex_to_text(latex_str)

    def _remove_low_information_tokens(self, tokens: list[str]) -> list[str]:
        # remove tokens that are 1 character or less and alpha-numeric
        tokens = [tok for tok in tokens if len(tok) > 1 or not tok.isalnum() or not tok.isascii()]

        # remove numbers
        tokens = [tok for tok in tokens if not tok.isnumeric()]

        # remove numbers written as words
        new_tokens = []
        for tok in tokens:
            try:
                w2n.word_to_num(tok)
            except ValueError:
                new_tokens.append(tok)
        tokens = new_tokens

        # remove stop words
        tokens = [tok for tok in tokens if tok not in self._stop_words]

        return tokens

    def _canonize_clean_list_words(self, list_words: list[str]) -> list[str]:
        list_words = [word.lower() for word in list_words]
        list_words = list(set(list_words))
        list_words = self._remove_low_information_tokens(list_words)

        # lemmatize and stem
        lemmatizer = WordNetLemmatizer()
        stemmer = LancasterStemmer()
        list_words = [stemmer.stem(lemmatizer.lemmatize(tok)) for tok in list_words]

        list_words = list(set(list_words))
        list_words = self._remove_low_information_tokens(list_words)

        return list_words

    @staticmethod
    def _remove_punkt(s: str) -> str:
        punkt = ".,;:!?()[]{}-_^=+*/\\|\"'`~@#$%&$§<>«»"
        return s.translate(str.maketrans(punkt, " " * len(punkt)))

    @staticmethod
    def _extract_variables_lean_thm(theorem: str) -> list[str]:
        # Pattern to match variables inside curly braces and parentheses
        # variables are of the forms: {var1 var2 : type}, (var1 var2 : type), [Type var1 var2]
        pattern = r"\{([^}]+)\}|\(([^)]+)\)|\[([^]]+)\]"

        matches = re.findall(pattern, theorem)

        variables = []
        for match in matches:
            if match[0] or match[1]:
                group = match[0] if match[0] else match[1]
                # Extract variable names by splitting by ':' and taking the first part
                if ":" in group:
                    group_vars = group.split(":")[0].strip()
                    local_vars = [var.strip() for var in group_vars.split()]
                    variables.extend(local_vars)
            else:
                group = match[2]
                local_vars = group.split()[1:]
                variables.extend(local_vars)

        return variables

    def _preprocess_lean_theorem(self, theorem_str: str):
        # extract hypotheses/variables names
        variables = self._extract_variables_lean_thm(theorem_str)

        # remove parentheses, brackets, and other special characters
        clean_thm_str = self._remove_punkt(theorem_str)

        # tokenize
        clean_thm_tok = clean_thm_str.strip().split()

        # remove variables
        clean_thm_tok = [tok for tok in clean_thm_tok if tok not in variables]

        # split words in CamelCase
        clean_thm_tok = [tok for tok in clean_thm_tok for tok in re.split("([A-Z][a-z]+)", tok) if tok]

        clean_thm_tok = self._canonize_clean_list_words(clean_thm_tok)

        return clean_thm_tok

    def _preprocess_latex_theorem(self, theorem_str: str):
        # first path: convert latex to text
        clean_thm_str_1 = self._latex_to_text(theorem_str)
        clean_thm_str_1 = self._remove_punkt(clean_thm_str_1)

        # second path: don't transform latex to text
        clean_thm_str_2 = self._remove_punkt(theorem_str)

        # tokenize and merge the two paths
        theorem_tok = list(set(word_tokenize(clean_thm_str_1)) | set(word_tokenize(clean_thm_str_2)))

        theorem_tok = self._canonize_clean_list_words(theorem_tok)

        return theorem_tok

    def _preprocess_theorem(self, theorem_str: str, theorem_type: TheoremType):
        match theorem_type:
            case TheoremType.LATEX:
                return self._preprocess_latex_theorem(theorem_str)
            case TheoremType.LEAN:
                return self._preprocess_lean_theorem(theorem_str)
            case _:
                raise ValueError(f"Unsupported theorem type: {theorem_type}")

    def get_top_k_theorems(
        self,
        query: str,
        k: int,
        theorem_type: TheoremType = TheoremType.LATEX,
        min_diversity_threshold: float | None = None,
    ) -> list[RAGResult]:
        """
        Get the top k theorems from the Mathlib library that are most similar to the query.
        """
        assert min_diversity_threshold is None or 0 <= min_diversity_threshold <= 1
        max_similarity_threshold = None if min_diversity_threshold is None else 100 * (1 - min_diversity_threshold)

        query = self._preprocess_theorem(query, theorem_type)
        scores = self._bm25.get_scores(query)

        # sort the scores in descending order
        sorted_doc_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        # keep the best k results that are diverse enough
        top_k: list[RAGResult] = []
        for rank, idx in enumerate(sorted_doc_idx):
            doc = self._mathlib_theorems[idx]
            if max_similarity_threshold is None or all(
                fuzz.ratio(doc, top_doc.doc) < max_similarity_threshold for top_doc in top_k
            ):
                top_k.append(RAGResult(rank=rank + 1, score=scores[idx], doc=doc))
            if len(top_k) == k:
                break

        return top_k


if __name__ == "__main__":
    rag = RAG()
    k = 5

    few_shot_theorems = []
    with jsonlines.open(os.path.join(DATA_DIR, "12_shot_proofnet_lean4.jsonl")) as reader:
        few_shot_theorems = list(obj for obj in reader)

    for few_shot_thm in few_shot_theorems:
        top_k = rag.get_top_k_theorems(few_shot_thm["nl_statement"], k=k, min_diversity_threshold=0.4)

        # Debug information
        query = rag._preprocess_theorem(few_shot_thm["nl_statement"], TheoremType.LATEX)
        console.print("Query:")
        console.print(few_shot_thm["nl_statement"])
        console.print(Syntax(few_shot_thm["formal_statement"], "lean4"))
        console.print(query)

        console.print(f"Top {k} BM25 results:")
        for result in top_k:
            console.print(f"Rank: {result.rank}, Score: {result.score:.2f}")
            console.print(Syntax(result.doc, "lean4"))
            console.print(rag._preprocess_theorem(result.doc, TheoremType.LEAN))
            console.print()
        console.print()
