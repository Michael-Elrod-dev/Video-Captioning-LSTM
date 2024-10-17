import math
import operator
import sys
import json
from functools import reduce
from typing import List, Dict, Union

def count_ngram(candidate_sentences: List[str], references: List[List[str]], n: int) -> tuple:
    total_clipped = 0
    total_count = 0
    total_reference_length = 0
    total_candidate_length = 0

    for index, candidate in enumerate(candidate_sentences):
        reference_ngram_counts = []
        reference_sentence_lengths = []

        for reference in references:
            ref_sentence = reference[index]
            ngram_d = build_ngram_dict(ref_sentence, n)
            reference_ngram_counts.append(ngram_d)
            reference_sentence_lengths.append(len(ref_sentence.strip().split()))

        cand_dict = build_ngram_dict(candidate, n)
        total_clipped += clip_count(cand_dict, reference_ngram_counts)
        total_count += max(0, len(candidate.strip().split()) - n + 1)
        total_reference_length += best_length_match(reference_sentence_lengths, len(candidate.strip().split()))
        total_candidate_length += len(candidate.strip().split())

    precision = float(total_clipped) / total_count if total_clipped > 0 else 0
    brevity = brevity_penalty(total_candidate_length, total_reference_length)
    return precision, brevity

def build_ngram_dict(sentence: str, n: int) -> Dict[str, int]:
    words = sentence.strip().split()
    ngram_dict = {}
    for i in range(max(0, len(words) - n + 1)):
        ngram = ' '.join(words[i:i+n]).lower()
        ngram_dict[ngram] = ngram_dict.get(ngram, 0) + 1
    return ngram_dict

def clip_count(cand_d: Dict[str, int], ref_ds: List[Dict[str, int]]) -> int:
    return sum(min(cand_d.get(ngram, 0), max(ref.get(ngram, 0) for ref in ref_ds)) for ngram in cand_d)

def best_length_match(ref_lengths: List[int], cand_length: int) -> int:
    return min(ref_lengths, key=lambda ref: abs(cand_length - ref))

def brevity_penalty(c: int, r: int) -> float:
    return 1 if c > r else math.exp(1 - float(r) / c)

def geometric_mean(precisions: List[float]) -> float:
    return (reduce(operator.mul, precisions)) ** (1.0 / len(precisions))

def BLEU(s: str, t: Union[str, List[str]], flag: bool = False) -> float:
    candidate = [s.strip()]
    references = [[t[i].strip()] for i in range(len(t))] if flag else [[t.strip()]]
    
    precision, brevity = count_ngram(candidate, references, 1)
    score = geometric_mean([precision]) * brevity
    return score

def load_test_data(file_path: str) -> List[Dict]:
    with open(file_path, 'r') as f:
        return json.load(f)

def load_output_data(file_path: str) -> Dict[str, str]:
    result = {}
    with open(file_path, 'r') as f:
        for line in f:
            test_id, caption = line.rstrip().split(',', 1)
            result[test_id] = caption
    return result

def calculate_bleu_scores(test_data: List[Dict], result: Dict[str, str]) -> List[float]:
    return [BLEU(result[item['id']], [x.rstrip('.') for x in item['caption']], True) for item in test_data]

def main():
    if len(sys.argv) != 2:
        print("Usage: python bleu_eval.py caption.txt")
        sys.exit(1)

    test_data = load_test_data('testing_label.json')
    output_data = load_output_data(sys.argv[1])
    bleu_scores = calculate_bleu_scores(test_data, output_data)
    average_bleu = sum(bleu_scores) / len(bleu_scores)
    
    print(f"Average BLEU score: {average_bleu:.4f}")

if __name__ == "__main__":
    main()