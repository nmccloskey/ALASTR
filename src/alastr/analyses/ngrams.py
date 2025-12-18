from collections import Counter
from math import log2
from typing import List, Dict
from alastr.backend.tools.logger import logger


def compute_ngrams(PM, sequence: List[str], row_base: Dict, prefix: str, gran: str) -> Dict[str, List[Dict]]:
    """
    Computes n-grams and associated statistics for a given sequence.

    Args:
        sequence (List[str]): Input sequence (graphemes, phonemes, etc.).
        row_base (Dict): Metadata row (doc_id, sent_id, etc.).
        prefix (str): Prefix for output table.
        gran (str): Granularity ('doc' or 'sent').

    Returns:
        Dict[str, List[Dict]]: 
            ngram_data: Summary + per-n-gram table data.
    """
    ngram_data = {}
    summary_data = {}
    summary_row = row_base.copy()
    current_ngram_id = PM.ngram_id_doc if gran == "doc" else PM.ngram_id_sent

    for n in range(1, PM.ngrams + 1):
        ngram_list = [tuple(sequence[i:i+n]) for i in range(len(sequence)-n+1)]
        if not ngram_list:
            continue

        ngram_counts = Counter(ngram_list)
        total_ngrams = sum(ngram_counts.values())
        unique_ngrams = len(ngram_counts)

        # Entropy
        probs = [count / total_ngrams for count in ngram_counts.values()]
        entropy = -sum(p * log2(p) for p in probs) if total_ngrams > 0 else 0

        # Coverage metrics
        sorted_counts = sorted(ngram_counts.values(), reverse=True)
        coverage3 = sum(sorted_counts[:3]) / total_ngrams if total_ngrams >= 3 else sum(sorted_counts) / total_ngrams
        coverage5 = sum(sorted_counts[:5]) / total_ngrams if total_ngrams >= 5 else sum(sorted_counts) / total_ngrams

        # Diversity
        diversity = unique_ngrams / total_ngrams if total_ngrams > 0 else 0

        # Add metrics to summary row
        summary_row[f"unique_n{n}grams"] = unique_ngrams
        summary_row[f"diversity_n{n}gram"] = diversity
        summary_row[f"entropy_n{n}gram"] = entropy
        summary_row[f"coverage3_n{n}gram"] = coverage3
        summary_row[f"coverage5_n{n}gram"] = coverage5

        # N-gram data table
        table_name = f"{prefix}_n{n}grams"
        records = []
        for rank, (ngram, count) in enumerate(ngram_counts.most_common(), start=1):
            row_data = row_base.copy()
            row_data.update({
                "ngram_id": current_ngram_id,
                "n": n,
                "ngram": "_".join(ngram),
                "count": count,
                "proportion": count / total_ngrams,
                "rank": rank
            })
            records.append(row_data)
            current_ngram_id += 1
        
        ngram_data[table_name] = records

    # Insert summary row as first entry in ngram_data
    summary_data[f"{prefix}_ngram_summary"] = summary_row

    if gran == "doc":
        PM.ngram_id_doc += current_ngram_id
    elif gran == "sent":
        PM.ngram_id_sent += current_ngram_id

    return summary_data, ngram_data
