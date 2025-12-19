from collections import Counter
from math import log2
from typing import List, Dict, Tuple, Any
from alastr.backend.tools.logger import logger


def compute_ngrams(
    PM,
    sequence: List[str],
    row_base: Dict[str, Any],
    prefix: str,
    gran: str
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    """
    Computes n-grams and associated statistics for a given sequence.

    Returns
    -------
    summary_data, ngram_data
        summary_data: { "<prefix>_ngram_summary": { ...metrics... } }
        ngram_data:   { "<prefix>_n{n}grams": [ {row}, {row}, ... ] }
    """
    # ----------------------------
    # Defensive validation
    # ----------------------------
    logger.info(
        "ENTER compute_ngrams: PM_id=%s PM_module=%s start_id=%s doc_id=%s",
        id(PM),
        PM.__class__.__module__,
        PM.ngram_id_doc,
        row_base.get("doc_id"),
    )

    summary_data: Dict[str, Dict[str, Any]] = {}
    ngram_data: Dict[str, List[Dict[str, Any]]] = {}

    if gran not in {"doc", "sent"}:
        logger.error(f"compute_ngrams: invalid gran='{gran}'. Expected 'doc' or 'sent'. Returning empty.")
        return summary_data, ngram_data

    if sequence is None:
        logger.warning("compute_ngrams: sequence is None. Returning empty.")
        return summary_data, ngram_data

    if not isinstance(sequence, list):
        logger.warning(f"compute_ngrams: sequence is {type(sequence)} not list; attempting to coerce.")
        try:
            sequence = list(sequence)
        except Exception:
            logger.exception("compute_ngrams: failed to coerce sequence to list. Returning empty.")
            return summary_data, ngram_data

    if not row_base or not isinstance(row_base, dict):
        logger.error("compute_ngrams: row_base missing or not a dict. Returning empty.")
        return summary_data, ngram_data

    # Helpful for tracking missing doc IDs
    doc_id = row_base.get("doc_id", None)
    sent_id = row_base.get("sent_id", None)

    if doc_id is None and gran == "doc":
        logger.warning("compute_ngrams: row_base has no 'doc_id' for gran='doc'. (This can break joins later.)")
    if sent_id is None and gran == "sent":
        logger.debug("compute_ngrams: row_base has no 'sent_id' for gran='sent'.")

    # PM validation
    try:
        max_n = int(PM.ngrams)
    except Exception:
        logger.exception("compute_ngrams: PM.ngrams missing/invalid. Returning empty.")
        return summary_data, ngram_data

    if max_n < 1:
        logger.info(f"compute_ngrams: PM.ngrams={max_n} < 1. Nothing to do.")
        return summary_data, ngram_data

    # Determine starting ID
    try:
        if gran == "doc":
            start_id = int(PM.ngram_id_doc)
        else:
            start_id = int(PM.ngram_id_sent)
    except Exception:
        logger.exception("compute_ngrams: PM.ngram_id_doc/sent missing/invalid. Returning empty.")
        return summary_data, ngram_data

    current_ngram_id = start_id
    summary_row = row_base.copy()

    logger.debug(
        f"compute_ngrams: start gran={gran} prefix={prefix} doc_id={doc_id} sent_id={sent_id} "
        f"len(sequence)={len(sequence)} max_n={max_n} start_id={start_id}"
    )

    # Empty sequence => still return a summary row (with zeros) if you want;
    # Here we return a summary with no n-metrics but log explicitly.
    if len(sequence) == 0:
        logger.info(f"compute_ngrams: empty sequence for prefix={prefix} gran={gran} doc_id={doc_id}.")
        summary_data[f"{prefix}_ngram_summary"] = summary_row
        return summary_data, ngram_data

    # ----------------------------
    # Main computation
    # ----------------------------
    total_records_written = 0

    for n in range(1, max_n + 1):
        try:
            if len(sequence) < n:
                logger.debug(
                    f"compute_ngrams: skipping n={n} because len(sequence)={len(sequence)} < n "
                    f"(prefix={prefix}, doc_id={doc_id}, gran={gran})"
                )
                continue

            ngram_list = [tuple(sequence[i:i + n]) for i in range(len(sequence) - n + 1)]
            if not ngram_list:
                logger.debug(f"compute_ngrams: n={n} produced no ngrams (unexpected). Skipping.")
                continue

            ngram_counts = Counter(ngram_list)
            total_ngrams = sum(ngram_counts.values())
            unique_ngrams = len(ngram_counts)

            # Entropy
            if total_ngrams > 0:
                probs = [count / total_ngrams for count in ngram_counts.values()]
                entropy = -sum(p * log2(p) for p in probs if p > 0)
            else:
                entropy = 0.0

            # Coverage metrics
            sorted_counts = sorted(ngram_counts.values(), reverse=True)
            if total_ngrams > 0:
                coverage3 = (sum(sorted_counts[:3]) / total_ngrams) if total_ngrams >= 3 else (sum(sorted_counts) / total_ngrams)
                coverage5 = (sum(sorted_counts[:5]) / total_ngrams) if total_ngrams >= 5 else (sum(sorted_counts) / total_ngrams)
            else:
                coverage3 = 0.0
                coverage5 = 0.0

            # Diversity
            diversity = (unique_ngrams / total_ngrams) if total_ngrams > 0 else 0.0

            # Add metrics to summary row
            summary_row[f"unique_n{n}grams"] = unique_ngrams
            summary_row[f"diversity_n{n}gram"] = diversity
            summary_row[f"entropy_n{n}gram"] = entropy
            summary_row[f"coverage3_n{n}gram"] = coverage3
            summary_row[f"coverage5_n{n}gram"] = coverage5

            table_name = f"{prefix}_n{n}grams"
            records: List[Dict[str, Any]] = []

            # Build record rows
            for rank, (ngram, count) in enumerate(ngram_counts.most_common(), start=1):
                row_data = row_base.copy()
                row_data.update({
                    "ngram_id": current_ngram_id,
                    "n": n,
                    "ngram": "_".join(ngram),
                    "rank": rank,
                    "count": count,
                    "prop": (count / total_ngrams) if total_ngrams > 0 else 0.0,
                    "coverage": ((count * n) / len(sequence)) if len(sequence) > 0 else 0.0,
                })
                records.append(row_data)
                current_ngram_id += 1

            # SAFER accumulation (prevents overwrite if same table_name is hit twice)
            ngram_data.setdefault(table_name, []).extend(records)

            total_records_written += len(records)

            logger.debug(
                f"compute_ngrams: wrote table={table_name} n={n} rows={len(records)} "
                f"(unique={unique_ngrams}, total={total_ngrams}) "
                f"id_range=[{current_ngram_id - len(records)}, {current_ngram_id - 1}] "
                f"doc_id={doc_id} gran={gran}"
            )

        except Exception:
            logger.exception(
                f"compute_ngrams: failed for n={n} prefix={prefix} gran={gran} doc_id={doc_id}. Continuing."
            )
            continue

    # Insert summary row
    summary_data[f"{prefix}_ngram_summary"] = summary_row

    # ----------------------------
    # FIXED ngram_id update
    # ----------------------------
    # current_ngram_id is the *next available* ID after writing records.
    # Set PM.ngram_id_* to this value.
    try:
        if gran == "doc":
            PM.ngram_id_doc = current_ngram_id
        else:
            PM.ngram_id_sent = current_ngram_id
    except Exception:
        logger.exception("compute_ngrams: failed to update PM.ngram_id_doc/sent.")

    logger.info(
        f"compute_ngrams: done prefix={prefix} gran={gran} doc_id={doc_id} "
        f"records_written={total_records_written} start_id={start_id} next_id={current_ngram_id}"
    )

    return summary_data, ngram_data, current_ngram_id
