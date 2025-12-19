from __future__ import annotations
from collections import Counter
from math import log2, sqrt
from typing import Any, Dict, List, Optional, Sequence, Tuple
from alastr.backend.tools.logger import logger


def compute_dispersion_features(
    sequence: Sequence[str],
    ngram: Tuple[str, ...],
    n: int,
    *,
    win_k: int = 10,
    allow_overlaps: bool = True,
) -> Dict[str, Optional[float]]:
    """
    Compute dispersion/location features for a given n-gram within a sequence.

    This treats n-gram occurrences as a 1-D point process over the sequence,
    using the *start indices* of each match.

    Features returned (all floats or None):
      - pos_mean, pos_var, pos_skew, pos_kurt
      - gap_mean, gap_cv, gap_min, gap_max
      - win_k, vmr, win_entropy

    Notes:
      - Positions are normalized to [0, 1] using denominator (L - n), so 0=start, 1=end.
      - Skew/kurtosis use common *sample* moment definitions; returned as None when unstable.
      - Gap metrics are computed on sorted start positions; None if fewer than 2 occurrences.
      - Window metrics partition the sequence into win_k contiguous windows and count occurrences per window.
        VMR = Var(counts)/Mean(counts). win_entropy is Shannon entropy over window proportions.

    Parameters
    ----------
    sequence:
        The full token/grapheme/tag sequence.
    ngram:
        The n-gram tuple to search for (length should equal n).
    n:
        n-gram length.
    win_k:
        Number of windows for windowed dispersion metrics (defaults to 10).
        Will be down-adjusted if sequence is short.
    allow_overlaps:
        If False, occurrences are taken non-overlapping (greedy left-to-right).
        MVP default True (matches compute_ngrams list construction which allows overlaps).

    Returns
    -------
    Dict[str, Optional[float]]
        Dispersion features for the n-gram in this sequence.
    """
    try:
        L = len(sequence)
        if L == 0 or n <= 0 or n > L:
            return {
                "pos_mean": None, "pos_var": None, "pos_skew": None, "pos_kurt": None,
                "gap_mean": None, "gap_cv": None, "gap_min": None, "gap_max": None,
                "win_k": float(win_k), "vmr": None, "win_entropy": None,
            }

        if len(ngram) != n:
            logger.warning(
                "compute_dispersion_features: len(ngram)=%s != n=%s; proceeding anyway.",
                len(ngram), n,
            )

        # ----------------------------
        # Find occurrence start indices
        # ----------------------------
        positions: List[int] = []
        i = 0
        while i <= L - n:
            if tuple(sequence[i:i + n]) == ngram:
                positions.append(i)
                i += (n if not allow_overlaps else 1)
            else:
                i += 1

        m = len(positions)

        # ----------------------------
        # Position-shape stats
        # ----------------------------
        # Normalize to [0,1] using (L-n) so last possible start maps to 1.
        denom = (L - n)
        if denom <= 0:
            # Only possible if L==n, in which case any match starts at 0; treat as position 0.
            xs = [0.0 for _ in positions]
        else:
            xs = [p / denom for p in positions]

        def _mean(vals: List[float]) -> Optional[float]:
            return sum(vals) / len(vals) if vals else None

        def _var_sample(vals: List[float]) -> Optional[float]:
            # sample variance (ddof=1)
            if len(vals) < 2:
                return None
            mu = sum(vals) / len(vals)
            return sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)

        pos_mean = _mean(xs)
        pos_var = _var_sample(xs)

        # sample skewness and (excess) kurtosis based on sample central moments
        # Returned as None if not enough data or variance is ~0.
        pos_skew = None
        pos_kurt = None
        if m >= 3:
            mu = sum(xs) / m
            m2 = sum((v - mu) ** 2 for v in xs) / m
            if m2 > 0:
                m3 = sum((v - mu) ** 3 for v in xs) / m
                # population-style skewness; adequate for MVP and stable interpretation
                pos_skew = m3 / (m2 ** 1.5)
        if m >= 4:
            mu = sum(xs) / m
            m2 = sum((v - mu) ** 2 for v in xs) / m
            if m2 > 0:
                m4 = sum((v - mu) ** 4 for v in xs) / m
                # excess kurtosis (kurtosis - 3)
                pos_kurt = (m4 / (m2 ** 2)) - 3.0

        # ----------------------------
        # Gap stats (on start indices)
        # ----------------------------
        gap_mean = None
        gap_cv = None
        gap_min = None
        gap_max = None
        if m >= 2:
            positions_sorted = sorted(positions)
            gaps = [positions_sorted[j + 1] - positions_sorted[j] for j in range(m - 1)]
            gap_mean = sum(gaps) / len(gaps)
            gap_min = float(min(gaps))
            gap_max = float(max(gaps))

            if gap_mean > 0 and len(gaps) >= 2:
                g_mu = gap_mean
                g_var = sum((g - g_mu) ** 2 for g in gaps) / (len(gaps) - 1)
                g_sd = sqrt(g_var) if g_var >= 0 else 0.0
                gap_cv = (g_sd / g_mu) if g_mu != 0 else None
            elif gap_mean > 0 and len(gaps) == 1:
                gap_cv = 0.0  # only one gap => no variability
            else:
                gap_cv = None

        # ----------------------------
        # Windowed metrics (VMR + entropy)
        # ----------------------------
        # Use win_k windows across possible start positions [0..L-n].
        # That makes window counting consistent across n.
        max_start = L - n
        if max_start < 0:
            # already handled by early return, but keep safe
            actual_k = 1
        else:
            # down-adjust K so windows aren't empty due to tiny sequences
            actual_k = int(win_k) if int(win_k) > 0 else 10
            actual_k = min(actual_k, max_start + 1)  # can't have more windows than points
            actual_k = max(actual_k, 1)

        # Build window edges over [0, max_start+1) integer starts
        # We'll assign each position to a window via floor(pos / window_size).
        counts = [0] * actual_k
        if m > 0 and max_start >= 0:
            # window_size in "start-index units"
            window_size = (max_start + 1) / actual_k
            for p in positions:
                # guard numeric edges; p is int within [0, max_start]
                idx = int(p / window_size) if window_size > 0 else 0
                if idx >= actual_k:
                    idx = actual_k - 1
                counts[idx] += 1

        mean_c = (sum(counts) / actual_k) if actual_k > 0 else None
        vmr = None
        if mean_c is not None and mean_c > 0 and actual_k >= 2:
            # sample variance of window counts / mean
            c_mu = mean_c
            c_var = sum((c - c_mu) ** 2 for c in counts) / (actual_k - 1)
            vmr = c_var / c_mu
        elif mean_c == 0:
            vmr = None

        win_entropy = None
        total_c = sum(counts)
        if total_c > 0:
            probs = [c / total_c for c in counts if c > 0]
            win_entropy = -sum(p * log2(p) for p in probs)

        return {
            "pos_mean": float(pos_mean) if pos_mean is not None else None,
            "pos_var": float(pos_var) if pos_var is not None else None,
            "pos_skew": float(pos_skew) if pos_skew is not None else None,
            "pos_kurt": float(pos_kurt) if pos_kurt is not None else None,

            "gap_mean": float(gap_mean) if gap_mean is not None else None,
            "gap_cv": float(gap_cv) if gap_cv is not None else None,
            "gap_min": float(gap_min) if gap_min is not None else None,
            "gap_max": float(gap_max) if gap_max is not None else None,

            "win_k": float(actual_k),
            "vmr": float(vmr) if vmr is not None else None,
            "win_entropy": float(win_entropy) if win_entropy is not None else None,
        }

    except Exception:
        logger.exception("compute_dispersion_features: unexpected error.")
        # fail closed: return Nones so pipeline keeps running
        return {
            "pos_mean": None, "pos_var": None, "pos_skew": None, "pos_kurt": None,
            "gap_mean": None, "gap_cv": None, "gap_min": None, "gap_max": None,
            "win_k": float(win_k), "vmr": None, "win_entropy": None,
        }


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

                disp = compute_dispersion_features(sequence, ngram, n, win_k=10, allow_overlaps=True)
                row_data.update(disp)

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
