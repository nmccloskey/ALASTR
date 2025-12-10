import re
import numpy as np
import logging
logger = logging.getLogger("CustomLogger")
import re
from pathlib import Path


def scrub_raw_text(text: str) -> str:
    """
    Normalizes raw text for storage/export:
    - Converts paragraph breaks to <p>
    - Removes exotic unicode breaks
    - Trims leading/trailing space

    Args:
        text (str): Raw input text (may contain newlines or special breaks)

    Returns:
        str: Normalized text safe for export.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")  # Normalize all to \n
    text = text.replace("\u2029", "\n").replace("\u2028", "\n")  # Paragraph & line separators
    text = re.sub(r'\n+', '<p>', text)  # Collapse multiple newlines into <p>
    return text.strip()

def clean_text(text: str) -> str:
    """
    Clean a text string by normalizing whitespace, fixing punctuation spacing,
    collapsing broken hyphenated words, and removing unwanted characters.

    Args:
        text (str): Input text.

    Returns:
        str: Cleaned version of the text.
    """
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r"([.!?])([A-Za-z])", r"\1 \2", text)  # Ensure sentence spacing
    text = re.sub(r'\s+(?=[.?!])', '', text)             # Remove spaces before punctuation
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1-\2', text)      # Rejoin split words
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    text = re.sub(r'[^\w\s.,;?!-]', '', text)             # Strip unwanted symbols
    # Repeat for good measure.
    text = re.sub(r'\s+', ' ', text).strip()

    return text.strip()

def get_text_from_cha(file_path: Path, exclude_speakers=[]) -> str:
    """
    Extracts utterances from a .cha CHAT file, concatenating multiline utterances
    and excluding specified speakers. Returns one large string of joined utterances.

    Args:
        file_path (Path): Path to the .cha file.
        exclude_speakers (list): Speakers to exclude (e.g., ["INV"]).

    Returns:
        str: Concatenated string of utterances from desired speakers.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # Merge lines for easier multiline utterance matching
    raw_text = raw_text.replace("\n", " ").replace("\t", " ")

    # Match utterances ending in .!? from lines like *SPEAKER: utterance
    utterance_pattern = re.compile(r"\*(\w+):\s(.+?[.!?])", re.DOTALL)
    
    text_content = ""
    for match in utterance_pattern.finditer(raw_text):
        speaker, utterance = match.groups()
        if speaker not in exclude_speakers:
            cleaned = " ".join(utterance.strip().split())
            cleaned = re.sub(r"\s(?=[.!?])", "", cleaned)
            text_content += " " + cleaned[0].upper() + cleaned[1:]

    return text_content.strip()

def process_clan_text(text: str, version: str = "target") -> str:
    """
    Process CLAN-formatted utterances by either:
    - Swapping in corrected tokens (e.g., [: birthday]) [target]
    - Retaining original phonological fragments [phon]

    Also removes paraphasias (&+word, &-word) appropriately.

    Args:
        text (str): Input utterance.
        version (str): 'target' = corrected forms; 'phon' = original forms.

    Returns:
        str: Processed utterance.
    """
    # Handle replacements like: birbday [: birthday]
    correction_pattern = re.compile(r'(\b\w+)\s*\[:\s*([^]]+)\]')

    if version == "phon":
        processed = correction_pattern.sub(r'\1', text)  # Keep original
        processed = re.sub(r"&[+-]", "", processed)  # Remove disfluency markers
    elif version == "target":
        processed = correction_pattern.sub(r'\2', text)  # Use corrected
        processed = re.sub(r"&[+-][\w'-]+", "", processed)  # Also remove paraphasias

    return processed.strip()

def get_two_cha_versions(text: str) -> tuple[str, str]:
    """
    Generates both semantic-cleaned and phonological-cleaned versions
    of a CLAN utterance.

    Args:
        text (str): Raw CHAT-formatted text.

    Returns:
        tuple[str, str]: (cleaned_target, cleaned_phon)
    """
    processed_cha_target = process_clan_text(text, version="target")
    processed_cha_phon = process_clan_text(text, version="phon")
    cleaned = clean_text(processed_cha_target)
    cleaned_phon = clean_text(processed_cha_phon)
    return cleaned, cleaned_phon


def calc_props(num_data, total):
    if total > 0:
        prop_data = {
            col.replace("num_", "prop_"): num_data[col] / total for col in num_data if col.startswith("num_")
        }
        return prop_data
    else:
        logger.warning(f"No nonzero total provided for proportion calculation in results {num_data}")
        return {}
    
def get_most_common(counter, num, label):
    results = {}
    most_common = counter.most_common(num)
    for i in range(num):
        try:
            results[f"rank{i+1}_commonest_{label}"] = most_common[i][0]
            results[f"rank{i+1}_commonest_{label}_count"] = most_common[i][1]
        except:
            pass
    return results

def matrix_metrics(mat, units, label):
    """
    Computes statistical metrics from a given distance/similarity matrix.

    This function is generalizable to both token-level and sentence-level similarity matrices.

    Args:
        mat (np.array): Distance or similarity matrix (sentence-sentence or token-token).
        units (list): List of linguistic units (sentences or tokens).
        label (str): Label for result keys (e.g., 'semantic_similarity', 'intrasentential_similarity').

    Returns:
        dict: Dictionary containing min, max, mean, variance, std_dev, cv, and weighted metrics.
    """
    results = {}

    try:
        # Ensure matrix is in float32 format
        mat = np.array(mat, dtype=np.float32)

        nonzero_values = mat[np.nonzero(mat)].astype(np.float32)  # Convert to float explicitly

        empty_results = {
                f"{label}_min": None, f"{label}_max": None, f"{label}_mean": None, f"{label}_median": None,
                f"{label}_var": None, f"{label}_std_dev": None, f"{label}_cv": None,
                f"{label}_weighted_mean_dist": None, f"{label}_normalized_diversity": None
            }

        if len(nonzero_values) == 0:
            logger.warning(f"No nonzero values found for {label}. Returning default None values.")
            return empty_results

        # Ensure valid unit lengths (tokens or sentences)
        unit_lengths = np.array([len(unit) if hasattr(unit, '__len__') else 1 for unit in units], dtype=np.float32)

        # Extract correct sentence or token indices
        nonzero_indices = np.nonzero(mat)
        if len(nonzero_indices) == 0:
            logger.warning(f"No nonzero indices found for {label}. Returning default None values.")
            return empty_results            
        
        valid_weights = unit_lengths[nonzero_indices[0]]

        if len(valid_weights) != len(nonzero_values):  # Double-check alignment
            logger.warning(f"Mismatch in weights vs. nonzero values for {label}. Using uniform weights.")
            valid_weights = np.ones_like(nonzero_values, dtype=np.float32)

        # Compute statistical metrics
        results.update({
            f"{label}_min": float(np.min(nonzero_values)),  # Ensure output is float
            f"{label}_max": float(np.max(nonzero_values)),
            f"{label}_mean": float(np.mean(nonzero_values)),
            f"{label}_median": float(np.median(nonzero_values)),
            f"{label}_var": float(np.var(nonzero_values)),
            f"{label}_std_dev": float(np.std(nonzero_values)),
            f"{label}_cv": float(np.std(nonzero_values) / np.mean(nonzero_values)) if np.mean(nonzero_values) != 0 else 0,
            f"{label}_weighted_mean_dist": float(np.average(nonzero_values, weights=valid_weights)),
            f"{label}_normalized_diversity": float(np.mean(nonzero_values) / (np.mean(valid_weights) if np.mean(valid_weights) != 0 else 1))
        })

    except Exception as e:
        logger.error(f"Error computing metrics for {label}: {e}")
        return {}

    return results
