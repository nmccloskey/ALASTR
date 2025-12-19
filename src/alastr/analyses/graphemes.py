import re
from collections import Counter
from alastr.backend.tools.logger import logger
from alastr.backend.nlp.data_processing import calc_props, get_most_common
from alastr.analyses.ngrams import compute_ngrams


SQL_PROBLEM_CHARS = {
    ',': "comma",
    '.': "period",
    ';': "semicolon",
    "?": "qst_mark",
    '"': "doublequote",
    "'": "singlequote",
    '\\': "backslash",
    '%': "percentage",
    '_': "underscore",
    "\t": "tab",
    "\n": "newline",
    "\r": "return"
}

def sanitize_character(c: str) -> str:
    """Converts problematic characters to a safe format for dictionary keys."""
    if c in SQL_PROBLEM_CHARS:
        return f"{SQL_PROBLEM_CHARS[c].upper()}"
    if not c.isprintable():
        return f"_U+{ord(c):04X}_"  # Convert to Unicode representation
    return c

def count_graphemes(text):
    """
    Count various types of graphemes in the given text, handling potential SQL-related character issues.

    Args:
        text (str): The input text to analyze.
    
    Returns:
        dict: A dictionary containing various grapheme statistics.
    """
    try:
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
        
        func_data = {"grapheme_basic_specs": {},
                   "grapheme_counts": {}, "grapheme_props": {}, "grapheme_modes": {},
                   "word_counts": {}, "word_props": {}} #, "word_modes": {}}
        gcounts = Counter(text)
        num_graphemes = len(text)

        func_data["grapheme_basic_specs"]["total_graphemes"] = num_graphemes
        
        if num_graphemes == 0:
            logger.error("No graphemes to count.")
            return func_data
    
        func_data["grapheme_basic_specs"]["unique_grapheme_count"] = len(gcounts)
        grapheme_lists = [list(t) for t in text.split(" ") if t != " "]
        unique_grapheme_lengths = [len(set(gl)) for gl in grapheme_lists]
        func_data["grapheme_basic_specs"]["min_unique_graphemes_per_word"] = min(unique_grapheme_lengths)
        func_data["grapheme_basic_specs"]["max_unique_graphmes_per_word"] = max(unique_grapheme_lengths)
        func_data["grapheme_basic_specs"]["avg_unique_graphemes_word"] = sum(unique_grapheme_lengths) / len(unique_grapheme_lengths) if unique_grapheme_lengths else 0

        func_data["grapheme_basic_specs"]["num_alphabetic"] = len(re.findall(r'[A-Za-z]', text))
        func_data["grapheme_basic_specs"]["num_digits"] = len(re.findall(r'\d', text))
        func_data["grapheme_basic_specs"]["num_punctuation"] = len(re.findall(r'[.,;!?]', text))
        func_data["grapheme_basic_specs"]["num_spaces"] = text.count(" ")
        func_data["grapheme_basic_specs"]["num_uppercase"] = sum(1 for c in text if c.isupper())

        gcounts = Counter(text.replace(" ", "").upper())
        func_data["grapheme_counts"].update({f"num_{sanitize_character(g)}": c for g, c in gcounts.items()})

        func_data["grapheme_props"].update(calc_props(func_data["grapheme_counts"], num_graphemes))

        # Count words by number of graphemes
        grapheme_word_distribution = Counter(len(gl) for gl in grapheme_lists)
        for grapheme_count, count in grapheme_word_distribution.items():
            func_data["word_counts"][f"num_{grapheme_count}grapheme_words"] = count
            func_data["word_props"][f"prop_{grapheme_count}grapheme_words"] = count / len(grapheme_lists)

        func_data["grapheme_modes"].update(get_most_common(gcounts, 5, "grapheme"))
    
        return func_data

    except Exception as e:
        logger.error(f"Error processing text: {e}")
        return {"error": str(e)}

def analyze_graphemes(PM, sample_data):
    """
    Performs graphemic analysis on document or sentence-level data.

    Args:
        sample_data (dict or list of dict): If sentence-level, a list of sentence dicts.
                                            If doc-level, a single document dict.

    Returns:
        dict: A dictionary of table names mapped to processed data (doc or sent level).
    """
    try:
        results = PM.sections["graphemes"].init_results_dict()

        if PM.sentence_level:
            if not isinstance(sample_data, list):
                raise ValueError("Expected a list of sentence dicts for sentence-level analysis.")

            doc_cleaned = ""
            doc_id = sample_data[0].get("doc_id")

            for sent in sample_data:
                sent_id = sent.get("sent_id")
                
                if sent.get("cleaned_phon", ""):
                    cleaned = sent.get("cleaned_phon", "")
                else:
                    cleaned = sent.get("cleaned", "")

                sent_data_base = {"doc_id": doc_id, "sent_id": sent_id}
                func_data = count_graphemes(cleaned)

                summary_data, ngram_data = compute_ngrams(PM, list(cleaned), sent_data_base, "grapheme", "sent")
                func_data.update(summary_data)

                for table, row_data in func_data.items():
                    sent_data = sent_data_base.copy()
                    sent_data.update(row_data)
                    results[f"{table}_sent"].append(sent_data)

                for table, data in ngram_data.items():
                    for row_data in data:
                        sent_ngram_data = sent_data_base.copy()
                        sent_ngram_data.update(row_data)
                        results[f"{table}_sent"].append(sent_ngram_data)                        

                doc_cleaned += " " + cleaned

            doc_cleaned = doc_cleaned.strip()

        else:
            if not isinstance(sample_data, dict):
                raise ValueError("Expected a single dict for document-level analysis.")
            
            doc_id = sample_data.get("doc_id")
            
            if sample_data.get("cleaned_phon", ""):
                doc_cleaned = sample_data.get("cleaned_phon", "")
            else:
                doc_cleaned = sample_data.get("cleaned", "")

        doc_data_base = {"doc_id": doc_id}
        func_data = count_graphemes(doc_cleaned)

        summary_data, ngram_data, current_ngram_id = compute_ngrams(PM, list(doc_cleaned), doc_data_base, "grapheme", "doc")
        func_data.update(summary_data)
        PM.ngram_id_doc = current_ngram_id

        for table, row_data in func_data.items():
            doc_data = doc_data_base.copy()
            doc_data.update(row_data)
            results[f"{table}_doc"].update(doc_data)

        for table, data in ngram_data.items():
            for row_data in data:
                doc_ngram_data = doc_data_base.copy()
                doc_ngram_data.update(row_data)
                results[f"{table}_doc"].append(doc_ngram_data)   

        logger.info(f"Graphemic analysis completed successfully.")
        return results

    except Exception as e:
        logger.error(f"Error analyzing graphemes: {e}")
        return {}
