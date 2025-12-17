import numpy as np
import language_tool_python
from collections import Counter
import logging
logger = logging.getLogger("CustomLogger")
from alastr.backend.nlp.NLPmodel import NLPmodel
from alastr.backend.nlp.data_processing import get_most_common


def apply_language_tool(doc, num):
    """
    Apply LanguageTool to detect grammar errors in sentences.

    Args:
        sentences (list): List of sentences.
        results (dict): Dictionary to store grammar analysis results.

    Returns:
        dict: Updated results dictionary with grammar error statistics.
    """
    try:
        results = {}

        tool = language_tool_python.LanguageTool('en-US')

        rules = Counter()
        cats = Counter()
        num_matches = []
        error_lengths = []

        sentences = [sent for sent in doc.sents]

        for sent in sentences:
            try:
                matches = tool.check(sent.text)
                if matches:
                    num_matches.append(len(matches))
                    for m in matches:
                        rules[m.ruleId] += 1
                        cats[m.category] += 1
                        error_lengths.append(m.errorLength)
            except Exception as e:
                logger.warning(f"LanguageTool error on sentence: {sent} | Error: {e}")

        # Compute sentence-level statistics
        results['num_lgtool_errors'] = sum(num_matches)
        results['num_sents_w_error'] = len(num_matches)
        results['avg_errors_per_sent'] = np.nanmean(num_matches) if np.nanmean(num_matches) > 0 else None
        results['prop_sent_w_error'] = len(num_matches) / len(sentences) if len(sentences) > 0 else None

        results['total_error_length'] = sum(error_lengths)
        results['avg_error_len_per_sent'] = sum(error_lengths) / len(sentences) if len(sentences) > 0 else None

        results.update(get_most_common(rules, num, "LgToolError"))
        results.update(get_most_common(cats, num, "LgToolCategory"))

        # Track occurrences of specific errors
        for error, count in rules.items():
            results[f"num_error_{error}"] = count
            results[f"prop_error_{error}"] = count / results['num_lgtool_errors'] if results['num_lgtool_errors'] > 0 else None
        
        for category, count in cats.items():
            results[f"num_category_{category}"] = count
            results[f"prop_category_{error}"] = count / results['num_lgtool_errors'] if results['num_lgtool_errors'] > 0 else None

        tool.close()

        logger.info(f"LanguageTool analysis completed. Total errors: {results['num_lgtool_errors']}")
        return results

    except Exception as e:
        logger.error(f"Error in apply_language_tool: {e}")
        return results  # Return partial results in case of failure

def analyze_mechanics(PM, sample_data):
    """
    Perform grammar analysis on a sample text.

    Args:
        sample_row (dict): Dictionary containing sample data.

    Returns:
        dict: Grammar analysis results.
    """
    try:
        results = PM.sections["mechanics"].init_results_dict()

        NLP = NLPmodel()
        nlp = NLP.get_nlp()

        if isinstance(sample_data, list):
            sample_data = sample_data[0]

        if sample_data.get("cleaned_phon", ""):
            doc_cleaned = sample_data.get("cleaned_phon", "")
        else:
            doc_cleaned = sample_data.get("cleaned", "")
        
        doc_id = sample_data.get("doc_id")
        doc = nlp(doc_cleaned)

        func_data = {}
        func_data["lg_tool"] = apply_language_tool(doc, 5)
        doc_data_base = {"doc_id": doc_id}
        for table, row_data in func_data.items():
            doc_data = doc_data_base.copy()
            doc_data.update(row_data)
            results[f"{table}_doc"].update(doc_data)

        logger.info(f"Mechanics analysis completed: {results}")
        return results

    except Exception as e:
        logger.error(f"Error analyzing mechanics: {e}")
        return {}
