import numpy as np
from collections import Counter
import logging
logger = logging.getLogger("CustomLogger")
from backend.nlp.NLPmodel import NLPmodel
from backend.nlp.data_processing import calc_props, get_most_common
from elasticas.analyses.ngrams import compute_ngrams


def estimate_mlu(doc):
    """
    Estimate Mean Length of Utterance (MLU) from a spaCy Doc.
    
    MLU = total morphemes / number of utterances (sentences)

    This is a rough approximation based on:
    - 1 morpheme per token
    - +1 per morphological feature (as proxy for bound morphemes)

    Args:
        doc (Doc): A spaCy processed Doc object with sentence boundaries.

    Returns:
        dict: A dictionary with the estimated MLU.
    """
    if not doc or not list(doc.sents):
        logger.warning("Document is empty or lacks sentence boundaries.")
        return {}

    total_morphemes = 0
    sents = list(doc.sents)

    for sent in sents:
        for token in sent:
            morph_features = str(token.morph)
            morph_count = 1 + len(morph_features.split("|")) if morph_features else 1
            total_morphemes += morph_count

    mlu = total_morphemes / len(sents)
    return round(mlu, 2)

def analyze_spacy_features(doc, num, feature_type="POS"):
    """
    Generalized function to analyze POS tags or dependency parsing features in a given `spaCy` doc.

    Args:
        doc (spacy.Doc): The processed text document.
        tokens (list): List of tokenized words.
        results (dict): Dictionary to store analysis results.
        feature_type (str): Feature to analyze ("POS" for POS tags, "DEP" for dependency parsing).

    Returns:
        dict: Updated results dictionary with feature statistics.
    """

    num_tokens = len([t for t in doc])

    if num_tokens < 1:
        logger.warning(f"No tokens to analyze {feature_type}.")
        return {}

    if feature_type == "POS":
        feature_tags = Counter(t.pos_ for t in doc)
        feature_prefix = "POStag"
        diversity_key = "pos_diversity"
        unique_key = "unique_pos_tag_count"

    elif feature_type == "DEP":
        feature_tags = Counter(t.dep_ for t in doc)
        feature_prefix = "Deptag"
        diversity_key = "dep_diversity"
        unique_key = "unique_dep_tag_count"

    else:
        raise ValueError("Invalid feature_type. Use 'POS' or 'DEP'.")

    dict_pfx = f"{feature_type.lower()}"
    func_data = {f"{dict_pfx}_tag_counts":{}, 
                 f"{dict_pfx}_tag_props":{},
                 f"{dict_pfx}_tags_commonest":{}}

    feature_types = {f"num_{feature_prefix}_{tag}": count for tag, count in feature_tags.items()}
    total_tags = sum(feature_types.values())
    func_data[f"{dict_pfx}_tag_counts"][f"total_{feature_prefix}s"] = total_tags
    func_data[f"{dict_pfx}_tag_counts"][unique_key] = len(feature_types)
    func_data[f"{dict_pfx}_tag_counts"][diversity_key] = len(feature_tags) / num_tokens if num_tokens > 0 else 0
    
    func_data[f"{dict_pfx}_tag_counts"].update(feature_types)
    func_data[f"{dict_pfx}_tag_props"].update(calc_props(feature_types, total_tags))
    func_data[f"{dict_pfx}_tags_commonest"].update(get_most_common(feature_tags, num, feature_prefix))

    return func_data

def morphological_analysis(doc, num):
    """
    Analyzes morphological features in a given `spaCy` doc.

    Args:
        doc (spacy.Doc): The processed text document.
        results (dict): Dictionary to store morphological analysis results.

    Returns:
        dict: Updated results dictionary with morphological statistics.
    """
    func_data = {"morpheme_basic_specs": {}, "morph_tag_counts": {}, "morph_tag_props": {}, "morph_tags_commonest": {}, "morph_tag_sets_commonest":{}}

    morphs = [str(t.morph) for t in doc]
    pooled_morphs = [m for mset in morphs for m in mset.split("|")] 
    morph_types = {f"num_Mtag_{mtype.replace('=','_')}": count for mtype, count in Counter(pooled_morphs).items()} 

    total_morph_tags = len(pooled_morphs)
    func_data["morpheme_basic_specs"]["total_morph_tags"] = total_morph_tags
    unique_morph_tags = len(set(pooled_morphs))
    func_data["morpheme_basic_specs"]["unique_morph_tag_count"] = unique_morph_tags
    func_data["morpheme_basic_specs"]["total_tag_sets"] = len(morphs)
    func_data["morpheme_basic_specs"]["unique_tag_set_count"] = len(set(morphs))
    func_data["morpheme_basic_specs"]["avg_tags_per_word"] = np.mean([len(m.split("|")) for m in morphs]) if morphs else 0
    func_data["morpheme_basic_specs"]["morph_tag_diversity"] = unique_morph_tags / total_morph_tags if total_morph_tags > 0 else 0
    num_words_with_tags = sum(1 for m in morphs if m)
    func_data["morpheme_basic_specs"]["num_morph_tagged_words"] = num_words_with_tags
    func_data["morpheme_basic_specs"]["ratio_morph_tagged_words"] = num_words_with_tags / len(doc) if len(doc) > 0 else 0

    func_data["morpheme_basic_specs"]["mlu_estimate"] = estimate_mlu(doc)

    func_data["morph_tag_counts"].update(morph_types)
    func_data["morph_tag_props"].update(calc_props(morph_types, total_morph_tags))
    func_data["morph_tags_commonest"].update(get_most_common(Counter(pooled_morphs), num, "morph_tag"))
    func_data["morph_tag_sets_commonest"].update(get_most_common(Counter(morphs), num, "morph_tag_set"))

    return func_data

def analyze_morphology(PM, sample_data):
    """
    Perform morphosyntactic analysis on a preprocessed text sample.

    Args:
        sample_row (dict): Dictionary containing sample data, including 'sentence_preprocessed'.

    Returns:
        dict: Morphosyntactic analysis results.
    """
    try:
        results = PM.sections["morphology"].init_results_dict()

        NLP = NLPmodel()
        nlp = NLP.get_nlp()

        if PM.sentence_level:
            if not isinstance(sample_data, list):
                raise ValueError("Expected a list of sentence dicts for sentence-level analysis.")
            
            doc_cleaned = ""
            doc_id = sample_data[0].get("doc_id")            
            
            for sent in sample_data:
                sent_id = sent.get("sent_id")
                cleaned = sent.get("cleaned", "")

                doc = nlp(cleaned)
                sent_data_base = {"doc_id": doc_id, "sent_id": sent_id}
                func_data = morphological_analysis(doc, 5)
                func_data.update(analyze_spacy_features(doc, 5, "POS"))

                pos_tags = [t.pos_ for t in doc]
                summary_data, ngram_data = compute_ngrams(PM, pos_tags, sent_data_base.copy(), "pos", "sent")
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
            
            doc_cleaned = sample_data.get("cleaned", "")
            doc_id = sample_data.get("doc_id")

        doc_data_base = {"doc_id": doc_id}
        doc = nlp(doc_cleaned)
        func_data = {}
        func_data.update(morphological_analysis(doc, 10))
        func_data.update(analyze_spacy_features(doc, 10, "POS"))

        pos_tags = [t.pos_ for t in doc]
        summary_data, ngram_data = compute_ngrams(PM, pos_tags, doc_data_base, "pos", "doc")
        func_data.update(summary_data)

        for table, row_data in func_data.items():
            doc_data = doc_data_base.copy()
            doc_data.update(row_data)
            results[f"{table}_doc"].update(doc_data)
        
        for table, data in ngram_data.items():
            for row_data in data:
                doc_ngram_data = doc_data_base.copy()
                doc_ngram_data.update(row_data)
                results[f"{table}_doc"].append(doc_ngram_data)   

        logger.info(f"Morphological analysis completed successfully.")
        return results

    except Exception as e:
        logger.error(f"Error analyzing morphology: {e}")
        return {}
