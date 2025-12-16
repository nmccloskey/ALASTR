import numpy as np
from collections import Counter
import logging
logger = logging.getLogger("CustomLogger")
from elasticas.backend.nlp.NLPmodel import NLPmodel
from lexicalrichness import LexicalRichness
from wordfreq import word_frequency, zipf_frequency
from elasticas.backend.nlp.data_processing import get_most_common
# from readability import Readability
import readability as Readability
import textstat as tx
from elasticas.analyses.ngrams import compute_ngrams


def calculate_frequencies(doc, label):
    """
    Compute word frequency, zipf frequency, and weighted frequencies.
    """
    tokens = [token.text for token in doc if token.is_alpha]

    if len(tokens) <= 1:
        logger.warning("Not enough tokens to calculate freqs - skipping.")
        return {}

    total_words = len(tokens)
    unique_words = set(tokens)
    freqs = {
        w: (word_frequency(w, 'en', wordlist='best'),
            zipf_frequency(w, 'en', wordlist='best'),
            tokens.count(w)) for w in unique_words
    }
    
    word_frequencies = [freqs[w][0] for w in tokens]
    weighted_freqs = [freqs[w][0] * (freqs[w][2] / len(tokens)) for w in freqs]
    zipf_frequencies = [freqs[w][1] for w in tokens]
    weighted_zipfs = [freqs[w][1] * (freqs[w][2] / len(tokens)) for w in freqs]

    rare_words = {freqs.get(word, (0, 0, 0))[1]:word for word in tokens if freqs.get(word, (0, 0, 0))[1] <= 3}

    func_data = {
        f"{label}_rare_words_count": len(rare_words),
        f"{label}_rare_word_ratio": len(rare_words) / total_words if total_words > 0 else 0,
        f"{label}_avg_word_freq": np.mean(word_frequencies) if word_frequencies else 0,
        f"{label}_var_word_freq": np.var(word_frequencies) if len(word_frequencies) > 1 else 0,
        f"{label}_weighted_avg_word_freq": np.mean(weighted_freqs) if weighted_freqs else 0,
        f"{label}_var_weighted_word_freq": np.var(weighted_freqs) if len(weighted_freqs) > 1 else 0,
        f"{label}_avg_zipf_freq": np.mean(zipf_frequencies) if zipf_frequencies else 0,
        f"{label}_var_zipf_freq": np.var(zipf_frequencies) if len(zipf_frequencies) > 1 else 0,
        f"{label}_weighted_avg_zipf_freq": np.mean(weighted_zipfs) if weighted_zipfs else 0,
        f"{label}_var_weighted_zipf_freq": np.var(weighted_zipfs) if len(weighted_zipfs) > 1 else 0,
    }

    func_data.update(get_most_common(Counter(tokens), 5, f"word_{label}"))

    return func_data

def compute_lexical_richness(doc, label):
    """
    Compute lexical richness measures for a given text.

    Args:
        text (str): The text to analyze.

    Returns:
        dict: A dictionary containing lexical richness measures.
    """
    try:
        func_data = {}
        
        tokens = [token.text.lower() for token in doc if token.is_alpha]

        if len(tokens) <= 1:
            logger.warning("Not enough tokens to compute lexical richness - skipping.")
            return {}

        token_lengths = [len(t) for t in tokens]
        func_data[f"num_words_{label}_spacy"] = len(tokens)
        func_data[f"{label}_unique_word_count_spacy"] = len(tokens)
        func_data[f"{label}_min_word_length"] = min(token_lengths)
        func_data[f"{label}_max_word_length"] = max(token_lengths)
        func_data[f"{label}_avg_word_length"] = sum(token_lengths) / len(tokens)
        func_data[f"{label}_var_word_length"] = np.var(token_lengths)

        if label == "cleaned":
            stop_words = [token.text.lower() for token in doc if token.is_alpha and token.is_stop]
            func_data[f"{label}_num_stop_words"] = len(stop_words)
            func_data[f"lexical_density"] = (len(tokens)-len(stop_words))/len(tokens)

        lex = LexicalRichness(doc.text)
        more_func_data = {
            f"num_words_{label}_textblob": lex.words,
            f"{label}_unique_word_count_spacy": lex.terms,
            f"{label}_ttr": lex.ttr,
            f"{label}_rttr": lex.rttr,
            f"{label}_cttr": lex.cttr,
            f"{label}_msttr": lex.msttr(segment_window=min(25, lex.words-1)),
            f"{label}_mattr": lex.mattr(window_size=min(25, lex.words-1)),
            f"{label}_mtld": lex.mtld(threshold=0.72),
            f"{label}_hdd": lex.hdd(draws=min(42, lex.words)),
            f"{label}_herdan": lex.Herdan,
            f"{label}_summer": lex.Summer,
            f"{label}_maas": lex.Maas,
            f"{label}_yulek": lex.yulek,
            f"{label}_herdanvm": lex.herdanvm,
            f"{label}_simpsond": lex.simpsond,
        }

        func_data.update(more_func_data)
        
        try:
            yulei = lex.yulei
            if yulei != np.inf:
                func_data.update({f"{label}_yulei": lex.yulei})
        except Exception as e:
            logger.warning(f"Error in a yulei calculation: {e}")
        try:
            func_data.update({f"{label}_dugast": lex.Dugast})
        except Exception as e:
            logger.warning(f"Error in dugast calculation: {e}")
        try:    
            func_data.update({f"{label}_vocd": lex.vocd()})  
        except Exception as e:
            logger.warning(f"Error in vocd calculation: {e}")      

        return func_data

    except Exception as e:
        logger.error(f"Error computing lexical richness: {e}")
        return {}

def process_named_entities(doc, num):
    """
    Extract named entity information including counts, types, and most common entities.
    """
    total_words = len([t for t in doc])
    NE_counts = Counter(ent.label_ for ent in doc.ents)
    NE_types = {f"NEtype_{netype}_count": count for netype, count in NE_counts.items()}
    
    most_common_NE = Counter(ent.text for ent in doc.ents)
    entity_lengths = [len(ent.text) for ent in doc.ents]

    results = {
        "num_NEs": len(doc.ents),
        "unique_NE_count": len(NE_types),
        "NE_density": len(doc.ents) / total_words if total_words > 0 else 0,
        "max_NE_length": max(entity_lengths, default=0),
        "min_NE_length": min(entity_lengths, default=0),
        "avg_NE_length": np.mean(entity_lengths) if entity_lengths else 0,
    }

    results.update(NE_types)
    results.update(get_most_common(most_common_NE, num, "NE"))

    return results

def safe_join(seq):
    return ", ".join(str(x) for x in seq)

def calc_readability(doc):
    """
    Calculates various readability metrics for a given spaCy Doc object.

    This function computes multiple readability scores (e.g., Flesch-Kincaid, Dale-Chall, ARI, etc.)
    and additional text complexity metrics (e.g., difficult words, reading time) for a document.
    It requires at least 100 alphabetic tokens in the document to produce results.

    Parameters
    ----------
    doc : spacy.tokens.Doc
        The spaCy Doc object to analyze.

    Returns
    -------
    dict
        A dictionary containing readability scores and related metrics. Returns an empty dictionary
        if the document contains fewer than 100 words or if an error occurs.

    Raises
    ------
    TypeError
        If the input is not a spaCy Doc object.
    """

    if not hasattr(doc, 'text') or not hasattr(doc, '__iter__'):
        raise TypeError("Input must be a spaCy Doc object.")

    tokens = [token.text for token in doc if token.is_alpha]

    if len(tokens) < 100:
        logger.warning("Readability calculations not applicable to documents of <100 words.")
        return {}

    try:
        func_data = {}
        r = Readability(doc.text)

        fk = r.flesch_kincaid()
        func_data["flesch_kincaid_score"] = fk.score
        func_data["fk_grade_level"] = fk.grade_level

        f = r.flesch()
        func_data["flesch_score"] = f.score
        func_data["flesch_ease"] = f.ease
        func_data["flesch_grade_levels"] = safe_join(f.grade_levels)

        dc = r.dale_chall()
        func_data["dale_chall_score"] = dc.score
        func_data["dc_grade_levels"] = safe_join(dc.grade_levels)

        ari = r.ari()
        func_data["ari_score"] = ari.score
        func_data["ari_grade_levels"] = safe_join(ari.grade_levels)
        func_data["ari_ages"] = safe_join(ari.ages)

        cl = r.coleman_liau()
        func_data["coleman_liau_score"] = cl.score
        func_data["cl_grade_level"] = cl.grade_level

        gf = r.gunning_fog()
        func_data["gunning_fog_score"] = gf.score
        func_data["gf_grade_level"] = gf.grade_level

        if len(list(doc.sents)) >= 30:
            smog = r.smog(all_sentences=True)
            func_data["smog_score"] = smog.score
            func_data["smog_grade_level"] = smog.grade_level

        s = r.spache()
        func_data["spache_score"] = s.score
        func_data["spache_grade_level"] = s.grade_level

        lw = r.linsear_write()
        func_data["linsear_write_score"] = lw.score
        func_data["lw_grade_level"] = lw.grade_level
    
    except Exception as e:
        logger.error(f"Failed to apply Readability library: {e}")

    try:
        func_data["text_standard"] = tx.text_standard(doc.text)
        func_data["num_difficult_words"] = tx.difficult_words(doc.text)
        func_data["prop_difficult_words"] = tx.difficult_words(doc.text) / len(doc)
        func_data["difficult_words"] = safe_join(tx.difficult_words_list(doc.text))
        func_data["mcalpine_eflaw"] = tx.mcalpine_eflaw(doc.text)
        func_data["reading_time"] = tx.reading_time(doc.text)
        func_data["LIX"] = tx.lix(doc.text)
        func_data["RIX"] = tx.rix(doc.text)
    except Exception as e:
        logger.error(f"Failed to apply textstat library: {e}")

    return func_data

def analyze_lexicon(PM, sample_data):
    """
    Perform lexical analysis.

    Args:
        doc_data (dict or list): If sentence-level, a list of dicts; if document-level, a single dict.
        sentence_level (bool): Whether to return sentence-level output.

    Returns:
        dict: Sentence-level and/or document-level lexical analysis results.
    """
    try:
        results = PM.sections["lexicon"].init_results_dict()
        
        NLP = NLPmodel()
        nlp = NLP.get_nlp()

        if PM.sentence_level:
            if not isinstance(sample_data, list):
                raise ValueError("Expected a list of sentence dicts for sentence-level analysis.")
            
            doc_cleaned = ""
            doc_tokenized = ""
            doc_id = sample_data[0].get("doc_id")
            
            for sent in sample_data:
                func_data = {}
                sent_id = sent.get("sent_id")
                cleaned = sent.get("cleaned", "")
                semantic = sent.get("semantic", "")
                sent_data_base = {"doc_id": doc_id, "sent_id": sent_id}
                
                doc = nlp(cleaned)
                tokens = [token.text for token in doc if token.is_alpha]
                func_data["freqs_cleaned"] = calculate_frequencies(doc, "cleaned")
                func_data["richness_cleaned"] = compute_lexical_richness(doc, "cleaned")
                func_data["named_entities"] = process_named_entities(doc, 3)

                doc = nlp(semantic)
                func_data["freqs_tokenized"] = calculate_frequencies(doc, "semantic")
                func_data["richness_tokenized"] = compute_lexical_richness(doc, "semantic")

                summary_data, ngram_data = compute_ngrams(PM, tokens, sent_data_base, "lex", "sent")
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
                
                doc_cleaned += " " + sent["cleaned"]
                doc_tokenized += " " + sent["semantic"] + "."
            
            doc_cleaned = doc_cleaned.strip()
            doc_tokenized = doc_tokenized.strip()
        
        else:
            if not isinstance(sample_data, dict):
                raise ValueError("Expected a single dict for document-level analysis.")
            
            doc_cleaned = sample_data.get("cleaned", "")
            doc_tokenized = sample_data.get("semantic", "")
            doc_id = sample_data.get("doc_id")

        func_data = {}
        doc_data_base = {"doc_id": doc_id}
            
        doc = nlp(doc_cleaned)
        tokens = [token.text for token in doc if token.is_alpha]
        func_data["freqs_cleaned"] = calculate_frequencies(doc, "cleaned")
        func_data["richness_cleaned"] = compute_lexical_richness(doc, "cleaned")
        func_data["named_entities"] = process_named_entities(doc, 10)
        func_data["readability"] = calc_readability(doc)

        doc = nlp(doc_tokenized)
        func_data["freqs_tokenized"] = calculate_frequencies(doc, "semantic")
        func_data["richness_tokenized"] = compute_lexical_richness(doc, "semantic")

        summary_data, ngram_data = compute_ngrams(PM, tokens, doc_data_base, "lex", "doc")
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

        logger.info(f"Lexical analysis completed.")
        return results

    except Exception as e:
        logger.error(f"Error analyzing lexicon: {e}")
        return {}
