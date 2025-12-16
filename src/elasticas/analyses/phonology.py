import re
import numpy as np
from g2p_en import G2p
from collections import Counter
import logging
logger = logging.getLogger("CustomLogger")
from elasticas.backend.nlp.NLPmodel import NLPmodel
from elasticas.backend.nlp.data_processing import calc_props, get_most_common


# Mapping phonemes to phonetic feature classes
PHONETIC_FEATURES_IPA = {
    "vowel": "aeiou",
    "nasal": "mnŋɲɳɴ",
    "stop": "pbtdkgʔ",
    "fricative": "fvszʃʒθðhɸβχʁ",
    "affricate": "tʃdʒtsdz",
    "liquid": "lrɹʎʟ",
    "glide": "jwɥ",
    "sonorant": "lmnrjwɥ",
    "obstruent": "pbtdkgfvszʃʒθðhɸβχʁ",
}

# Define phonetic features using ARPAbet mappings
PHONETIC_FEATURES_ARPABET = {
    "vowel": {"AA", "AE", "AH", "AO", "AW", "AX", "AXR", "AY", "EH", "ER", "EY", "IH", "IX", "IY", "OW", "OY", "UH", "UW", "UX"},
    "nasal": {"M", "EM", "N", "EN", "NG", "NX"},
    "stop": {"P", "B", "T", "D", "K", "G", "Q"},
    # "tap": {"DX"},
    "fricative": {"F", "V", "TH", "DH", "S", "Z", "SH", "ZH", "HH"},
    "affricate": {"CH", "JH"},
    "liquid": {"L", "R", "EL"},
    "glide": {"W", "Y"},
    "voiceless": {"P", "T", "K", "F", "TH", "S", "SH", "CH", "HH"},
    "voiced": {"B", "D", "G", "V", "DH", "Z", "ZH", "JH", "M", "N", "NG", "L", "R", "W", "Y"},
}

def analyze_phonemes(doc):
    """
    Extract phonological features from a given text using ARPAbet.

    Args:
        text (str): Input text.

    Returns:
        dict: Dictionary of phoneme counts, phonetic feature proportions, and phoneme-word distributions.
    """
    try:
        tokens = [token.text.lower() for token in doc if token.is_alpha]

        if len(tokens) < 1:
            logger.warning("Not enough tokens to analyze phonology - skipping.")
            return {}

        g2p = G2p()
        phonemized_tokens = [g2p(t) for t in tokens if t.isalpha()]
        pt_lengths = [len(pt) for pt in phonemized_tokens]
        unique_pt_lengths = [len(set(pt)) for pt in phonemized_tokens]
        phoneme_list = [p for t in phonemized_tokens for p in t if p != ' ']
        phoneme_counts = Counter(phoneme_list)
        total_phonemes = sum(phoneme_counts.values())
        unique_phonemes = len(phoneme_counts)
        
        func_data ={"phoneme_basic_specs":{}, "phoneme_counts":{}, "phoneme_props":{}, "phoneme_commonest":{},
                    "phon_feature_counts":{}, "phon_feature_props":{}, "word_lens_counts":{}, "word_lens_props":{}}

        func_data["phoneme_basic_specs"] = {
            "total_phonemes": total_phonemes,
            "unique_phoneme_count": unique_phonemes,
            "min_phonemes_per_word": min(pt_lengths) if pt_lengths else np.nan,
            "max_phonemes_per_word": max(pt_lengths) if pt_lengths else np.nan,
            "avg_phonemes_per_word": total_phonemes / len(tokens) if tokens else 0,
            "var_phonemes_per_word": np.var(pt_lengths) if len(pt_lengths) > 2 else np.nan,
            "min_unique_phonemes_per_word": min(unique_pt_lengths) if unique_pt_lengths else np.nan,
            "max_unique_phonemes_per_word": max(unique_pt_lengths) if unique_pt_lengths else np.nan,
            "avg_unique_phonemes_per_word": sum(unique_pt_lengths) / len(tokens) if tokens else 0,
            "var_unique_phonemes_per_word": np.var(unique_pt_lengths) if len(unique_pt_lengths) > 2 else np.nan
        }

        # Individual phoneme counts (including stress-marked versions)
        total_phoneme_counts = Counter()  # For stress-independent tallies
        for phoneme, count in phoneme_counts.items():
            if re.match(r'^\w+$', phoneme):
                func_data["phoneme_counts"][f"num_{phoneme}"] = count  # Keep stress-marked version
                if re.search(r'\d', phoneme):
                    base_phoneme = re.sub(r'\d', '', phoneme)  # Remove stress numbers
                    # Track stress-independent count
                    total_phoneme_counts[base_phoneme] += count
            else:
                logger.warning(f"Cannot parse '{phoneme}' as phoneme.")

        # Add stress-independent phoneme totals
        for phoneme, count in total_phoneme_counts.items():
            func_data["phoneme_counts"][f"num_total_{phoneme}"] = count
        
        func_data["phoneme_props"].update(calc_props(func_data["phoneme_counts"], total_phonemes))

        # Track phonetic feature counts
        feature_counts = {feature: 0 for feature in PHONETIC_FEATURES_ARPABET}

        for phoneme, count in phoneme_counts.items():
            for feature, members in PHONETIC_FEATURES_ARPABET.items():
                # Remove stress indicator
                if re.sub(r'\d','',phoneme) in members:
                    feature_counts[feature] += count

        # Add phonetic feature counts and proportions
        for feature, count in feature_counts.items():
            func_data["phon_feature_counts"][f"num_{feature}"] = count
            func_data["phon_feature_props"][f"prop_{feature}"] = count / total_phonemes if total_phonemes > 0 else 0

        # Count words by number of phonemes
        phoneme_word_distribution = Counter(pt_lengths)
        for phoneme_count, count in phoneme_word_distribution.items():
            func_data["word_lens_counts"][f"num_{phoneme_count}phoneme_words"] = count
            func_data["word_lens_props"][f"prop_{phoneme_count}phoneme_words"] = count / len(tokens)
        
        func_data["phoneme_commonest"].update(get_most_common(phoneme_counts, 5, "phoneme"))

        # logger.info(f"Phoneme analysis completed successfully.")
        return func_data

    except Exception as e:
        logger.error(f"Error in phonological analysis: {e}")
        return {}

def count_syllables(word):
    """
    Get syllable count and stress pattern for a word using CMU Pronouncing Dictionary.

    Args:
        word (str): The word to analyze.

    Returns:
        tuple: (syllable count, primary stress count, secondary stress count)
    """
    NLP = NLPmodel()
    cmu_dict = NLP.get_cmu_dict()
    # cmu_dict = NLPmodel.get_cmu_dict()

    word = word.lower()
    
    if word in cmu_dict:
        syllable_counts = []
        primary_stress = 0
        secondary_stress = 0
        
        for pronunciation in cmu_dict[word]:
            syllables = [p for p in pronunciation if p[-1].isdigit()]
            syllable_counts.append(len(syllables))

            # Count primary (1) and secondary (2) stress markers
            primary_stress += sum(1 for p in syllables if "1" in p)
            secondary_stress += sum(1 for p in syllables if "2" in p)

        # Return max syllable count to handle multiple pronunciations
        return max(syllable_counts), primary_stress, secondary_stress
    
    return 1, 0, 0  # Default: unknown words = 1 syllable, no stress info

def analyze_syllables(doc):
    """
    Analyze syllable structure and stress patterns in a spaCy `Doc`.

    Args:
        doc (spacy.Doc): Tokenized document.

    Returns:
        dict: Syllable statistics including total count, average, and breakdown by word length.
    """
    func_data = {}
    func_data["syllable_stats"] = {
        "total_syllables": 0,
        "avg_syllables_per_word": 0,
        "total_primary_stress": 0,
        "total_secondary_stress": 0
    }

    syllable_counts = []

    tokens = [t for t in doc if t.is_alpha]

    if not tokens:
        logger.error(f"No tokens for syllable analysis.")
        return {}
    
    for token in tokens:
        if token.is_alpha:  # Ignore punctuation/numbers
            num_syllables, primary_stress, secondary_stress = count_syllables(token.text)
            syllable_counts.append(num_syllables)

            # Stress totals
            func_data["syllable_stats"]["total_primary_stress"] += primary_stress
            func_data["syllable_stats"]["total_secondary_stress"] += secondary_stress

            # Categorize words by syllable count
            att = f"num_{num_syllables}syllable_words"
            if att not in func_data["syllable_stats"].keys():
                func_data["syllable_stats"][att] = 1
            else:
                func_data["syllable_stats"][att] += 1
    
    for att in list(func_data["syllable_stats"].keys()):
        if att.endswith('syllable_words'):
            func_data["syllable_stats"][att.replace("num_", "prop_")] = func_data["syllable_stats"][att] / len(tokens)

    # Compute overall stats
    func_data["syllable_stats"]["total_syllables"] = sum(syllable_counts)
    func_data["syllable_stats"]["avg_syllables_per_word"] = sum(syllable_counts) / len(syllable_counts) if syllable_counts else 0

    return func_data

def analyze_phonology(PM, sample_data):

    try:
        results = PM.sections["phonology"].init_results_dict()

        NLP = NLPmodel()
        nlp = NLP.get_nlp()

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

                doc = nlp(cleaned)
                sent_data_base = {"doc_id": doc_id, "sent_id": sent_id}
                func_data = analyze_syllables(doc)
                func_data.update(analyze_phonemes(doc))

                for table, row_data in func_data.items():
                    sent_data = sent_data_base.copy()
                    sent_data.update(row_data)
                    results[f"{table}_sent"].append(sent_data)

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
        doc = nlp(doc_cleaned)
        func_data = analyze_syllables(doc)
        func_data.update(analyze_phonemes(doc))

        for table, row_data in func_data.items():
            doc_data = doc_data_base.copy()
            doc_data.update(row_data)
            results[f"{table}_doc"].update(doc_data)

        logger.info(f"Phonological analysis completed successfully.")
        return results

    except Exception as e:
        logger.error(f"Error analyzing phonemes: {e}")
        return {}
