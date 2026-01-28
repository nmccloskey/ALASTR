import numpy as np
from afinn import Afinn
from nrclex import NRCLex
from textblob import TextBlob
from collections import Counter
from alastr.backend.tools.logger import logger
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def apply_NRCLex(doc):
    """
    Apply NRCLex emotion analysis to a spaCy Doc.

    This function analyzes the emotional content of a given document using 
    NRCLex, extracting total emotion counts, proportions, and frequency-based 
    metrics such as max, min, average, median, variance, standard deviation, 
    and coefficient of variation.

    Args:
        doc (spacy.tokens.Doc): A spaCy document object containing tokenized sentences.

    Returns:
        dict: Dictionary with NRCLex emotion and sentiment scores.
    """
    try:
        subresults = {}
        total_words = 0

        emotions = [
            'anticipation', 'joy', 'positive', 'trust', 'surprise',
            'fear', 'anger', 'disgust', 'sadness', 'negative'
        ]

        raw = Counter()
        freqs = {emotion: [] for emotion in emotions}

        for sentence in doc.sents:
            nrc_obj = NRCLex(sentence.text)
            total_words += len(nrc_obj.words)

            for emotion in emotions:
                raw[emotion] += nrc_obj.raw_emotion_scores.get(emotion, 0)
                freqs[emotion].append(nrc_obj.affect_frequencies.get(emotion, 0))

        for emotion in emotions:
            subresults[f"NRCLex_{emotion}_total"] = raw[emotion]
            subresults[f"NRCLex_{emotion}_prop"] = raw[emotion] / total_words if total_words > 0 else 0

        for emotion in emotions:
            emotion_values = freqs[emotion] if freqs[emotion] else [0]

            subresults.update({
                f"NRC_max_{emotion}_freq": max(emotion_values),
                f"NRC_min_{emotion}_freq": min(emotion_values),
                f"NRC_avg_{emotion}_freq": np.nanmean(emotion_values),
                f"NRC_median_{emotion}_freq": np.median(emotion_values),
                f"NRC_var_{emotion}_freq": np.nanvar(emotion_values),
                f"NRC_std_dev_{emotion}_freq": np.std(emotion_values),
                f"NRC_cv_{emotion}_freq": np.std(emotion_values) / np.mean(emotion_values) if np.mean(emotion_values) != 0 else 0
            })

        # logger.info(f"NRCLex analysis completed successfully.")
        return subresults

    except Exception as e:
        logger.error(f"Error in NRCLex emotion analysis: {e}")
        return {}


def apply_VADER(doc):
    """
    Apply VADER sentiment analysis to a spaCy Doc.

    This function evaluates the sentiment of each sentence in a spaCy document 
    using the VADER sentiment analysis tool. It computes sentiment scores for 
    positive, negative, neutral, and compound sentiment values, and returns 
    statistical measures such as max, min, average, median, variance, 
    standard deviation, and coefficient of variation.

    Args:
        doc (spacy.tokens.Doc): A spaCy document object containing tokenized sentences.

    Returns:
        dict: Dictionary with VADER sentiment scores.
    """
    try:
        subresults = {}
        labels = ['pos', 'compound', 'neu', 'neg']
        sample_scores = {label: [] for label in labels}

        analyzer = SentimentIntensityAnalyzer()
        for sentence in doc.sents:
            vs = analyzer.polarity_scores(sentence.text)
            for label in labels:
                sample_scores[label].append(vs[label])

        for label in labels:
            sentiment_values = sample_scores[label] if sample_scores[label] else [0]

            subresults.update({
                f"VADER_max_{label}": max(sentiment_values),
                f"VADER_min_{label}": min(sentiment_values),
                f"VADER_avg_{label}": np.nanmean(sentiment_values),
                f"VADER_median_{label}": np.median(sentiment_values),
                f"VADER_var_{label}": np.nanvar(sentiment_values),
                f"VADER_std_dev_{label}": np.std(sentiment_values),
                f"VADER_cv_{label}": np.std(sentiment_values) / np.mean(sentiment_values) if np.mean(sentiment_values) != 0 else 0
            })

        # logger.info(f"VADER analysis completed successfully.")
        return subresults

    except Exception as e:
        logger.error(f"Error in VADER sentiment analysis: {e}")
        return {}


def apply_TextBlob(doc):
    """
    Apply TextBlob sentiment analysis to a spaCy Doc.

    This function evaluates sentiment polarity and subjectivity for each sentence in a 
    given spaCy document using the TextBlob library. It computes statistical metrics 
    such as max, min, average, median, variance, standard deviation, and coefficient 
    of variation.

    Args:
        doc (spacy.tokens.Doc): A spaCy document object containing tokenized sentences.

    Returns:
        dict: Dictionary with TextBlob sentiment scores.
    """
    try:
        subresults = {}
        labels = ['pol', 'subj']
        sample_scores = {label: [] for label in labels}

        for sentence in doc.sents:
            blob = TextBlob(sentence.text)
            sample_scores['pol'].append(blob.sentiment.polarity)
            sample_scores['subj'].append(blob.sentiment.subjectivity)

        for label in labels:
            sentiment_values = sample_scores[label] if sample_scores[label] else [0]

            subresults.update({
                f"TextBlob_max_{label}": max(sentiment_values),
                f"TextBlob_min_{label}": min(sentiment_values),
                f"TextBlob_avg_{label}": np.nanmean(sentiment_values),
                f"TextBlob_median_{label}": np.median(sentiment_values),
                f"TextBlob_var_{label}": np.nanvar(sentiment_values),
                f"TextBlob_std_dev_{label}": np.std(sentiment_values),
                f"TextBlob_cv_{label}": np.std(sentiment_values) / np.mean(sentiment_values) if np.mean(sentiment_values) != 0 else 0
            })

        # logger.info(f"TextBlob analysis completed successfully.")
        return subresults

    except Exception as e:
        logger.error(f"Error in TextBlob sentiment analysis: {e}")
        return {}


def apply_Afinn(doc):
    """
    Apply Afinn sentiment analysis to a spaCy Doc.

    This function assigns sentiment scores using the Afinn lexicon, categorizing 
    sentences as positive, negative, or overall sentiment. It computes statistical 
    metrics such as max, min, average, median, variance, standard deviation, and 
    coefficient of variation.

    Args:
        doc (spacy.tokens.Doc): A spaCy document object containing tokenized sentences.

    Returns:
        dict: Dictionary with Afinn sentiment scores.
    """
    try:
        subresults = {}
        afinn = Afinn()
        labels = ['pos', 'neg', 'overall']
        sample_scores = {label: [] for label in labels}

        for sentence in doc.sents:
            score = afinn.score(sentence.text)
            label = None
            if score > 0:
                label = 'pos'
            elif score < 0:
                label = 'neg'
            
            if label:
                sample_scores[label].append(score)
            sample_scores['overall'].append(score)

        for label in labels:
            sentiment_values = sample_scores[label] if sample_scores[label] else [0]

            subresults.update({
                f"Afinn_max_{label}_score": max(sentiment_values),
                f"Afinn_min_{label}_score": min(sentiment_values),
                f"Afinn_avg_{label}_score": np.nanmean(sentiment_values),
                f"Afinn_median_{label}_score": np.median(sentiment_values),
                f"Afinn_var_{label}_score": np.nanvar(sentiment_values),
                f"Afinn_std_dev_{label}_score": np.std(sentiment_values),
                f"Afinn_cv_{label}_score": np.std(sentiment_values) / np.mean(sentiment_values) if np.mean(sentiment_values) != 0 else 0
            })

        # logger.info(f"Afinn analysis completed successfully.")
        return subresults

    except Exception as e:
        logger.error(f"Error in Afinn sentiment analysis: {e}")
        return {}
