import warnings
import numpy as np
import logging
logger = logging.getLogger("CustomLogger")
from elasticas.backend.nlp.NLPmodel import NLPmodel
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from elasticas.backend.nlp.data_processing import matrix_metrics
from elasticas.analyses.semantic_scoring import apply_Afinn, apply_VADER, apply_NRCLex, apply_TextBlob

warnings.filterwarnings("ignore", message=".*TreeCRF.*does not define `arg_constraints`.*")

def apply_sklearn_TruncSVD(doc, num_topics=5):
    """
    Apply Truncated SVD (Latent Semantic Analysis) to a spaCy Doc.

    This function extracts latent topics from a document using TF-IDF and 
    Truncated SVD. It returns the top words associated with each topic and 
    measures topic strength, variability (CV), and drop-off in sentence importance.

    Args:
        doc (spacy.tokens.Doc): A spaCy document object containing semantic sentences.
        num_topics (int, optional): The number of topics to extract. Defaults to 5.

    Returns:
        dict: Dictionary with extracted topics, strength, variability, and importance drop-off measures.
    """
    try:
        subresults = {}

        # Convert spaCy Doc to list of long-enough sentences
        sentences = [sent.text for sent in doc.sents if len(sent.text.split(" ")) >= 3]

        if len(sentences) < 1:
            logger.warning("Not enough valid sentences for topic modeling.")
            return {}

        # Vectorize text using TF-IDF
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(sentences)

        # # Ensure at least 2 valid features exist for SVD
        # if X.shape[1] < 2:
        #     logger.warning("Not enough unique words after vectorization - skipping doc.")
        #     return {}

        # # Apply Truncated SVD for topic extraction
        # svd = TruncatedSVD(n_components=num_topics)

        num_features = X.shape[1]
        safe_num_topics = min(num_topics, num_features - 1)  # At least 1 topic less than features
        if safe_num_topics < 1:
            logger.warning(f"Too few features ({num_features}) for topic modeling. Skipping.")
            return {}

        svd = TruncatedSVD(n_components=safe_num_topics)

        topic_matrix = svd.fit_transform(X)  # Sentence-topic importance scores
        topic_components = svd.components_  # Topic-word importance scores

        words = vectorizer.get_feature_names_out()

        for idx, topic in enumerate(topic_components):
            # Get top words for the topic
            top_word_indices = topic.argsort()[:-6:-1]  # Top 5 words
            subresults[f"tSVD_Topic{idx+1}"] = ", ".join([words[i] for i in top_word_indices])

            # Topic strength (sum of importance scores)
            topic_strength = np.sum(topic)
            subresults[f"tSVD_Topic{idx+1}_Strength"] = topic_strength

            # Topic variability (CV)
            topic_std = np.std(topic)
            topic_mean = np.mean(topic)
            subresults[f"tSVD_Topic{idx+1}_CV"] = topic_std / topic_mean if topic_mean != 0 else 0

            # Sentence importance for this topic
            sentence_importance = topic_matrix[:, idx]  # Importance of each sentence

            if len(sentence_importance) > 0:
                # Sort in descending order to analyze drop-off
                sorted_importance = np.sort(sentence_importance)[::-1]

                # Top sentence importance fraction
                top_sentence_frac = sorted_importance[0] / np.sum(sorted_importance) if np.sum(sorted_importance) > 0 else 0
                subresults[f"tSVD_Topic{idx+1}_TopSentenceFrac"] = top_sentence_frac

                # Ratio of first to second sentence importance
                if len(sorted_importance) > 1:
                    top_vs_second_ratio = sorted_importance[0] / sorted_importance[1] if sorted_importance[1] > 0 else 0
                else:
                    top_vs_second_ratio = 0
                subresults[f"tSVD_Topic{idx+1}_TopVsSecondRatio"] = top_vs_second_ratio

        logger.info(f"Truncated SVD completed successfully: Extracted {num_topics} topics.")
        return subresults

    except Exception as e:
        logger.error(f"Error in Sklearn TruncatedSVD: {e}")
        return {}

def compute_token_embeddings(doc):
    """
    Computes token embeddings using spaCy's `en_core_web_lg` model.
    
    Unlike transformer-based embeddings (`en_core_web_trf`), `en_core_web_lg` provides 
    static word embeddings trained using word co-occurrences.

    If a token has no embedding available, zero-padding is used to maintain uniform shape.

    Args:
        doc (spacy.tokens.Doc): A spaCy document object.

    Returns:
        np.array: A 2D array where each row is an embedding for a token.
    """
    try:
        logger.info("Computing token embeddings using `en_core_web_lg`.")

        # Load en_core_web_lg for token embeddings
        NLP = NLPmodel()
        nlp_lg = NLP.get_nlp("en_core_web_lg")
        # nlp_lg = NLPmodel.get_nlp("en_core_web_lg")
        doc = nlp_lg(doc.text)

        tokens = [token for token in doc if not token.is_punct]

        if len(tokens) < 2:
            logger.warning("Not enough valid tokens found for token embeddings computation.")
            return np.array([])

        # Get the embedding size from en_core_web_lg
        embedding_dim = nlp_lg.vocab.vectors_length  # Typically 300 for lg models

        embeddings = []
        for token in tokens:
            if token.has_vector:
                token_embedding = token.vector.astype(np.float32)
                token_embedding = normalize(token_embedding.reshape(1, -1))[0]  # Normalize
                
                # Ensure all embeddings have the same shape
                if token_embedding.shape[0] < embedding_dim:
                    pad_size = embedding_dim - token_embedding.shape[0]
                    token_embedding = np.pad(token_embedding, (0, pad_size), mode='constant')
                elif token_embedding.shape[0] > embedding_dim:
                    token_embedding = token_embedding[:embedding_dim]

            else:
                token_embedding = np.zeros(embedding_dim, dtype=np.float32)  # Fallback to zeros
                logger.warning(f"Token '{token.text}' has no valid `en_core_web_lg` embedding. Using zeros.")

            embeddings.append(token_embedding)

        # Convert list to NumPy array
        embeddings = np.array(embeddings, dtype=np.float32)

        return embeddings

    except Exception as e:
        logger.error(f"Error computing token embeddings: {e}")
        return np.array([])

def compute_sentence_embeddings(doc):
    """
    Computes sentence embeddings using transformer-based embeddings (`en_core_web_trf`).
    Handles mismatches between transformer tokenization and spaCy sentence segmentation.

    Args:
        doc (spacy.tokens.Doc): A spaCy document object.

    Returns:
        np.array: A 2D array where each row is an embedding for a sentence.
    """
    try:
        logger.info("Computing transformer-based sentence embeddings.")
        sentences = list(doc.sents)

        if len(sentences) < 1:
            logger.warning("No sentences found in document.")
            return np.array([])

        # Extract transformer embeddings (full document)
        trf_tensor = doc._.trf_data.last_hidden_layer_state.data
        trf_tensor = np.array(trf_tensor, dtype=np.float32)  # Ensure correct dtype

        if trf_tensor.shape[0] < len(doc):
            logger.warning(f"Transformer produced fewer embeddings ({trf_tensor.shape[0]}) than tokens ({len(doc)}).")

        # Compute sentence embeddings by averaging token vectors within each sentence
        embeddings = []
        for sent in sentences:
            if sent.start < len(trf_tensor):  # Ensure the index is within bounds
                sentence_embedding = trf_tensor[sent.start:sent.end].mean(axis=0)
                embeddings.append(sentence_embedding)
            else:
                logger.warning(f"Skipping sentence at index {sent.start} due to out-of-bounds embedding reference.")
        
        if len(embeddings) == 0:
            logger.warning("No valid sentence embeddings were computed.")
            return np.array([])

        return np.array(embeddings, dtype=np.float32)

    except Exception as e:
        logger.error(f"Error computing sentence embeddings: {e}")
        return np.array([])

def compute_similarity_matrix(embeddings):
    """
    Computes a cosine similarity matrix from embeddings.

    Args:
        embeddings (np.array): Embedding matrix (sentence or token level).

    Returns:
        np.array: Pairwise cosine similarity matrix.
    """
    try:
        if embeddings.shape[0] < 2:
            logger.warning("Fewer than two embeddings. Returning trivial similarity matrix.")
            return np.array([[1.0]])

        logger.info("Checking embedding dimensions before computing similarity.")
        
        # Ensure embeddings have at least one feature
        if embeddings.shape[1] == 0:
            raise ValueError("Embeddings have no features (empty vectors). Ensure the model provides valid vector outputs.")

        # Normalize embeddings before computing cosine similarity
        embeddings = normalize(embeddings, axis=1)  # Normalize to unit vectors

        logger.info("Computing cosine similarity matrix.")
        return cosine_similarity(embeddings)

    except Exception as e:
        logger.error(f"Error computing similarity matrix: {e}")
        return np.array([])

def compute_cohesion_decay(sim_matrix, label):
    """
    Measures cohesion decay by analyzing how similarity declines across a document.

    Can be used for both sentence-level and token-level similarity matrices.

    Args:
        sim_matrix (np.array): Pairwise similarity matrix (sentence-sentence or token-token).
        label (str): Label for result keys.

    Returns:
        dict: Cohesion decay measures including min, max, avg, std_dev, and CV.
    """
    try:
        num_units = sim_matrix.shape[0]  # Can be sentences or tokens

        if num_units < 3:
            return {}

        # Extract consecutive similarities (i.e., similarity between unit i and i+1)
        consecutive_similarities = np.array(
            [sim_matrix[i, i + 1] for i in range(num_units - 1)], dtype=np.float32
        )

        if len(consecutive_similarities) < 2:
            return {}

        # Compute statistical measures (ensure they return standard Python floats)
        min_sim = float(np.min(consecutive_similarities))
        max_sim = float(np.max(consecutive_similarities))
        mean_sim = float(np.mean(consecutive_similarities))
        std_dev_sim = float(np.std(consecutive_similarities))
        cv_sim = float(std_dev_sim / mean_sim) if mean_sim != 0 else None  # Avoid division by zero

        return {
            f"{label}_cohesion_decay_min": min_sim,
            f"{label}_cohesion_decay_max": max_sim,
            f"{label}_cohesion_decay_mean": mean_sim,
            f"{label}_cohesion_decay_std_dev": std_dev_sim,
            f"{label}_cohesion_decay_cv": cv_sim
        }

    except Exception as e:
        logger.error(f"Error computing cohesion decay for {label}: {e}")
        return {}

def sentence_level_similarity(doc):
    """
    Computes token-level semantic similarity within a single sentence.

    This function generates a token-token similarity matrix and extracts statistical 
    metrics on token cohesion.

    Args:
        doc (spacy.tokens.Doc): A spaCy document object representing a single sentence - lemmatized without stop words or punctuation.

    Returns:
        dict: Dictionary containing intrasentential similarity metrics and cohesion decay.
    """
    results = {}

    try:
        logger.info("Computing sentence-level token similarity.")
        embeddings = compute_token_embeddings(doc)
        sim_matrix = compute_similarity_matrix(embeddings)

        logger.info("Extracting token-level similarity metrics.")
        results.update(matrix_metrics(sim_matrix, [t for t in doc if not t.is_punct], "intrasentential_similarity"))

        logger.info("Computing cohesion decay for intrasentential similarity.")
        results.update(compute_cohesion_decay(sim_matrix, "cosine"))

    except Exception as e:
        logger.error(f"Error in sentence_level_similarity: {e}")

    return results

def document_level_similarity(doc):
    """
    Computes sentence-level semantic similarity within a document.

    This function generates a sentence-sentence similarity matrix using transformer embeddings.

    Statistical metrics and cohesion decay are computed.

    Args:
        doc (spacy.tokens.Doc): A spaCy document object - lightly preprocessed ("cleaned" version).

    Returns:
        dict: Dictionary containing document-level similarity metrics and cohesion decay.
    """
    results = {}

    try:
        logger.info("Computing document-level sentence similarity.")
        embeddings = compute_sentence_embeddings(doc)
        sim_matrix = compute_similarity_matrix(embeddings)

        logger.info("Extracting sentence-level similarity metrics.")
        results.update(matrix_metrics(sim_matrix, list(doc.sents), "cosine_semantic_similarity"))

        logger.info("Computing cohesion decay for sentence-level similarity.")
        results.update(compute_cohesion_decay(sim_matrix, "cosine"))

    except Exception as e:
        logger.error(f"Error in document_level_similarity: {e}")

    return results

def analyze_semantics(PM, sample_data):
    """
    Perform semantic analysis on a preprocessed text sample.

    Args:
        sample_row (dict): Dictionary containing sample data, including 'sentence_preprocessed'.

    Returns:
        dict: Semantic analysis results.
    """
    try:
        results = PM.sections["semantics"].init_results_dict()
        
        NLP = NLPmodel()
        nlp = NLP.get_nlp()

        if PM.sentence_level:
            if not isinstance(sample_data, list):
                raise ValueError("Expected a list of sentence dicts for sentence-level analysis.")
            
            doc_cleaned = ""
            doc_semantic = ""
            doc_id = sample_data[0].get("doc_id")
            
            for sent in sample_data:
                func_data = {}
                sent_id = sent.get("sent_id")
                if sent.get("cleaned_phon", ""):
                    cleaned = sent.get("cleaned_phon", "")
                else:
                    cleaned = sent.get("cleaned", "")
                semantic = sent.get("semantic", "")
                sent_data_base = {"doc_id": doc_id, "sent_id": sent_id}
                
                doc = nlp(cleaned)
                func_data["unit_sim"] = sentence_level_similarity(doc)
                func_data["NRCLex"] = apply_NRCLex(doc)
                func_data["VADER"] = apply_VADER(doc)
                func_data["TextBlob"] = apply_TextBlob(doc)
                func_data["Afinn"] = apply_Afinn(doc)

                doc = nlp(semantic)
                func_data["topics"] = apply_sklearn_TruncSVD(doc, 3)

                for table, row_data in func_data.items():
                    sent_data = sent_data_base.copy()
                    sent_data.update(row_data)
                    results[f"{table}_sent"].append(sent_data)
                
                doc_cleaned += " " + sent["cleaned"]
                doc_semantic += " " + sent["semantic"] + "."
            
            doc_cleaned = doc_cleaned.strip()
            doc_semantic = doc_semantic.strip()
        
        else:
            if not isinstance(sample_data, dict):
                raise ValueError("Expected a single dict for document-level analysis.")
            
            doc_id = sample_data.get("doc_id")
            doc_semantic = sample_data.get("semantic", "")
            
            if sample_data.get("cleaned_phon", ""):
                doc_cleaned = sample_data.get("cleaned_phon", "")
            else:
                doc_cleaned = sample_data.get("cleaned", "")
            
        func_data = {}
        doc_data_base = {"doc_id": doc_id}
            
        doc = nlp(doc_cleaned)
        func_data["unit_sim"] = sentence_level_similarity(doc)
        func_data["NRCLex"] = apply_NRCLex(doc)
        func_data["VADER"] = apply_VADER(doc)
        func_data["TextBlob"] = apply_TextBlob(doc)
        func_data["Afinn"] = apply_Afinn(doc)

        doc = nlp(doc_semantic)
        func_data["topics"] = apply_sklearn_TruncSVD(doc, 7)

        for table, row_data in func_data.items():
            doc_data = doc_data_base.copy()
            doc_data.update(row_data)
            results[f"{table}_doc"].update(doc_data)

        logger.info(f"Semantic analysis completed.")
        return results

    except Exception as e:
        logger.error(f"Error analyzing semantics: {e}")
        return {}
