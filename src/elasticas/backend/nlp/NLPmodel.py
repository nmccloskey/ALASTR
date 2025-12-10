import spacy
import benepar
from nltk.corpus import cmudict
import logging
logger = logging.getLogger("CustomLogger")
import subprocess
import sys


class NLPmodel:
    """
    NLP Model Singleton that loads and manages multiple SpaCy models,
    Benepar for constituency parsing, and the CMU Pronouncing Dictionary.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._nlp_models = {}
            cls._instance._tokenizer = None
            cls._instance._cmu_dict = None
            cls._instance.load_nlp()
        return cls._instance

    def _ensure_benepar(self):
        try:
            benepar.download('benepar_en3')
        except Exception as e:
            logger.warning(f"Benepar model not found: {e}")

    def _ensure_spacy_model(self, model_name: str):
        try:
            spacy.load(model_name)
        except OSError:
            logger.warning(f"{model_name} not found. Attempting to install...")
            subprocess.run([sys.executable, "-m", "spacy", "download", model_name], check=True)

    def load_nlp(self, model_name="en_core_web_trf"):
        """
        Lazy-loads a specified SpaCy model with optional Benepar for constituency parsing.

        Args:
            model_name (str): Name of the SpaCy model to load. Defaults to "en_core_web_trf".

        Returns:
            spacy.Language: Loaded SpaCy language model.
        """
        if model_name not in self._nlp_models:
            self._ensure_spacy_model(model_name)
            self._nlp_models[model_name] = spacy.load(model_name)

            # Add Benepar (works with both Transformer and non-Transformer models)
            if "benepar" not in self._nlp_models[model_name].pipe_names:
                try:
                    self._nlp_models[model_name].add_pipe("benepar", config={"model": "benepar_en3"})
                except Exception as e:
                    logger.warning(f"Failed to add Benepar to {model_name}: {e}")

        logger.info(f"Successfully loaded {model_name} into NLPModel.")

        
    def get_nlp(self, model_name="en_core_web_trf"):
        if model_name not in self._nlp_models:
            self.load_nlp(model_name)
        return self._nlp_models[model_name]


    def get_tokenizer(self, model_name="en_core_web_trf"):
        """
        Retrieves the tokenizer from the specified SpaCy model.

        Args:
            model_name (str): Name of the SpaCy model to use. Defaults to "en_core_web_trf".

        Returns:
            spacy.tokenizer.Tokenizer: The tokenizer instance.
        """
        if model_name not in self._nlp_models:
            self.get_nlp(model_name)  # Ensure the model is loaded before accessing tokenizer

        return self._nlp_models[model_name].tokenizer


    def get_cmu_dict(self):
        """
        Lazy-loads the CMU Pronouncing Dictionary.

        Returns:
            dict: CMU Pronouncing Dictionary.
        """
        if self._cmu_dict is None:
            self._cmu_dict = cmudict.dict()
        return self._cmu_dict
