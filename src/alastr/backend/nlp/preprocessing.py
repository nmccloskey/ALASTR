from __future__ import annotations

from tqdm import tqdm
from pathlib import Path
from alastr.backend.nlp.NLPmodel import NLPmodel
from alastr.backend.etl.OutputManager import IOManager
from alastr.backend.nlp.data_processing import clean_text, get_two_cha_versions
from alastr.backend.tools.logger import logger
from alastr.backend.nlp.sample_prep import prep_samples
from typing import Any, Dict, List, Sequence, Set, Tuple


def process_sents(doc, sample_data, is_cha=False):
    doc_id = sample_data["doc_id"]
    base_sent_data = {k:v for k,v in sample_data.items() if k not in ["doc_id", "text"]}

    cleaned_doc, semantic_doc, cleaned_phon_doc = "", "", ""
    sent_data_results, sent_text_results = [], []

    for i, sent in enumerate(doc.sents):
        sent_id = i + 1

        if is_cha:
            cleaned, cleaned_phon = get_two_cha_versions(sent.text)
        else:
            cleaned = clean_text(sent.text)
        
        semantic = [token.lemma_ for token in sent if token.is_alpha and not token.is_stop]

        sent_data = {"doc_id": doc_id, "sent_id": sent_id}
        sent_data.update(base_sent_data.copy())
        sent_data_results.append(sent_data)

        sent_text = {
            "doc_id": doc_id,
            "sent_id": sent_id,
            "raw": sent.text,
            "cleaned": cleaned,
            "semantic": " ".join(semantic),
        }

        if is_cha:
            sent_text.update({"cleaned_phon": cleaned_phon})

        sent_text_results.append(sent_text)

        cleaned_doc += " " + cleaned
        semantic_doc += " " + " ".join(semantic)
        
        if is_cha:
            cleaned_phon_doc += " " + cleaned_phon

    return sent_data_results, sent_text_results, cleaned_doc.strip(), semantic_doc.strip(), cleaned_phon_doc.strip()


def process_sample_data(PM, sample_data):
    """
    Processes text docs to store three versions for later analysis:
    - Raw text
    - Cleaned text
    - semantic (lemmatized) text
    - Sentence segmentation (if applicable)

    Args:
        sample_data (dict): A dictionary containing document information, including 'text'.

    Returns:
        dict: A dictionary containing:
              - 'doc': Document-level text versions.
              - 'sent': List of sentence-level dictionaries (if sentence-level processing is enabled).
    """
    try:
        if not isinstance(sample_data['text'], str):
            raise ValueError(f"Expected 'text' to be a string, but got {type(sample_data['text'])}")

        doc_id = sample_data["doc_id"]
        is_cha = sample_data["doc_label"].endswith(".cha")

        results = PM.sections["preprocessing"].init_results_dict()
       
        NLP = NLPmodel()
        nlp = NLP.get_nlp()
        doc = nlp(sample_data['text'])
        
        if PM.sentence_level:
            sent_data_results, sent_text_results, cleaned_doc, semantic_doc, cleaned_phon_doc = process_sents(doc, sample_data, is_cha)
            results["sample_data_sent"].extend(sent_data_results)
            results["sample_text_sent"].extend(sent_text_results)

        else:
            if is_cha:
                cleaned_doc, cleaned_phon_doc = get_two_cha_versions(doc.text)
            else:
                cleaned_doc = clean_text(doc.text)
            
            semantic_doc = " ".join([token.lemma_ for token in doc if token.is_alpha and not token.is_stop])

        results["sample_data_doc"].update({k:v for k,v in sample_data.items() if k not in ["text"]})
        results["sample_text_doc"].update({
            "doc_id": doc_id,
            "raw": doc.text,
            "cleaned": cleaned_doc,
            "semantic": semantic_doc
        })

        if is_cha:
            results["sample_text_doc"].update({"cleaned_phon": cleaned_phon_doc})

        # Logging success
        doc_label = sample_data.get("doc_label", "Unknown")
        logger.info(f"Preprocessed for doc {doc_id}: {doc_label}.")
        
        return results

    except Exception as e:
        doc_id = sample_data.get("doc_id", "Unknown")
        doc_label = sample_data.get("doc_label", "Unknown")
        logger.error(f"Error preprocessing for {doc_id}: {doc_label}: {e}")
        return {}


def _process_one_input_file(
    file_path: Path,
    next_doc_id: int,
    exclude_speakers: Sequence[str],
    OM: Any,
    PM: Any,
    progress_bar: Any,
) -> Tuple[List[int], int]:
    """
    Process one input file into one or more sample records, update OM tables,
    and return (doc_ids_added, updated_next_doc_id).

    Raises exceptions upward so the caller decides whether to skip or stop.
    """
    file_name = file_path.name
    logger.info("Processing file: name=%s path=%s", file_name, file_path)

    samples = prep_samples(
        file_path=file_path,
        doc_id=next_doc_id,
        exclude_speakers=exclude_speakers,
        OM=OM,
    )

    if not samples:
        logger.warning("No samples produced for file %s (skipping).", file_name)
        return [], next_doc_id

    doc_ids_added: List[int] = []

    for sample_data in samples:
        # Defensive validation (keeps failures informative)
        required_keys = {"doc_id", "doc_label", "text"}
        missing = required_keys - set(sample_data.keys())
        if missing:
            raise ValueError(
                f"Sample record missing required keys {sorted(missing)} "
                f"from file {file_name}. Got keys={list(sample_data.keys())}"
            )

        msg = f"Processing sample {sample_data['doc_id']}: {sample_data['doc_label']}"
        progress_bar.set_description(msg)
        logger.info(msg)

        results: Dict[str, Any] = process_sample_data(PM, sample_data)
        if not isinstance(results, dict):
            raise TypeError(
                f"process_sample_data must return dict[table_name, data]. Got {type(results)}"
            )

        for table_name, data in results.items():
            if table_name not in OM.tables:
                raise KeyError(
                    f"process_sample_data returned unknown table {table_name!r}. "
                    f"Known tables: {list(OM.tables.keys())}"
                )
            OM.tables[table_name].update_data(data)

        doc_ids_added.append(int(sample_data["doc_id"]))
        next_doc_id += 1  # increment per sample

    return doc_ids_added, next_doc_id


def preprocess_text(PM) -> list[int]:
    """
    Orchestrate reading/processing/storing docs from OM.input_dir.

    Returns a list of doc_ids successfully stored.
    """
    OM = IOManager()

    input_dir = Path(OM.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist or is not a directory: {input_dir}")

    exclude_speakers = OM.config.get("exclude_speakers", ["INV"])

    PM.sections["preprocessing"].create_raw_data_tables()

    allowed_extensions: Set[str] = {".cha", ".txt", ".docx", ".csv", ".xlsx"}
    file_paths = sorted(
        [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in allowed_extensions],
        key=lambda p: (p.suffix.lower(), str(p).lower()),
    )

    logger.info("Found %d files in %s. Processing started...", len(file_paths), input_dir)

    doc_ids: List[int] = []
    next_doc_id = 1

    progress_bar = tqdm(file_paths, desc="Reading Files", dynamic_ncols=True)

    for file_path in progress_bar:
        progress_bar.set_description(f"Processing {file_path.name}")

        try:
            added, next_doc_id = _process_one_input_file(
                file_path=file_path,
                next_doc_id=next_doc_id,
                exclude_speakers=exclude_speakers,
                OM=OM,
                PM=PM,
                progress_bar=progress_bar
            )
            doc_ids.extend(added)

        except Exception:
            logger.exception("Error processing file %s. Skipping.", file_path)
            continue

    OM.num_docs = len(doc_ids)
    logger.info("Processing completed. %d docs stored.", OM.num_docs)

    for table_name, table in OM.tables.items():
        try:
            table.export_to_excel()
        except Exception:
            logger.exception("Failed exporting table %s to Excel.", table_name)

    return doc_ids
