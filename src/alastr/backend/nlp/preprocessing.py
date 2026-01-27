import os
import pandas as pd
import docx2txt as dx
from tqdm import tqdm
from pathlib import Path
from alastr.backend.nlp.NLPmodel import NLPmodel
from alastr.backend.etl.OutputManager import OutputManager
from alastr.backend.nlp.data_processing import scrub_raw_text, clean_text, get_text_from_cha, get_two_cha_versions
from alastr.backend.tools.logger import logger, _rel

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
        is_cha = sample_data["doc_label"].endswith(".cha") ###U or transcript table derived

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

def read_chat_file(file_path: str) -> dict:
    OM = OutputManager()
    exclude_speakers = OM.config.get("exclude_speakers", ["INV"])
    text_content = get_text_from_cha(file_path, exclude_speakers)
    logger.info(f"Processed CHAT file: {file_path}")
    return text_content

def read_text_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="replace") as file:
        text_content = file.read()
        text_content = scrub_raw_text(text_content)
        logger.info(f"Processed TXT file: {file_path}")
        return text_content #.replace('\n',' ')

def read_docx_file(file_path: str) -> str:
    text_content = dx.process(file_path)
    text_content = scrub_raw_text(text_content)
    logger.info(f"Processed DOCX file: {file_path}")
    return text_content #.replace('\n',' ')

def read_spreadsheet(file_path, file_name, doc_id, OM):
    """
    Reads a spreadsheet (.xlsx or .csv), processes it, and returns a list of document records.

    Args:
        file_path (str): Path to the spreadsheet file.
        file_name (str): Name of the file (used in `doc_label`).
        doc_id (int): Starting document ID.
        OM (OutputManager): OutputManager instance handling tier creation.

    Returns:
        list: List of dictionaries, each representing a processed row.
    """
    df = pd.read_excel(file_path) if file_name.endswith(".xlsx") else pd.read_csv(file_path)

    if df.empty:
        raise ValueError(f"{file_name} contains no valid text entries.")
                
    if "text" not in df.columns:
        raise ValueError(f"{file_name} must contain 'text' column.")
    
    other_columns = [col for col in df.columns if col != "text"]
    new_tiers = [OM.tm.make_tier(col) for col in other_columns]
    OM.tm.tiers.update({tier.name: tier for tier in new_tiers if tier})
    logger.info(f"TierManager's tiers: {[(t.name) for t in OM.tm.tiers.values()]}")

    df.insert(0, "doc_label", file_name + "|" + df[other_columns].astype(str).agg("|".join, axis=1) + "|" + df.index.astype(str))
    df.insert(0, "doc_id", range(doc_id, doc_id + len(df)))
    df = df.dropna(subset=["text"])

    samples = df.to_dict(orient="records")
    logger.info(f"Processed {len(samples)} rows from file: {file_name}")

    return samples

def prep_samples(file_name, file_path, doc_id, OM):

    if not file_name.endswith((".xlsx", ".cha", ".txt", ".docx", ".csv")):
        logger.warning(f"Unsupported file format: {file_name}. Skipping.")
        return []
    
    if file_name.endswith(".xlsx") or file_name.endswith(".csv"):
        samples = read_spreadsheet(file_path, file_name, doc_id, OM)
    
    else:
        if file_name.endswith(".cha"):
            text_content = read_chat_file(file_path)

        elif file_name.endswith(".txt"):
            text_content = read_text_file(file_path)

        elif file_name.endswith(".docx"):
            text_content = read_docx_file(file_path)

        sample_data = {
                "doc_id": doc_id,
                "doc_label": file_name,
                "text": text_content,
                **OM.tm.match_tiers(file_name)
            }
        samples = [sample_data]
    
    return samples

def preprocess_text(PM) -> list:
    """
    Reads, processes, and stores text docs from various file formats.

    This function extracts text from multiple file formats, processes each doc 
    incrementally, assigns doc IDs, and updates the database.

    Supported Formats:
    - `.cha` (CHAT files) → Uses `pylangacq` for transcription.
    - `.txt` and `.docx` → Reads raw text.
    - `.csv` and `.xlsx` → Must contain 'label' and 'text' columns.

    Returns:
        list: A list of assigned doc IDs.
    
    Raises:
        FileNotFoundError: If `input_dir` does not exist.
        ValueError: If file format is unsupported.
    """
    OM = OutputManager()

    if not os.path.isdir(OM.input_dir):
        raise FileNotFoundError(f"Input directory '{OM.input_dir}' does not exist.")
    
    PM.sections["preprocessing"].create_raw_data_tables()

    doc_id = 1
    doc_ids = []
    allowed_extensions = {".cha", ".txt", ".docx", ".csv", ".xlsx"}

    file_paths = [f for f in Path(OM.input_dir).rglob("*") if f.suffix.lower() in allowed_extensions and f.is_file()]
    file_names = [str(os.path.basename(f)) for f in file_paths]

    logger.info(f"Found {len(file_paths)} files in '{OM.input_dir}'. Processing started...")
    progress_bar = tqdm(zip(file_names, file_paths), desc="Reading Files", dynamic_ncols=True)

    for file_name, file_path in progress_bar:
        progress_bar.set_description(f"Processing {file_name}")
        logger.info(f"name: {file_name}, path: {file_path}")

        try:
            samples = prep_samples(file_name, file_path, doc_id, OM)
                
            for sample_data in samples:
                process_str = f"Processing sample {sample_data['doc_id']}: {sample_data['doc_label']}"
                progress_bar.set_description(process_str)
                logger.info(process_str)

                results = process_sample_data(PM, sample_data)

                for table_name, data in results.items():
                    OM.tables[table_name].update_data(data)

                doc_ids.append(sample_data["doc_id"])
                doc_id += 1

        except Exception as e:
            logger.error(f"Error processing file '{file_name}': {e}")

    num_docs = len(doc_ids)
    OM.num_docs = num_docs
    logger.info(f"Processing completed. {num_docs} docs stored.")

    for table_name in results:
        OM.tables[table_name].export_to_excel()
    
    return doc_ids
