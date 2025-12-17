# ALASTR – Aphasiological Language Analysis of Speech Transcripts for Research

**Status:** Active development (early-stage).  
**Stability:** APIs, module layout, and CLI interfaces are subject to change.  
**Audience:** Researchers and clinicians working with clinical aphasiology and SLP discourse data.

ALASTR is a Python toolkit for **scalable, scriptable analysis of clinical speech and language transcripts**, with an emphasis on aphasia-focused workflows. It is designed to complement existing CHAT/CLAN-based pipelines by adding reproducible batch processing, richer linguistic feature extraction, and integration with downstream statistical analyses.

While ALASTR draws on concepts and components piloted in earlier prototypes (e.g., CLATR), it is being developed as the **lab-facing, aphasiology-specialized system**, with a clearer focus on clinical narratives, paraphasias, disfluencies, and other discourse-level phenomena relevant to treatment and outcomes research.

---

## Core Aims

- **Scalability:** Process many transcripts in batch (across participants, timepoints, or conditions) with consistent configuration and logging.
- **Clinical relevance:** Target metrics and summaries that are meaningful for aphasiology and speech–language pathology.
- **Interoperability with CHAT/CLAN:** Work *with* existing CHAT-formatted (.cha) transcripts rather than replacing them, enabling semi-automated workflows.
- **Integration with other tools:** Provide hooks for metrics and outputs from systems such as RASCAL (monologic discourse analysis) and DIAAD (dialogue analysis).

---

## High-Level Functionality (Planned / Emerging)

- **Transcript ingestion and organization**
  - Read, validate, and organize transcripts (e.g., by group, site, timepoint).
  - Support CHAT-formatted transcripts, with planned adapters for other formats.

- **Linguistic feature extraction**
  - Token-level and utterance-level features using spaCy and related NLP libraries.
  - Tier-aware processing (e.g., mapping CHAT tiers into structured tables).
  - Preliminary support for paraphasia and disfluency-related annotations.

- **Batch summarization and export**
  - Participant-level and group-level summary tables (e.g., lexical, syntactic, discourse measures).
  - Integration points for CoreLex counts (via RASCAL) and other domain metrics.
  - Consistent output schemas suitable for downstream statistics in R, Python, or other tools.

---

## Installation (Early Preview)

ALASTR is not yet distributed on PyPI. During early development, you can install it from a local clone:

```bash
git clone https://github.com/nmccloskey/ALASTR.git
cd ALASTR
pip install -e .
```

You may wish to create and activate a dedicated virtual environment or conda environment before installing.

---

## Usage (Very Early Sketch)

CLI and API interfaces are still evolving. A minimal example of the intended usage pattern might eventually look like:

```bash
alastr run \
  --config path/to/config.yaml \
  --input-transcripts path/to/cha/files \
  --output-dir path/to/output
```

or, in Python:

```python
from alastr.pipeline import run_pipeline

run_pipeline(
    config_path="path/to/config.yaml",
    input_root="path/to/cha/files",
    output_root="path/to/output",
)
```

Exact function names and options are likely to change as the design stabilizes.

---

## Project Status and Roadmap

ALASTR is **under active development** and not yet recommended for routine clinical or research deployment. Near-term goals include:

- Stabilizing the package layout and configuration system.
- Implementing an end-to-end demo pipeline on a small aphasia dataset.
- Adding basic tests and continuous integration.
- Documenting example workflows and key metrics for clinical researchers.

---

## Citation and Contributions

A formal citation will be provided once an ALASTR methods paper is available. Until then, if you use concepts or code from this repository in academic work, please:

- Cite the GitHub repository URL, and
- Acknowledge ALASTR as an early-stage tool under development.

Issues, suggestions, and (well-scoped) pull requests are welcome, with the understanding that the codebase is still evolving.
