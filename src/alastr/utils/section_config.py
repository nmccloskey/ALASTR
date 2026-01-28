"""
Goals implemented:
- One callable per *subsection* (analysis unit), not one per section.
- Canonical IDs are `<section>.<subsection>` (“analysis_id”) to avoid collisions.
- Output structure is described *per analysis_id* (workbook + sheets).
- “Variants” (cleaned/tokenized/chat_* etc.) are treated as first-class optional
  dimensions: analyses declare what they support; the orchestrator iterates only
  what's declared.
- Your config.yaml section_selection maps cleanly onto analysis_id enablement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

# ---- Import subsection functions (one per analysis unit) ----
# Preprocessing
from alastr.analyses.preprocessing.preprocess_text import preprocess_text

# Lexicon subsections
from alastr.analyses.lexicon.frequencies import analyze_frequencies
from alastr.analyses.lexicon.richness import analyze_richness
from alastr.analyses.lexicon.named_entities import analyze_named_entities
from alastr.analyses.lexicon.readability import analyze_readability
from alastr.analyses.lexicon.lexicon_ngrams import analyze_lexicon_ngrams

# Morphology subsections
from alastr.analyses.morphology.morph_basic_specs import analyze_morph_basic_specs
from alastr.analyses.morphology.morpheme_tags import analyze_morpheme_tags
from alastr.analyses.morphology.morphology_ngrams import analyze_morphology_ngrams

# Syntax subsections
from alastr.analyses.syntax.syntactic_trees import analyze_syntactic_trees
from alastr.analyses.syntax.tree_comparisons import analyze_tree_comparisons
from alastr.analyses.syntax.dependency_tags import analyze_dependency_tags
from alastr.analyses.syntax.pos_tags import analyze_pos_tags
from alastr.analyses.syntax.dep_tag_ngrams import analyze_dep_tag_ngrams
from alastr.analyses.syntax.pos_tag_ngrams import analyze_pos_tag_ngrams

# Phonology subsections
from alastr.analyses.phonology.phon_basic_specs import analyze_phon_basic_specs
from alastr.analyses.phonology.syllables import analyze_syllables
from alastr.analyses.phonology.phonemes import analyze_phonemes
from alastr.analyses.phonology.phon_features import analyze_phon_features
from alastr.analyses.phonology.phon_word_lengths import analyze_phon_word_lengths
from alastr.analyses.phonology.phon_ngrams import analyze_phon_ngrams

# Semantics subsections
from alastr.analyses.semantics.unit_similarity import analyze_unit_similarity
from alastr.analyses.semantics.NRCLex import analyze_nrclex
from alastr.analyses.semantics.VADER import analyze_vader
from alastr.analyses.semantics.TextBlob import analyze_textblob
from alastr.analyses.semantics.Afinn import analyze_afinn
from alastr.analyses.semantics.topics import analyze_topics

# Mechanics subsections
from alastr.analyses.mechanics.lg_tool_errors import analyze_lg_tool_errors


# ----------------------------
# Canonical metadata structures
# ----------------------------

Level = str      # "doc" | "sent"
Variant = str    # e.g., "cleaned" | "tokenized" | "chat_phon" | "chat_sem"

@dataclass(frozen=True)
class OutputSpec:
    """
    Describes how an analysis writes tables.
    - workbook_stem: base filename without _<level>.xlsx suffix
    - sheets: list of sheet stems. Conventionally you append _<variant> when needed.
      You can also keep variants as separate sheets, or encode them as a `variant` column.
    """
    workbook_stem: str
    sheets: List[str]


@dataclass(frozen=True)
class AnalysisSpec:
    """
    One runnable analysis unit (subsection).
    """
    analysis_id: str                                  # e.g., "lexicon.readability"
    func: Callable[..., dict]                          # returns a record dict (or dict of dicts)
    section: str                                       # e.g., "lexicon"
    subsection: str                                    # e.g., "readability"
    levels: Tuple[Level, ...]                          # ("doc","sent") usually
    variants: Tuple[Variant, ...]                      # () if variants not relevant
    outputs: Tuple[OutputSpec, ...]                    # one or more workbooks
    requires: Tuple[str, ...] = ()                     # required metadata keys (e.g., narrative, speaking_time)
    # Optional quality-of-life flags
    is_ngram: bool = False

# ----------------------------
# SECTION_CONFIG (now per analysis_id)
# ----------------------------

def build_section_config(ngrams: int = 5) -> Dict[str, AnalysisSpec]:
    """
    Build the analysis registry (your former SECTION_CONFIG), now keyed by analysis_id.

    Parameters
    ----------
    ngrams : int
        Maximum n for any n-gram analyses.

    Returns
    -------
    dict[str, AnalysisSpec]
        Registry describing all analyses ALASTR can run.
    """
    # Common dynamic sheet lists for ngrams
    def ngram_sheets(prefix: str) -> List[str]:
        return [f"{prefix}_ngram_summary"] + [f"{prefix}_n{n}grams" for n in range(1, ngrams + 1)]

    return {

        # ---- preprocessing ----
        "preprocessing.preprocess_text": AnalysisSpec(
            analysis_id="preprocessing.preprocess_text",
            func=preprocess_text,
            section="preprocessing",
            subsection="preprocess_text",
            levels=("doc", "sent"),            # if you truly preprocess both, keep; else change to ("doc",)
            variants=(),                       # preprocessing itself is the thing producing variants
            outputs=(
                OutputSpec(
                    workbook_stem="preprocessing",
                    sheets=["preprocessed"],   # one sheet holding sample_data/sample_text etc.
                ),
            ),
        ),

        # ---- lexicon ----
        "lexicon.frequencies": AnalysisSpec(
            analysis_id="lexicon.frequencies",
            func=analyze_frequencies,
            section="lexicon",
            subsection="frequencies",
            levels=("doc", "sent"),
            variants=("cleaned", "tokenized"),
            outputs=(
                OutputSpec(
                    workbook_stem="lex_measures",
                    sheets=["freqs"],          # convention: write freqs_cleaned and freqs_tokenized (or two sheets)
                ),
            ),
        ),

        "lexicon.richness": AnalysisSpec(
            analysis_id="lexicon.richness",
            func=analyze_richness,
            section="lexicon",
            subsection="richness",
            levels=("doc", "sent"),
            variants=("cleaned", "tokenized"),
            outputs=(
                OutputSpec(workbook_stem="lex_measures", sheets=["richness"]),
            ),
        ),

        "lexicon.named_entities": AnalysisSpec(
            analysis_id="lexicon.named_entities",
            func=analyze_named_entities,
            section="lexicon",
            subsection="named_entities",
            levels=("doc", "sent"),
            variants=("cleaned",),             # pick your policy; often NER wants cleaned text
            outputs=(
                OutputSpec(workbook_stem="lex_measures", sheets=["named_entities"]),
            ),
        ),

        "lexicon.readability": AnalysisSpec(
            analysis_id="lexicon.readability",
            func=analyze_readability,
            section="lexicon",
            subsection="readability",
            levels=("doc", "sent"),
            variants=("cleaned",),             # readability typically wants cleaned, non-tokenized
            outputs=(
                OutputSpec(workbook_stem="lex_measures", sheets=["readability"]),
            ),
        ),

        "lexicon.lexicon_ngrams": AnalysisSpec(
            analysis_id="lexicon.lexicon_ngrams",
            func=analyze_lexicon_ngrams,
            section="lexicon",
            subsection="lexicon_ngrams",
            levels=("doc", "sent"),
            variants=("tokenized",),           # token sequences; adjust if you also support cleaned
            outputs=(
                OutputSpec(workbook_stem="lex_ngrams", sheets=ngram_sheets("lex")),
            ),
        ),

        # ---- morphology ----
        "morphology.morph_basic_specs": AnalysisSpec(
            analysis_id="morphology.morph_basic_specs",
            func=analyze_morph_basic_specs,
            section="morphology",
            subsection="morph_basic_specs",
            levels=("doc", "sent"),
            variants=("tokenized",),           # usually depends on token/POS/morph pipeline
            outputs=(
                OutputSpec(workbook_stem="morph_stats", sheets=["morpheme_basic_specs"]),
            ),
        ),

        "morphology.morpheme_tags": AnalysisSpec(
            analysis_id="morphology.morpheme_tags",
            func=analyze_morpheme_tags,
            section="morphology",
            subsection="morpheme_tags",
            levels=("doc", "sent"),
            variants=("tokenized",),
            outputs=(
                OutputSpec(
                    workbook_stem="morph_stats",
                    sheets=[
                        "morph_tag_counts", "morph_tag_props", "morph_tags_commonest",
                        "morph_tag_sets_commonest",
                    ],
                ),
            ),
        ),

        "morphology.morphology_ngrams": AnalysisSpec(
            analysis_id="morphology.morphology_ngrams",
            func=analyze_morphology_ngrams,
            section="morphology",
            subsection="morphology_ngrams",
            levels=("doc", "sent"),
            variants=("tokenized",),
            outputs=(
                OutputSpec(workbook_stem="morph_ngrams", sheets=ngram_sheets("morph")),
            ),
        ),

        # ---- syntax ----
        "syntax.syntactic_trees": AnalysisSpec(
            analysis_id="syntax.syntactic_trees",
            func=analyze_syntactic_trees,
            section="syntax",
            subsection="syntactic_trees",
            levels=("doc", "sent"),
            variants=("tokenized",),
            outputs=(
                OutputSpec(workbook_stem="syntax_measures", sheets=["syn_trees"]),
            ),
        ),

        "syntax.tree_comparisons": AnalysisSpec(
            analysis_id="syntax.tree_comparisons",
            func=analyze_tree_comparisons,
            section="syntax",
            subsection="tree_comparisons",
            levels=("doc", "sent"),
            variants=("tokenized",),
            outputs=(
                OutputSpec(workbook_stem="syntax_measures", sheets=["tree_comp"]),
            ),
        ),

        "syntax.dependency_tags": AnalysisSpec(
            analysis_id="syntax.dependency_tags",
            func=analyze_dependency_tags,
            section="syntax",
            subsection="dependency_tags",
            levels=("doc", "sent"),
            variants=("tokenized",),
            outputs=(
                OutputSpec(workbook_stem="syntax_measures", sheets=["dep_tag_counts", "dep_tag_props", "dep_tags_commonest"]),
            ),
        ),

        "syntax.pos_tags": AnalysisSpec(
            analysis_id="syntax.pos_tags",
            func=analyze_pos_tags,
            section="syntax",
            subsection="pos_tags",
            levels=("doc", "sent"),
            variants=("tokenized",),
            outputs=(
                OutputSpec(workbook_stem="morph_stats", sheets=["pos_tag_counts", "pos_tag_props", "pos_tags_commonest"]),
            ),
        ),

        "syntax.dep_tag_ngrams": AnalysisSpec(
            analysis_id="syntax.dep_tag_ngrams",
            func=analyze_dep_tag_ngrams,
            section="syntax",
            subsection="dep_tag_ngrams",
            levels=("doc", "sent"),
            variants=("tokenized",),
            outputs=(
                OutputSpec(workbook_stem="dep_tag_ngrams", sheets=ngram_sheets("dep_tag")),
            ),
        ),

        "syntax.pos_tag_ngrams": AnalysisSpec(
            analysis_id="syntax.pos_tag_ngrams",
            func=analyze_pos_tag_ngrams,
            section="syntax",
            subsection="pos_tag_ngrams",
            levels=("doc", "sent"),
            variants=("tokenized",),
            outputs=(
                OutputSpec(workbook_stem="pos_ngrams", sheets=ngram_sheets("pos")),
            ),
        ),

        # ---- phonology ----
        "phonology.phon_basic_specs": AnalysisSpec(
            analysis_id="phonology.phon_basic_specs",
            func=analyze_phon_basic_specs,
            section="phonology",
            subsection="phon_basic_specs",
            levels=("doc", "sent"),
            variants=("chat_phon",),            # your optional “CHAT version”
            outputs=(
                OutputSpec(workbook_stem="phoneme_stats", sheets=["phoneme_basic_specs"]),
            ),
        ),

        "phonology.syllables": AnalysisSpec(
            analysis_id="phonology.syllables",
            func=analyze_syllables,
            section="phonology",
            subsection="syllables",
            levels=("doc", "sent"),
            variants=("chat_phon",),
            outputs=(
                OutputSpec(workbook_stem="phoneme_stats", sheets=["syllable_stats"]),
            ),
        ),

        "phonology.phonemes": AnalysisSpec(
            analysis_id="phonology.phonemes",
            func=analyze_phonemes,
            section="phonology",
            subsection="phonemes",
            levels=("doc", "sent"),
            variants=("chat_phon",),
            outputs=(
                OutputSpec(workbook_stem="phoneme_stats", sheets=["phoneme_counts", "phoneme_props", "phoneme_commonest"]),
            ),
        ),

        "phonology.phon_features": AnalysisSpec(
            analysis_id="phonology.phon_features",
            func=analyze_phon_features,
            section="phonology",
            subsection="phon_features",
            levels=("doc", "sent"),
            variants=("chat_phon",),
            outputs=(
                OutputSpec(workbook_stem="phoneme_stats", sheets=["phon_feature_counts", "phon_feature_props"]),
            ),
        ),

        "phonology.phon_word_lengths": AnalysisSpec(
            analysis_id="phonology.phon_word_lengths",
            func=analyze_phon_word_lengths,
            section="phonology",
            subsection="phon_word_lengths",
            levels=("doc", "sent"),
            variants=("chat_phon",),
            outputs=(
                OutputSpec(workbook_stem="phoneme_stats", sheets=["word_lens_counts", "word_lens_props"]),
            ),
        ),

        "phonology.phon_ngrams": AnalysisSpec(
            analysis_id="phonology.phon_ngrams",
            func=analyze_phon_ngrams,
            section="phonology",
            subsection="phon_ngrams",
            levels=("doc", "sent"),
            variants=("chat_phon",),
            outputs=(
                OutputSpec(workbook_stem="phon_ngrams", sheets=ngram_sheets("phon")),
            ),
        ),

        # ---- semantics ----
        "semantics.unit_similarity": AnalysisSpec(
            analysis_id="semantics.unit_similarity",
            func=analyze_unit_similarity,
            section="semantics",
            subsection="unit_similarity",
            levels=("doc", "sent"),
            variants=("cleaned",),              # or tokenized; depends on your implementation
            outputs=(
                OutputSpec(workbook_stem="semantic_data", sheets=["unit_sim"]),
            ),
        ),

        "semantics.NRCLex": AnalysisSpec(
            analysis_id="semantics.NRCLex",
            func=analyze_nrclex,
            section="semantics",
            subsection="NRCLex",
            levels=("doc", "sent"),
            variants=("cleaned",),
            outputs=(
                OutputSpec(workbook_stem="semantic_data", sheets=["NRCLex"]),
            ),
        ),

        "semantics.VADER": AnalysisSpec(
            analysis_id="semantics.VADER",
            func=analyze_vader,
            section="semantics",
            subsection="VADER",
            levels=("doc", "sent"),
            variants=("cleaned",),
            outputs=(
                OutputSpec(workbook_stem="semantic_data", sheets=["VADER"]),
            ),
        ),

        "semantics.TextBlob": AnalysisSpec(
            analysis_id="semantics.TextBlob",
            func=analyze_textblob,
            section="semantics",
            subsection="TextBlob",
            levels=("doc", "sent"),
            variants=("cleaned",),
            outputs=(
                OutputSpec(workbook_stem="semantic_data", sheets=["TextBlob"]),
            ),
        ),

        "semantics.Afinn": AnalysisSpec(
            analysis_id="semantics.Afinn",
            func=analyze_afinn,
            section="semantics",
            subsection="Afinn",
            levels=("doc", "sent"),
            variants=("cleaned",),
            outputs=(
                OutputSpec(workbook_stem="semantic_data", sheets=["Afinn"]),
            ),
        ),

        "semantics.topics": AnalysisSpec(
            analysis_id="semantics.topics",
            func=analyze_topics,
            section="semantics",
            subsection="topics",
            levels=("doc", "sent"),
            variants=("tokenized",),
            outputs=(
                OutputSpec(workbook_stem="semantic_data", sheets=["topics"]),
            ),
        ),

        # ---- mechanics ----
        "mechanics.lg_tool_errors": AnalysisSpec(
            analysis_id="mechanics.lg_tool_errors",
            func=analyze_lg_tool_errors,
            section="mechanics",
            subsection="lg_tool_errors",
            levels=("doc", "sent"),
            variants=("cleaned",),
            outputs=(
                OutputSpec(workbook_stem="errors", sheets=["lg_tool"]),
            ),
        ),
    }


# ---------------------------------------------------------
# Optional: mapping config.yaml selection keys -> analysis_id
# ---------------------------------------------------------

CONFIG_SELECTION_TO_ANALYSIS_ID: Dict[str, Dict[str, str]] = {
    "lexicon_selections": {
        "frequencies": "lexicon.frequencies",
        "richness": "lexicon.richness",
        "named_entities": "lexicon.named_entities",
        "readability": "lexicon.readability",
        "lexicon_ngrams": "lexicon.lexicon_ngrams",
    },
    "morphology_selections": {
        "morph_basic_specs": "morphology.morph_basic_specs",
        "morpheme_tags": "morphology.morpheme_tags",
        "morphology_ngrams": "morphology.morphology_ngrams",
    },
    "syntax_selections": {
        "syntactic_trees": "syntax.syntactic_trees",
        "tree_comparisons": "syntax.tree_comparisons",
        "dependency_tags": "syntax.dependency_tags",
        "pos_tags": "syntax.pos_tags",
        "dep_tag_ngrams": "syntax.dep_tag_ngrams",
        "pos_tag_ngrams": "syntax.pos_tag_ngrams",
    },
    "phonology_selections": {
        "phon_basic_specs": "phonology.phon_basic_specs",
        "syllables": "phonology.syllables",
        "phonemes": "phonology.phonemes",
        "phon_features": "phonology.phon_features",
        "phon_word_lengths": "phonology.phon_word_lengths",
        "phon_ngrams": "phonology.phon_ngrams",
    },
    "semantics_selections": {
        "unit_similarity": "semantics.unit_similarity",
        "NRCLex": "semantics.NRCLex",
        "VADER": "semantics.VADER",
        "TextBlob": "semantics.TextBlob",
        "Afinn": "semantics.Afinn",
        "topics": "semantics.topics",
    },
    "mechanics_selections": {
        "lg_tool_errors": "mechanics.lg_tool_errors",
    },
}


def resolve_enabled_analyses(cfg: dict, registry: Dict[str, AnalysisSpec]) -> Set[str]:
    """
    Deterministically resolve which analysis_ids are enabled from your config.yaml.

    Precedence (most specific wins):
      1) subsection selection within `<section>_selections`
      2) `<section>_all`
      3) `sections_all`

    Global ngrams override:
      - If cfg["ngrams_all"] == 1, any analysis_id whose subsection endswith "_ngrams"
        (or otherwise tagged) is enabled unless explicitly disabled at level (1).

    Notes:
      - I kept this conservative: explicit 0 at the most specific level disables.
      - You can change the heuristic for “is ngram analysis” to a tag in AnalysisSpec.
    """
    sections_all = int(cfg.get("sections_all", 0)) == 1
    ngrams_all = int(cfg.get("ngrams_all", 0)) == 1

    sel_root = cfg.get("section_selection", {}) or {}

    enabled: Set[str] = set()

    # Step 1: base enablement via section/subsection toggles
    for section_id in ("lexicon", "morphology", "syntax", "phonology", "semantics", "mechanics"):
        section_all_key = f"{section_id}_all"
        selections_key = f"{section_id}_selections"

        section_all = int(sel_root.get(section_all_key, 0)) == 1
        subsection_map = sel_root.get(selections_key, {}) or {}
        key_map = CONFIG_SELECTION_TO_ANALYSIS_ID.get(selections_key, {})

        for subsection_key, analysis_id in key_map.items():
            # Most specific: subsection toggle
            if subsection_key in subsection_map:
                if int(subsection_map.get(subsection_key, 0)) == 1:
                    enabled.add(analysis_id)
                continue

            # Next: section_all
            if section_all:
                enabled.add(analysis_id)
                continue

            # Next: sections_all
            if sections_all:
                enabled.add(analysis_id)

    # Step 2: optional ngrams_all behavior (override-on enable only)
    if ngrams_all:
        for analysis_id, spec in registry.items():
            if spec.subsection.endswith("_ngrams"):
                enabled.add(analysis_id)

    # Always include preprocessing (or gate it separately if you want)
    if "preprocessing.preprocess_text" in registry:
        enabled.add("preprocessing.preprocess_text")

    # Filter to what exists in registry
    return {aid for aid in enabled if aid in registry}
