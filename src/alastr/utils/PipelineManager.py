from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

from alastr.backend.tools.logger import logger, _rel
from alastr.utils.section_config import AnalysisSpec, build_section_config, resolve_enabled_analyses

Level = str
Variant = str


class PipelineManager:
    """
    Orchestrates ALASTR execution using an analysis registry keyed by analysis_id.

    PipelineManager governs:
      - analysis registry construction
      - enabled analysis resolution
      - execution order
      - orchestration of levels / variants / samples
      - delegation to IOManager for persistence

    De facto singleton:
      - Use PipelineManager.get_instance(...) for normal runs
      - Instantiate directly in tests if needed
    """

    _instance: Optional["PipelineManager"] = None
    _initialized: bool = False

    # -------------------------
    # De facto singleton factory
    # -------------------------

    @classmethod
    def get_instance(
        cls,
        io_manager,
        *,
        ngrams: int = 5,
        export_after_each_analysis: bool = True,
        reset: bool = False,
    ) -> "PipelineManager":
        """
        Preferred construction method for normal runs.

        Parameters
        ----------
        io_manager:
            IOManager instance governing config, inputs, outputs.
        ngrams:
            Maximum n-gram size for n-gram analyses.
        export_after_each_analysis:
            If True, exports after each analysis_id completes.
        reset:
            If True, discards any cached instance and creates a new one.
            Useful for tests and Streamlit reruns.
        """
        if reset:
            cls._instance = None
            cls._initialized = False

        if cls._instance is None:
            cls._instance = cls(
                io_manager,
                ngrams=ngrams,
                export_after_each_analysis=export_after_each_analysis,
            )
        return cls._instance

    # -----------------
    # Construction
    # -----------------

    def __init__(
        self,
        io_manager,
        ngrams: int = 5,
        export_after_each_analysis: bool = True,
    ):
        """
        Notes on guarding:
        - We only guard *re-initialization* when this object is the cached singleton.
        - If you instantiate PipelineManager(...) directly in a test, it will fully initialize.
        """
        if self.__class__._initialized and self is self.__class__._instance:
            return

        # --- core references ---
        self.io = io_manager               # rename to self.om if you keep OM naming
        self.cfg = io_manager.config

        # --- run controls ---
        self.export_after_each_analysis = export_after_each_analysis

        # --- levels ---
        self.sentence_level = bool(self.cfg.get("sentence_level", False))
        self.levels: Tuple[Level, ...] = ("doc", "sent") if self.sentence_level else ("doc",)

        # --- registry + enabled ids ---
        self.ngrams = int(ngrams)
        self.registry: Dict[str, "AnalysisSpec"] = build_section_config(ngrams=self.ngrams)
        self.enabled_analysis_ids: Set[str] = resolve_enabled_analyses(self.cfg, self.registry)

        # --- lazy init bookkeeping ---
        self._created_tables: Set[str] = set()

        # mark singleton state
        self.__class__._instance = self
        self.__class__._initialized = True
        logger.info("PipelineManager initialized successfully.")

    # ------------------------
    # Public run entry points
    # ------------------------

    def run(self) -> None:
        """
        Run the full pipeline:
        - preprocessing always runs
        - analyses run in deterministic order (sorted by section then subsection)
        """
        # Always run preprocessing first if present
        if "preprocessing.preprocess_text" in self.registry:
            self.run_analysis("preprocessing.preprocess_text")

        # Deterministic run order for remaining analyses
        ordered = sorted(
            (aid for aid in self.enabled_analysis_ids if not aid.startswith("preprocessing.")),
            key=lambda aid: (self.registry[aid].section, self.registry[aid].subsection, aid),
        )
        for aid in ordered:
            self.run_analysis(aid)

        # If you prefer a single final export pass:
        if not self.export_after_each_analysis:
            self.export_all()

    def run_analysis(self, analysis_id: str) -> None:
        """
        Run one analysis_id across its declared dimensions and input samples.
        Tables are lazily created for this analysis right before writing.
        """
        spec = self.registry[analysis_id]

        # Ensure output tables exist for this analysis (lazy creation)
        self._ensure_tables_for_analysis(spec)

        # Decide levels to run = intersection(spec.levels, pm.levels)
        levels_to_run = tuple(lv for lv in spec.levels if lv in self.levels)
        if not levels_to_run:
            return

        # Variants: if spec has none, treat as a single "no variant" pass
        variants_to_run: Tuple[Optional[Variant], ...] = spec.variants or (None,)

        # Iterate samples:
        # - doc_id list must come from OM's grouping table; adapt as needed.
        for level in levels_to_run:
            doc_ids = self._get_doc_ids(level=level)

            for variant in variants_to_run:
                for doc_id in doc_ids:
                    sample = self.get_sample_data(doc_id=doc_id, level=level)

                    # Attach any extra metadata required by this analysis
                    meta = self._extract_required_metadata(sample, level=level, requires=spec.requires)

                    # The analysis function signature is up to you.
                    # Recommendation: func(pm=self, sample=..., level=..., variant=..., **meta)
                    try:
                        record_or_records = spec.func(
                            pm=self,
                            sample=sample,
                            level=level,
                            variant=variant,
                            **meta,
                        )
                    except Exception as e:
                        # Your logger integration goes here
                        # logger.exception(...)
                        raise

                    # Write outputs: a single dict record or list[dict] records
                    self._write_analysis_output(
                        spec=spec,
                        level=level,
                        variant=variant,
                        doc_id=doc_id,
                        record_or_records=record_or_records,
                    )

        # Export policy: after each analysis, flush its section outputs
        if self.export_after_each_analysis:
            self.export_section(spec.section)

    def export_section(self, section_id: str) -> None:
        """
        Export the section's workbooks to Excel.
        Adapt to IOManager's export API.
        """
        if hasattr(self.om, "export_section"):
            self.om.export_section(section_id)
        elif hasattr(self.om, "export_all"):
            # fallback: export everything (less ideal but safe)
            self.om.export_all()
        else:
            # If OM writes incrementally, you may not need explicit export.
            pass

    def export_all(self) -> None:
        if hasattr(self.om, "export_all"):
            self.om.export_all()

    # ------------------------
    # Data access helpers
    # ------------------------

    def get_sample_data(self, doc_id: int, level: Level):
        """
        Retrieve the sample data for one document (and sentences if level=='sent').

        This generalizes your old get_sample_data and removes the implicit use of
        self.sentence_level so analyses can explicitly choose level.
        """
        fact_table = f"sample_text_{level}"
        df = self.om.tables[fact_table].get_data(filters={"doc_id": doc_id})
        if level == "sent":
            df = df.sort_values(by="sent_id").to_dict(orient="records")
        else:
            df = df.to_dict(orient="records")[0]
        return df

    def _get_doc_ids(self, level: Level) -> List[int]:
        """
        Pull doc_ids from your grouping table.
        Adapt if your OM uses a different table name or provides a helper.
        """
        grouping_table = f"sample_data_{level}" if f"sample_data_{level}" in self.om.tables else "sample_data_doc"
        df = self.om.tables[grouping_table].get_data()
        if "doc_id" not in df.columns:
            raise ValueError(f"Grouping table {grouping_table} missing doc_id column.")
        return sorted(df["doc_id"].unique().tolist())

    def _extract_required_metadata(self, sample, level: Level, requires: Tuple[str, ...]) -> dict:
        """
        Extract metadata required by an analysis (e.g., narrative, speaking_time).
        This is intentionally conservative: you should standardize where these live.
        """
        meta = {}
        if not requires:
            return meta

        # Example strategy:
        # - If level == "doc", sample is a dict
        # - If level == "sent", sample is list[dict]; take from first element
        source = sample[0] if isinstance(sample, list) and sample else sample

        for key in requires:
            if key in source:
                meta[key] = source[key]
            else:
                # Prefer to fail fast; or set None and let analysis decide
                meta[key] = None
        return meta

    # ------------------------
    # Output / table creation
    # ------------------------

    def _ensure_tables_for_analysis(self, spec: AnalysisSpec) -> None:
        """
        Lazily create IOManager tables for this analysis.

        Policy:
        - Create tables only for the levels this run will execute (intersection done later),
          but safe to create for spec.levels; cost is small.
        - Create one table per (sheet, level, variant-policy).
          For simplicity, this implementation creates separate tables per variant by
          suffixing sheet names. If you prefer a `variant` column, adjust here.
        """
        # Primary keys by level
        pk_by_level = {"doc": ["doc_id"], "sent": ["doc_id", "sent_id"]}

        # Determine if this analysis has variants
        has_variants = bool(spec.variants)

        for out in spec.outputs:
            for level in spec.levels:
                # workbook path policy: <section>/<level>/<workbook_stem>_<level>.xlsx
                # You can also keep level in filename only; either is fine—just be consistent.
                file_name = f"{out.workbook_stem}_{level}.xlsx"

                for sheet_stem in out.sheets:
                    if has_variants:
                        # Separate sheet per variant
                        for variant in spec.variants:
                            sheet_name = f"{sheet_stem}_{variant}"
                            table_name = f"{spec.analysis_id}.{sheet_name}_{level}"
                            self._create_table_if_needed(
                                table_name=table_name,
                                sheet_name=sheet_name,
                                section=spec.section,
                                subdir=spec.section,          # keep output folders aligned to section_id
                                file_name=file_name,
                                primary_keys=self._primary_keys_for(spec, level, sheet_stem),
                                pivot=self._pivot_for(spec, sheet_stem),
                                level=level,
                                source_fn=spec.func.__name__,
                                analysis_id=spec.analysis_id,
                            )
                    else:
                        sheet_name = sheet_stem
                        table_name = f"{spec.analysis_id}.{sheet_name}_{level}"
                        self._create_table_if_needed(
                            table_name=table_name,
                            sheet_name=sheet_name,
                            section=spec.section,
                            subdir=spec.section,
                            file_name=file_name,
                            primary_keys=self._primary_keys_for(spec, level, sheet_stem),
                            pivot=self._pivot_for(spec, sheet_stem),
                            level=level,
                            source_fn=spec.func.__name__,
                            analysis_id=spec.analysis_id,
                        )

    def _primary_keys_for(self, spec: AnalysisSpec, level: Level, sheet_stem: str) -> List[str]:
        """
        Decide PK strategy, including AUTO for ngram-style long tables with pivots.

        This mirrors your old heuristic:
        - If a table is pivoted by ngram into wide form, PK can be AUTO.
        - Otherwise doc/sent composite PKs.
        """
        if spec.is_ngram or sheet_stem.endswith("grams") or "n" in sheet_stem and "grams" in sheet_stem:
            return ["AUTO"]
        return ["doc_id"] if level == "doc" else ["doc_id", "sent_id"]

    def _pivot_for(self, spec: AnalysisSpec, sheet_stem: str) -> Optional[dict]:
        """
        Provide pivot metadata for ngram sheets (if your OM uses it).
        Adjust this to match your existing Table pivot semantics.
        """
        if spec.is_ngram or sheet_stem.endswith("grams") or "grams" in sheet_stem:
            return {"index": "doc_id", "columns": "ngram", "values": "prop"}
        return None

    def _create_table_if_needed(
        self,
        table_name: str,
        sheet_name: str,
        section: str,
        subdir: str,
        file_name: str,
        primary_keys: List[str],
        pivot: Optional[dict],
        level: Level,
        source_fn: str,
        analysis_id: str,
    ) -> None:
        if table_name in self._created_tables:
            return

        self.om.create_table(
            name=table_name,
            sheet_name=sheet_name,
            section=section,
            subdir=subdir,
            file_name=file_name,
            primary_keys=primary_keys,
            pivot=pivot,
        )

        t = self.om.tables[table_name]
        # Attach metadata similar to your old pattern (adapt as needed)
        t.granularity = level
        t.source_fn = source_fn
        t.analysis_id = analysis_id  # new: keep canonical id on the table object if you want
        # Tagging policy
        if section == "preprocessing":
            t.tags = getattr(t, "tags", []) + ["raw"]
        else:
            t.tags = getattr(t, "tags", []) + ["raw"]
            t.fact_table = f"sample_text_{level}"
            t.grouping_table = f"sample_data_{level}"

        self._created_tables.add(table_name)

    def _write_analysis_output(
        self,
        spec: AnalysisSpec,
        level: Level,
        variant: Optional[Variant],
        doc_id: int,
        record_or_records,
    ) -> None:
        """
        Route analysis output into the appropriate OM tables.

        Expectation:
        - If spec.outputs has one sheet, the analysis can return a single record dict.
        - If multiple sheets, return either:
            a) dict[sheet_stem -> record|records]
            b) or a dict containing a "target_sheet" key per record (less nice)
        This function supports both patterns, but I recommend (a).
        """
        # Normalize to mapping of sheet_stem -> records
        if isinstance(record_or_records, dict) and any(
            isinstance(v, (dict, list)) for v in record_or_records.values()
        ):
            sheet_map = record_or_records
        else:
            # single-sheet assumption: first declared sheet of first output
            first_sheet = spec.outputs[0].sheets[0]
            sheet_map = {first_sheet: record_or_records}

        for sheet_stem, recs in sheet_map.items():
            if recs is None:
                continue
            if isinstance(recs, dict):
                rec_list = [recs]
            else:
                rec_list = list(recs)

            # Decide sheet/table name (variant suffix if applicable)
            if variant is not None and spec.variants:
                sheet_name = f"{sheet_stem}_{variant}"
            else:
                sheet_name = sheet_stem

            table_name = f"{spec.analysis_id}.{sheet_name}_{level}"

            # Write each record
            for rec in rec_list:
                # guarantee doc_id presence
                rec.setdefault("doc_id", doc_id)
                self.om.tables[table_name].insert(rec)
