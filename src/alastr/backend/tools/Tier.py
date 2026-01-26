import re
from typing import List, Union, Optional
from alastr.backend.tools.logger import logger


class Tier:
    def __init__(self, name: str, values: Optional[Union[str, List[str]]]):
        """
        Initializes a Tier object.

        Parameters
        ----------
        name : str
            Tier name (e.g., "site", "study_id").
        values : str | list[str] | None
            - If a single string (or list of length 1): treated as a user-provided regex.
            - If a list of length > 1: treated as literal values; we build a regex matching
              any of them (escaped) via a non-capturing group.
        """
        self.name = name

        # Normalize values to list[str]
        if values is None:
            self.values: List[str] = []
        elif isinstance(values, str):
            self.values = [values]
        else:
            self.values = list(values)

        # Decide whether to treat values as a direct regex or as literal choices
        if len(self.values) == 1:
            self.is_user_regex = True
            self.search_str = self.values[0]
            try:
                self.pattern = re.compile(self.search_str)
            except re.error as e:
                raise ValueError(
                    f"Tier '{self.name}': invalid regex provided: {self.search_str!r}. "
                    f"Regex compile error: {e}"
                )
            logger.info(f"Initialized Tier '{self.name}' with user regex: {self.search_str!r}")
        else:
            self.is_user_regex = False
            self.search_str = self._make_search_string(self.values)
            try:
                self.pattern = re.compile(self.search_str)
            except re.error as e:
                raise ValueError(
                    f"Tier '{self.name}': failed to compile built regex {self.search_str!r}. "
                    f"Compile error: {e}"
                )
            logger.info(
                f"Initialized Tier '{self.name}' with {len(self.values)} literal values: "
                f"Regex={self.search_str!r}"
            )

    def _make_search_string(self, values: List[str]) -> str:
        """
        Build a regex from provided literal values (escaped, joined with '|').
        Returns a non-capturing group: (?:v1|v2|...)
        """
        if not values:
            logger.warning(f"Tier '{self.name}' received empty values; regex will never match.")
            return r"(?!x)x"  # matches nothing

        escaped = [re.escape(v) for v in values]
        return "(?:" + "|".join(escaped) + ")"

    def match(self, text: str, return_None: bool = False, must_match: bool = False):
        m = self.pattern.search(text)
        if m:
            return m.group(0)
        if return_None:
            if must_match:
                logger.warning(f"No match for tier '{self.name}' in text: {text!r}")
            return None
        if must_match:
            logger.error(f"No match for tier '{self.name}' in text: {text!r}. Returning tier name.")
        return self.name


class TierManager:
    _instance = None

    def __new__(cls, OM=None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.tiers = {}
            cls._instance.OM = OM
            cls._instance._init_tiers()
            logger.info("TierManager instance created.")
        return cls._instance

    @staticmethod
    def default_tiers() -> dict:
        """Return a default single-tier mapping that matches the entire filename."""
        logger.warning("No valid tiers detected — defaulting to full filename match ('.*(?=\\.cha)').")
        default_name = "file_name"
        return {default_name: Tier(name=default_name, values=r".*(?=\.cha)")}

    def read_tiers(self, config_tiers: dict | None) -> dict[str, Tier]:
        """
        Parse tier definitions from the config.

        Expects:
          tiers:
            tier_name:
              values: <str regex> OR [<literal>, <literal>, ...]
        """
        if not config_tiers or not isinstance(config_tiers, dict):
            logger.warning("Tier config missing or invalid; using default tiers.")
            return self.default_tiers()

        tiers: dict[str, Tier] = {}

        for raw_name, tier_data in config_tiers.items():
            try:
                tier_name = self.OM.db.sanitize_column_name(raw_name) if self.OM else raw_name

                # Allow shorthand: tier_name: "REGEX"  or  tier_name: ["A","B"]
                if isinstance(tier_data, (str, list)):
                    tier_data = {"values": tier_data}

                values = tier_data.get("values", [])
                tier_obj = Tier(tier_name, values)
                tiers[tier_name] = tier_obj

            except Exception as e:
                logger.error(f"Failed to parse tier '{raw_name}': {e}")

        if not tiers:
            logger.warning("No valid tiers created — using default tiers.")
            return self.default_tiers()

        logger.info(f"Finished parsing tiers. Total: {len(tiers)}")
        return tiers

    def _init_tiers(self):
        """Initialize tiers once, using the config's `tiers` -> `values` logic."""
        if self.OM is None:
            logger.warning("TierManager initialized without OM; using default tiers.")
            self.tiers = self.default_tiers()
            return

        tier_config = self.OM.config.get("tiers", {})
        self.tiers = self.read_tiers(tier_config)
        logger.info(f"Tiers: {[(t.name, t.search_str) for t in self.tiers.values()]}")

    def get_tier_names(self):
        return list(self.tiers.keys())

    def match_tiers(self, text: str):
        return {tier.name: tier.match(text) for tier in self.tiers.values()}

    def make_tier(self, tier_name: str, values: Optional[Union[str, List[str]]] = None):
        tier_name = self.OM.db.sanitize_column_name(tier_name) if self.OM else tier_name
        if tier_name in self.tiers:
            logger.warning(f"Tier {tier_name} already exists.")
            return self.tiers[tier_name]

        new_tier = Tier(tier_name, values)
        self.tiers[tier_name] = new_tier
        logger.info(f"Added Tier '{tier_name}' with search_str={new_tier.search_str!r}")
        return new_tier
