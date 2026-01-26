import re
import random
from typing import List
from alastr.backend.tools.logger import logger


class Tier:
    def __init__(self, name: str, values: List[str], partition: bool, blind: bool):
        """
        Initializes a Tier object.

        Parameters:
        - name (str): The name of the tier.
        - values (list[str]): Values used to create/represent the regex. If len(values) == 1,
                              we treat values[0] as a user-provided regex. If len(values) > 1,
                              we build a regex that matches any of the literal values.
        - partition (bool): Whether this tier is used for partitioning.
        - blind (bool): Whether this tier is blinded in CU summaries.
        """
        self.name = name
        self.values = values or []
        self.partition = partition
        self.blind = blind

        # Decide whether to treat values as a direct regex or as literal choices
        if len(self.values) == 1:
            # User provided a single regex string
            self.is_user_regex = True
            self.search_str = self.values[0]
            try:
                self.pattern = re.compile(self.search_str)
            except re.error as e:
                raise ValueError(
                    f"Tier '{self.name}': invalid regex provided: {self.search_str!r}. "
                    f"Regex compile error: {e}"
                )
            logger.info(
                f"Initialized Tier '{self.name}' with user regex: {self.search_str!r} "
                f"(partition={self.partition}, blind={self.blind})"
            )
        else:
            # Build a regex from multiple literal values
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
                f"Initialized Tier '{self.name}' with {len(self.values)} literal values "
                f"(partition={self.partition}, blind={self.blind}). Regex={self.search_str!r}"
            )

    def _make_search_string(self, values: List[str]) -> str:
        """
        Generates a regex from provided literal values (escaped, joined with '|').
        Returns a non-capturing group: (?:v1|v2|...)
        """
        if not values:
            logger.warning(f"Tier '{self.name}' received empty values; regex will never match.")
            return r"(?!x)x"  # matches nothing

        # Escape each literal to avoid accidental regex meta-characters
        escaped = [re.escape(v) for v in values]
        search_str = "(?:" + "|".join(escaped) + ")"
        logger.debug(f"Tier '{self.name}': generated search string from literals: {search_str}")
        return search_str

    def match(self, text: str, return_None: bool = False, must_match: bool = False):
        """
        Applies the compiled regex pattern to a given text.

        Returns:
        - str: The matched value if found (match.group(0)).
        - None: If no match is found and return_None is True.
        - str: The tier name if no match is found and return_None is False (legacy behavior).
        """
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

    def make_blind_codes(self):
        """
        Generates a blinded coding system for the tier values (for literal-value tiers).
        For user-regex tiers, 'values' may not be an exhaustive set—use with caution.
        """
        logger.info(f"Generating blind codes for tier: {self.name}")
        if not self.values:
            logger.warning(f"Tier '{self.name}' has no values; blind code mapping will be empty.")
            return {self.name: {}}

        blind_codes = list(range(len(self.values)))
        random.shuffle(blind_codes)
        blind_code_mapping = {k: v for k, v in zip(self.values, blind_codes)}
        logger.debug(f"Blind code mapping for '{self.name}': {blind_code_mapping}")
        return {self.name: blind_code_mapping}


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

    def default_tiers() -> dict:
        """Return a default single-tier mapping that matches the entire filename."""
        logger.warning("No valid tiers detected — defaulting to full filename match ('.*(?=\.cha)').")
        default_name = "file_name"
        # one regex string in a list → treated as user regex
        return {default_name: Tier(name=default_name, values=[r".*(?=\.cha)"], partition=False, blind=False)}

    def read_tiers(self, config_tiers: dict | None) -> dict[str, Tier]:
        """
        Parse tier definitions from a configuration dictionary into Tier objects.

        Behavior
        --------
        - Input must be a dict mapping tier name → definition.
        - Each definition may be:
            * dict with keys:
                - 'values': list[str] | str (required)
                - 'partition': bool (optional)
                - 'blind': bool (optional)
            * or legacy shorthand: list[str] or str.
        - A single 'values' entry is treated as a user regex (validated).
        - Multiple entries are treated literally.
        - Empty, invalid, or missing tiers trigger fallback behavior.

        Returns
        -------
        dict[str, Tier]
            Mapping of tier name → Tier object.

        Logging
        --------
        - Warns if no usable tiers found.
        - Errors on regex compilation failures or invalid structures.
        - Info-level notices for partition/blind flags.
        """

        if not config_tiers or not isinstance(config_tiers, dict):
            logger.warning("Tier config missing or invalid; using default tiers.")
            return self.default_tiers()

        tiers: dict[str, Tier] = {}

        for tier_name, tier_data in config_tiers.items():
            try:
                # Normalize structure
                if isinstance(tier_data, (str, list)):
                    tier_data = {"values": [tier_data] if isinstance(tier_data, str) else tier_data}

                values = tier_data.get("values", [])
                if isinstance(values, str):
                    values = [values]

                partition = bool(tier_data.get("partition", False))
                blind = bool(tier_data.get("blind", False))

                if not values:
                    logger.warning(f"Tier '{tier_name}' has no values; it will never match.")
                    tiers[tier_name] = Tier(tier_name, [], partition=partition, blind=blind)
                    continue

                # Validate / build regex behavior
                if len(values) == 1:
                    user_regex = values[0]
                    try:
                        re.compile(user_regex)
                    except re.error as e:
                        logger.error(f"Tier '{tier_name}': invalid regex {user_regex!r} — {e}")
                        continue
                    logger.info(f"Tier '{tier_name}' using user regex {user_regex!r}")
                    tier_obj = Tier(tier_name, [user_regex], partition=partition, blind=blind)
                else:
                    logger.info(f"Tier '{tier_name}' using {len(values)} literal values.")
                    tier_obj = Tier(tier_name, values, partition=partition, blind=blind)

                tiers[tier_name] = tier_obj

                if partition:
                    logger.info(f"Tier '{tier_name}' marked as partition level.")
                if blind:
                    logger.info(f"Tier '{tier_name}' marked as blind column.")

            except Exception as e:
                logger.error(f"Failed to parse tier '{tier_name}': {e}")

        if not tiers:
            logger.warning("No valid tiers created — using default tiers.")
            tiers = self.default_tiers()

        logger.info(f"Finished parsing tiers. Total: {len(tiers)}")
        return tiers

    def _init_tiers(self):
        """
        Initializes tiers based on configuration.

        Args:
            config (dict): Configuration dictionary containing tier names and optional regex patterns.
        """
        tier_config = self.OM.config.get("tiers", {})
        if not tier_config:
            logger.warning("No configuration provided for TierManager.")
            return

        try:
            tiers = self.read_tiers(tier_config)
            self.tiers = tiers
        except Exception as e:
            logger.error(f"Error reading tiers: {e}")
            return {}

        for tier_name in tier_config:
            try:
                tier_name = self.OM.db.sanitize_column_name(tier_name)
                new_tier = Tier(tier_name, tier_config[tier_name]["partition"], tier_config[tier_name]["regex"])
                self.tiers[tier_name] = new_tier

            except ValueError as e:
                logger.error(f"Skipping Tier '{tier_name}' due to invalid regex: {e}")
                continue

        logger.info(f"Tiers: {[(t.name, t.partition, t.search_str) for t in self.tiers.values()]}")
    
    def get_tier_names(self):
        """Returns list of tier names."""
        return list(self.tiers.keys())

    def get_partition_tiers(self):
        """
        Retrieves partitioning tiers.

        Returns:
            list: List of Tier names used for partitioning.
        """
        return [tier.name for tier in self.tiers.values() if tier.partition]

    def match_tiers(self, text):
        """
        Applies all tiers to the given text.

        Args:
            text (str): The text to be analyzed.

        Returns:
            dict: Mapping of tier names to their matched values.
        """
        results = {}
        for tier in self.tiers.values():
            results[tier.name] = tier.match(text)
        return results

    def make_tier(self, tier_name, partition=False, search_str=None):
        if tier_name not in self.tiers.keys():
            tier_name = self.OM.db.sanitize_column_name(tier_name)
            new_tier = Tier(tier_name, partition, search_str)
            logger.info(f"Added Tier '{tier_name}' partition: {partition}")
            return new_tier
        else:
            logger.warning(f"Tier {tier_name} already exists.")
