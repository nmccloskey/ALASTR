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
