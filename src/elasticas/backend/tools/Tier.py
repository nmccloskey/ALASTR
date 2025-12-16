import re
import logging
logger = logging.getLogger("CustomLogger")


class Tier:
    def __init__(self, name, partition=False, search_str=None):
        """
        Initializes a Tier object.

        Args:
            name (str): The name of the tier (column name if no regex).
            search_str (str, optional): A regex pattern to match values.
            partition (bool): Whether this tier is used for partitioning.
        """
        try:
            self.name = str(name)
            self.partition = partition

            if search_str:
                self.search_str = re.compile(search_str)
                logger.info(f"Initialized Tier: {self.name}, partition: {partition}, regex: {search_str}")
            else:
                self.search_str = None
                logger.info(f"Initialized Tier: {self.name} (column match), partition: {partition}")

        except re.error as e:
            logger.error(f"Invalid regex pattern for Tier '{name}': {e}")
            raise ValueError(f"Invalid regex pattern: {search_str}")

    def match(self, text):
        """
        Matches text against the regex pattern (if provided).

        Args:
            text (str): The text to be analyzed.

        Returns:
            str: Matched value, tier name (for columns), or None if no match.
        """
        try:
            if self.search_str:
                match = self.search_str.search(text)
                if match:
                    logger.debug(f"Match found for '{self.name}' in text: {text}")
                    return match.group(0)
                else:
                    logger.warning(f"No match found for regex tier '{self.name}' in text: {text}.")
                    return None
            else:
                return None

        except Exception as e:
            logger.error(f"Error processing text match for Tier '{self.name}': {e}")
            return None


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
