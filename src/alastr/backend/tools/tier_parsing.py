import re
from .tier import Tier
from alastr.backend.tools.logger import logger


def default_tiers() -> dict:
    """Return a default single-tier mapping that matches the entire filename."""
    logger.warning("No valid tiers detected — defaulting to full filename match ('.*(?=\.cha)').")
    default_name = "file_name"
    # one regex string in a list → treated as user regex
    return {default_name: Tier(name=default_name, values=[r".*(?=\.cha)"], partition=False, blind=False)}

def read_tiers(config_tiers: dict | None) -> dict[str, Tier]:
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
        return default_tiers()

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
        tiers = default_tiers()

    logger.info(f"Finished parsing tiers. Total: {len(tiers)}")
    return tiers
