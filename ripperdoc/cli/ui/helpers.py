"""Shared helper functions and constants for the Rich UI."""

import random
from typing import List, Optional

from ripperdoc.core.config import get_current_model_profile, get_global_config, ModelProfile


# Fun words to display while the AI is "thinking"
THINKING_WORDS: List[str] = [
    "Accomplishing",
    "Actioning",
    "Actualizing",
    "Baking",
    "Booping",
    "Brewing",
    "Calculating",
    "Cerebrating",
    "Channelling",
    "Churning",
    "Coalescing",
    "Cogitating",
    "Computing",
    "Combobulating",
    "Concocting",
    "Conjuring",
    "Considering",
    "Contemplating",
    "Cooking",
    "Crafting",
    "Creating",
    "Crunching",
    "Deciphering",
    "Deliberating",
    "Determining",
    "Discombobulating",
    "Divining",
    "Doing",
    "Effecting",
    "Elucidating",
    "Enchanting",
    "Envisioning",
    "Finagling",
    "Flibbertigibbeting",
    "Forging",
    "Forming",
    "Frolicking",
    "Generating",
    "Germinating",
    "Hatching",
    "Herding",
    "Honking",
    "Ideating",
    "Imagining",
    "Incubating",
    "Inferring",
    "Manifesting",
    "Marinating",
    "Meandering",
    "Moseying",
    "Mulling",
    "Mustering",
    "Musing",
    "Noodling",
    "Percolating",
    "Perusing",
    "Philosophising",
    "Pontificating",
    "Pondering",
    "Processing",
    "Puttering",
    "Puzzling",
    "Reticulating",
    "Ruminating",
    "Scheming",
    "Schlepping",
    "Shimmying",
    "Simmering",
    "Smooshing",
    "Spelunking",
    "Spinning",
    "Stewing",
    "Sussing",
    "Synthesizing",
    "Thinking",
    "Tinkering",
    "Transmuting",
    "Unfurling",
    "Unravelling",
    "Vibing",
    "Wandering",
    "Whirring",
    "Wibbling",
    "Wizarding",
    "Working",
    "Wrangling",
]


def get_random_thinking_word() -> str:
    """Return a random thinking word for spinner display."""
    return random.choice(THINKING_WORDS)


def get_profile_for_pointer(pointer: str = "main") -> Optional[ModelProfile]:
    """Return the configured ModelProfile for a logical pointer or default."""
    profile = get_current_model_profile(pointer)
    if profile:
        return profile
    config = get_global_config()
    if "default" in config.model_profiles:
        return config.model_profiles.get("default")
    if config.model_profiles:
        first_name = next(iter(config.model_profiles))
        return config.model_profiles.get(first_name)
    return None


__all__ = ["get_profile_for_pointer", "THINKING_WORDS", "get_random_thinking_word"]
