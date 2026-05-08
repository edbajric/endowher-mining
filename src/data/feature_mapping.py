import re
from typing import Dict, Iterable


def normalize_feature_name(name: str) -> str:
    lowered = name.strip().lower()
    lowered = re.sub(r"[^a-z0-9]+", "_", lowered)
    return lowered.strip("_")


def build_feature_map(features: Iterable[str]) -> Dict[str, str]:
    return {feature: normalize_feature_name(feature) for feature in features}
