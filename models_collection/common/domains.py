import re

# Unified domain map used across models
DOMAIN_MAP = {
    'QIM': 0,
    'PMS': 1,
    'LSB': 2,
    'AHCM': 3,
}

def normalize_domain_name(name: str) -> str:
    """Normalize user-provided domain tokens to canonical keys in DOMAIN_MAP."""
    if name is None:
        return ''
    token = re.sub(r"\s+", "", str(name)).upper()
    return token

def parse_domain_names_to_ids(domains_csv: str):
    """Parse comma-separated domain names into sorted unique domain ids.

    - Performs strip and whitespace removal on each token
    - Case-insensitive match against DOMAIN_MAP keys
    - Returns a sorted list of unique ids
    """
    if domains_csv is None:
        return []
    tokens = [normalize_domain_name(t) for t in str(domains_csv).split(',')]
    ids = [DOMAIN_MAP.get(t, -1) for t in tokens if t]
    ids = [i for i in ids if i != -1]
    return sorted(list(set(ids)))


