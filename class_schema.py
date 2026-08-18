"""Shared class-name and temporary synthetic-label conventions."""

SYNTHETIC_CLASS_ID_OFFSET = 1_000_000_000

_CLASS_NAME_ALIASES = {
    "BAG_BIBIGO_CHEONGYANG_MEAT_DUMPLINGS_COLD_200G":
        "BAG_BIBIGO_CHEONGYANG_MEAT_DUMPLINGS_200G",
}


def canonical_class_name(name: str) -> str:
    value = str(name or "").strip()
    return _CLASS_NAME_ALIASES.get(value.upper(), value)


def canonical_class_key(name: str) -> str:
    return canonical_class_name(name).casefold()


def encode_synthetic_class_id(train_product_idx: int) -> int:
    return SYNTHETIC_CLASS_ID_OFFSET + int(train_product_idx)
