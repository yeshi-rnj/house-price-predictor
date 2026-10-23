"""Indian-style currency formatting."""
from __future__ import annotations


def format_inr(amount: float) -> str:
    """Format rupees with Indian digit grouping: 7500000 -> "₹75,00,000"."""
    value = int(round(amount))
    sign = "-" if value < 0 else ""
    digits = str(abs(value))
    if len(digits) <= 3:
        return f"{sign}₹{digits}"
    head, tail = digits[:-3], digits[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    return f"{sign}₹{','.join(groups)},{tail}"


def inr_in_words(amount: float) -> str:
    """Short form: 7500000 -> "₹75 lakh", 25000000 -> "₹2.5 crore"."""
    value = abs(amount)
    if value >= 10_000_000:
        number, unit = value / 10_000_000, "crore"
    elif value >= 100_000:
        number, unit = value / 100_000, "lakh"
    else:
        return format_inr(amount)
    text = f"{number:.2f}".rstrip("0").rstrip(".")
    return f"₹{text} {unit}"
