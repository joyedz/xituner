"""Rules for the order-extraction use case, split the same two ways.

This use case exists to test whether XiTuner's machinery is actually generic or
merely intended to be. Its task shape is deliberately UNLIKE brand voice:

    brand_voice       -- prose. Rules are about how the output SOUNDS.
    order_extraction  -- JSON. Rules are about whether the output PARSES and
                         conforms to a schema.

If contract locking, drift detection, bootstrap statistics and the ship verdict
work unchanged across both, "generic" is demonstrated rather than asserted.

The articulable/tacit split follows the same principle as the brand voice: the
guide states what a human would actually write down, and the tacit rules are the
conventions that only show up by reading hundreds of real rows -- the ones
nobody documents because nobody notices them consciously.
"""

from __future__ import annotations

# --- ARTICULABLE: stated in schema_guide.md -------------------------------
ARTICULABLE_RULES: dict[str, str] = {
    "is_valid_json": "Output must be valid JSON",
    "has_all_fields": "Must contain produk, jumlah, ukuran, catatan",
    "no_prose_outside_json": "No explanation or preamble around the JSON",
}

# --- TACIT: present only in the examples ----------------------------------
# A schema doc says "output JSON with these fields". It does not say that a
# missing value is null rather than "" or "-", that jumlah is an int rather
# than "2", that ukuran is normalised to a fixed vocabulary, or that a markdown
# fence around the JSON breaks the downstream parser. Those are learned by
# looking at what the system actually accepted.
TACIT_RULES: dict[str, str] = {
    "missing_is_null": 'Missing values are null -- never "", "-", or omitted',
    "jumlah_is_int": 'jumlah is a JSON number, or null when absent -- never "2"',
    "ukuran_normalised": "ukuran is one of the allowed values, lowercase",
    "produk_lowercase": "produk is lowercase",
    "no_extra_fields": "No fields beyond the four in the schema",
    "no_code_fence": "No markdown code fence around the JSON",
}

REQUIRED_FIELDS = ["produk", "jumlah", "ukuran", "catatan"]

# The normalisation vocabulary. Customers write "gede", "jumbo", "yg besar";
# the system only accepts these.
ALLOWED_UKURAN = ["kecil", "sedang", "besar"]
