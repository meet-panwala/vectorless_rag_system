"""
index_builder.py
----------------
Builds a PageIndex-style hierarchical catalog tree from the Flipkart JSON dataset.

Field mapping fixed to match actual JSON structure:
  title          → product name
  selling_price  → price (string "921" → float)
  average_rating → rating (string "3.9" → float)
  discount       → discount % ("69% off" → float 69.0)
"""

import json
import os
import re
from collections import defaultdict
from pathlib import Path


# ── Price band definitions ────────────────────────────────────────────────────
PRICE_BANDS = [
    ("budget",  "Under ₹500",      0,    499),
    ("mid",     "₹500 – ₹1,500", 500,  1499),
    ("premium", "Above ₹1,500", 1500, 999999),
]


def _band_for_price(price: float) -> str:
    for key, _, lo, hi in PRICE_BANDS:
        if lo <= price <= hi:
            return key
    return "budget"


def _clean_price(val) -> float:
    """Parse price strings like '921' or '2,999' → float."""
    if val is None:
        return 0.0
    s = str(val).replace("₹", "").replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return 0.0


def _clean_rating(val) -> float:
    """
    Parse rating from actual JSON field 'average_rating'.
    Handles: "3.9" → 3.9, "3.9 out of 5" → 3.9, 3.9 → 3.9
    """
    if val is None:
        return 0.0
    s = str(val).strip()
    m = re.search(r"([\d.]+)", s)
    if m:
        try:
            v = float(m.group(1))
            return v if 0 <= v <= 5 else 0.0
        except ValueError:
            return 0.0
    return 0.0


def _clean_discount(val) -> float:
    """Parse discount: "69% off" → 69.0, 69 → 69.0"""
    if val is None:
        return 0.0
    s = str(val).strip()
    m = re.search(r"([\d.]+)", s)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return 0.0
    return 0.0


def build_catalog_tree(json_path: str, output_path: str) -> dict:
    """
    Load the Flipkart JSON dataset and build a hierarchical catalog tree.
    """
    print(f"[IndexBuilder] Loading data from {json_path} …")
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        products = raw
    elif isinstance(raw, dict):
        for key in ("products", "data", "items", "records"):
            if key in raw:
                products = raw[key]
                break
        else:
            products = []
    else:
        raise ValueError("Unexpected JSON structure.")

    total_products = len(products)
    print(f"[IndexBuilder] Loaded {total_products:,} products.")

    # Debug: show actual field names from first product
    if products:
        sample_keys = list(products[0].keys())
        print(f"[IndexBuilder] Sample fields: {sample_keys}")

    # ── Aggregate stats ───────────────────────────────────────────────────────
    agg = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: {"prices": [], "ratings": [], "discounts": [], "brands": defaultdict(int), "count": 0}
            )
        )
    )
    global_brands = defaultdict(int)

    for idx, p in enumerate(products):
        cat    = str(p.get("category", "Unknown")).strip() or "Unknown"
        subcat = str(p.get("sub_category", "General")).strip() or "General"
        brand  = str(p.get("brand", "Unknown")).strip() or "Unknown"

        # Use correct field names from actual JSON
        price    = _clean_price(p.get("selling_price",
                   p.get("discounted_price", 0)))
        rating   = _clean_rating(p.get("average_rating",
                   p.get("rating", 0)))
        discount = _clean_discount(p.get("discount",
                   p.get("discount_percentage", 0)))
        band     = _band_for_price(price)

        bucket = agg[cat][subcat][band]
        bucket["count"]  += 1
        bucket["prices"].append(price)
        bucket["ratings"].append(rating)
        bucket["discounts"].append(discount)
        bucket["brands"][brand] += 1
        global_brands[brand] += 1

    # ── Build tree ────────────────────────────────────────────────────────────
    categories_node  = {}
    cat_node_counter = 0

    for cat, subcats in sorted(agg.items()):
        cat_node_counter += 1
        cat_all_prices    = [p for sc in subcats for b in agg[cat][sc] for p in agg[cat][sc][b]["prices"]]
        cat_all_ratings   = [r for sc in subcats for b in agg[cat][sc] for r in agg[cat][sc][b]["ratings"] if r > 0]
        cat_all_brands    = defaultdict(int)
        for sc in subcats:
            for b in agg[cat][sc]:
                for brand, cnt in agg[cat][sc][b]["brands"].items():
                    cat_all_brands[brand] += cnt

        cat_total    = sum(agg[cat][sc][b]["count"] for sc in subcats for b in agg[cat][sc])
        subcat_nodes = {}
        subcat_counter = 0

        for subcat, bands in sorted(subcats.items()):
            subcat_counter += 1
            sc_total   = sum(bands[b]["count"] for b in bands)
            sc_prices  = [p for b in bands for p in bands[b]["prices"]]
            sc_ratings = [r for b in bands for r in bands[b]["ratings"] if r > 0]
            sc_brands  = defaultdict(int)
            for b in bands:
                for brand, cnt in bands[b]["brands"].items():
                    sc_brands[brand] += cnt

            band_nodes = {}
            for band_key, band_label, lo, hi in PRICE_BANDS:
                if band_key not in bands:
                    continue
                bkt = bands[band_key]
                bp  = [x for x in bkt["prices"]  if x > 0]
                br  = [x for x in bkt["ratings"] if x > 0]
                top_brands = sorted(bkt["brands"].items(), key=lambda x: -x[1])[:8]

                band_nodes[band_key] = {
                    "node_id":    f"{cat_node_counter:02d}_{subcat_counter:02d}_{band_key}",
                    "label":      band_label,
                    "price_range": {"min": lo, "max": hi},
                    "count":      bkt["count"],
                    "avg_price":  round(sum(bp) / len(bp), 2) if bp else 0,
                    "avg_rating": round(sum(br) / len(br), 2) if br else 0,
                    "top_brands": [b for b, _ in top_brands],
                }

            top_sc_brands = sorted(sc_brands.items(), key=lambda x: -x[1])[:10]
            avg_sc_rating = round(sum(sc_ratings) / len(sc_ratings), 2) if sc_ratings else 0

            subcat_nodes[subcat] = {
                "node_id":    f"{cat_node_counter:02d}_{subcat_counter:02d}",
                "total":      sc_total,
                "price_range": {
                    "min": round(min(sc_prices), 2) if sc_prices else 0,
                    "max": round(max(sc_prices), 2) if sc_prices else 0,
                    "avg": round(sum(sc_prices) / len(sc_prices), 2) if sc_prices else 0,
                },
                "avg_rating":  avg_sc_rating,
                "top_brands":  [b for b, _ in top_sc_brands],
                "price_bands": band_nodes,
                "summary": (
                    f"{sc_total:,} {subcat} products, "
                    f"price ₹{round(min(sc_prices)):,}–₹{round(max(sc_prices)):,}, "
                    f"avg rating {avg_sc_rating if avg_sc_rating > 0 else 'N/A'}. "
                    f"Top brands: {', '.join([b for b, _ in top_sc_brands[:4]])}."
                ) if sc_prices else f"{sc_total:,} {subcat} products.",
            }

        top_cat_brands = sorted(cat_all_brands.items(), key=lambda x: -x[1])[:10]
        avg_cat_rating = round(sum(cat_all_ratings) / len(cat_all_ratings), 2) if cat_all_ratings else 0

        categories_node[cat] = {
            "node_id":        f"{cat_node_counter:02d}",
            "total":          cat_total,
            "price_range": {
                "min": round(min(cat_all_prices), 2) if cat_all_prices else 0,
                "max": round(max(cat_all_prices), 2) if cat_all_prices else 0,
            },
            "avg_rating":     avg_cat_rating,
            "top_brands":     [b for b, _ in top_cat_brands],
            "sub_categories": subcat_nodes,
            "summary": (
                f"{cat_total:,} products across {len(subcats)} sub-categories. "
                f"Price range ₹{round(min(cat_all_prices)):,}–₹{round(max(cat_all_prices)):,}. "
                f"Top brands: {', '.join([b for b, _ in top_cat_brands[:4]])}."
            ) if cat_all_prices else f"{cat_total:,} products.",
        }

    top_global_brands = sorted(global_brands.items(), key=lambda x: -x[1])[:20]

    catalog_tree = {
        "schema_version":    "1.0",
        "total_products":    total_products,
        "total_categories":  len(categories_node),
        "top_brands":        [b for b, _ in top_global_brands],
        "categories":        categories_node,
        "price_band_definitions": [
            {"key": k, "label": l, "min": lo, "max": hi}
            for k, l, lo, hi in PRICE_BANDS
        ],
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(catalog_tree, f, indent=2, ensure_ascii=False)

    print(f"[IndexBuilder] Catalog tree saved → {output_path}")
    print(f"[IndexBuilder] Categories: {len(categories_node)}")
    for cat, node in categories_node.items():
        avg_r = node["avg_rating"]
        print(f"  {cat}: {node['total']:,} products, avg_rating={avg_r}")

    return catalog_tree


def load_catalog_tree(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)