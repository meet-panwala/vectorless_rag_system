import json
import re
from pathlib import Path
from typing import Optional


# ── Field parsers ─────────────────────────────────────────────────────────────

def _to_float(val, default: float = 0.0) -> float:
    """Parse price strings like '2,999' or '₹921' → float."""
    if val is None:
        return default
    s = str(val).replace("₹", "").replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return default


def _to_int(val, default: int = 0) -> int:
    try:
        s = str(val).replace(",", "").strip()
        return int(float(s))
    except (ValueError, TypeError):
        return default


def _parse_discount(val) -> float:
    """
    Parse discount field.
    Handles: "69% off" → 69.0
             69        → 69.0
             "69"      → 69.0
    """
    if val is None:
        return 0.0
    s = str(val).strip()
    # Extract first number from string like "69% off"
    m = re.search(r"([\d.]+)", s)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return 0.0
    return 0.0


def _parse_rating(val) -> float:
    """Parse rating: "3.9" → 3.9, "3.9 out of 5" → 3.9"""
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


def _parse_image(val) -> str:
    """Handle image as string or list — return first URL."""
    if not val:
        return ""
    if isinstance(val, list):
        return str(val[0]).strip() if val else ""
    return str(val).strip()


# ── DataStore ─────────────────────────────────────────────────────────────────

class DataStore:
    """
    Loads the Flipkart JSON dataset into memory and provides
    structured query methods used by catalog_tools.py.
    """

    def __init__(self, json_path: str):
        self.json_path = json_path
        self.products: list[dict] = []
        self._load()

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load(self):
        print(f"[DataStore] Loading products from {self.json_path} …")
        with open(self.json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if isinstance(raw, list):
            raw_products = raw
        elif isinstance(raw, dict):
            for key in ("products", "data", "items", "records"):
                if key in raw:
                    raw_products = raw[key]
                    break
            else:
                raw_products = []
        else:
            raw_products = []

        for idx, p in enumerate(raw_products):
            self.products.append(self._normalise(p, idx))

        print(f"[DataStore] {len(self.products):,} products ready.")

        # Debug: show first product's rating to confirm parsing works
        if self.products:
            sample = self.products[0]
            print(f"[DataStore] Sample — name: '{sample['product_name'][:40]}', "
                  f"price: ₹{sample['discounted_price']}, "
                  f"rating: {sample['rating']}, "
                  f"discount: {sample['discount_percentage']}%")

    def _normalise(self, p: dict, idx: int) -> dict:
        """
        Standardise field names from the actual Flipkart JSON structure.

        Actual JSON fields used:
          title, selling_price, actual_price, discount,
          average_rating, url, images (list), brand,
          category, sub_category, description, pid
        """
        return {
            # identity
            "pid":          str(p.get("pid", p.get("_id", idx))),

            # name — JSON uses 'title', fallback to 'product_name'
            "product_name": str(p.get("title", p.get("product_name", ""))).strip(),

            # description
            "description":  str(p.get("description", "")).strip(),

            # brand / category
            "brand":        str(p.get("brand", "Unknown")).strip() or "Unknown",
            "category":     str(p.get("category", "Unknown")).strip(),
            "sub_category": str(p.get("sub_category", "General")).strip(),

            # price — JSON uses 'selling_price' (string with commas)
            "discounted_price": _to_float(p.get("selling_price",
                                p.get("discounted_price", 0))),
            "actual_price":     _to_float(p.get("actual_price", 0)),

            # discount — JSON stores as "69% off" string
            "discount_percentage": _parse_discount(p.get("discount",
                                   p.get("discount_percentage", 0))),

            # rating — JSON uses 'average_rating' string "3.9"
            "rating":       _parse_rating(p.get("average_rating",
                            p.get("rating", 0))),

            # rating count — JSON doesn't have this, default 0
            "rating_count": _to_int(p.get("rating_count",
                            p.get("no_of_ratings", 0))),

            # media — JSON uses 'images' (list) or 'url'
            "image":         _parse_image(p.get("images", p.get("image", ""))),
            "product_link":  str(p.get("url", p.get("product_link", ""))).strip(),
        }

    # ── Public query API ──────────────────────────────────────────────────────

    def filter_products(
        self,
        category:       Optional[str]   = None,
        sub_category:   Optional[str]   = None,
        brand:          Optional[str]   = None,
        min_price:      Optional[float] = None,
        max_price:      Optional[float] = None,
        min_rating:     Optional[float] = None,
        max_rating:     Optional[float] = None,
        min_discount:   Optional[float] = None,
        sort_by:        str = "rating",
        limit:          int = 10,
        keyword:        Optional[str]   = None,
    ) -> list[dict]:
        results = self.products

        if category:
            cat_lower = category.lower()
            results = [p for p in results if cat_lower in p["category"].lower()]

        if sub_category:
            sc_lower = sub_category.lower()
            results = [p for p in results if sc_lower in p["sub_category"].lower()]

        if brand:
            brand_lower = brand.lower()
            results = [p for p in results if brand_lower in p["brand"].lower()]

        if min_price is not None:
            results = [p for p in results if p["discounted_price"] >= min_price]
        if max_price is not None:
            results = [p for p in results if p["discounted_price"] <= max_price]

        if min_rating is not None:
            results = [p for p in results if p["rating"] >= min_rating]
        if max_rating is not None:
            results = [p for p in results if p["rating"] <= max_rating]

        if min_discount is not None:
            results = [p for p in results if p["discount_percentage"] >= min_discount]

        if keyword:
            kw = keyword.lower()
            results = [
                p for p in results
                if kw in p["product_name"].lower()
                or kw in p["description"].lower()
                or kw in p["brand"].lower()
            ]

        sort_key, sort_rev = self._sort_params(sort_by)
        results = sorted(results, key=sort_key, reverse=sort_rev)

        limit = min(max(1, limit), 50)
        return [self._format(p) for p in results[:limit]]

    def get_product_by_id(self, pid: str) -> Optional[dict]:
        for p in self.products:
            if p["pid"] == pid:
                return self._format(p)
        return None

    def search_by_name(self, query: str, limit: int = 10) -> list[dict]:
        words = query.lower().split()
        scored = []
        for p in self.products:
            text = f"{p['product_name']} {p['brand']} {p['description']}".lower()
            score = sum(1 for w in words if w in text)
            if score > 0:
                scored.append((score, p))
        scored.sort(key=lambda x: (-x[0], -x[1]["rating"]))
        return [self._format(p) for _, p in scored[:limit]]

    def get_brands_in_category(self, category: str, sub_category: Optional[str] = None) -> list[str]:
        results = [p for p in self.products if category.lower() in p["category"].lower()]
        if sub_category:
            results = [p for p in results if sub_category.lower() in p["sub_category"].lower()]
        brands: dict[str, int] = {}
        for p in results:
            brands[p["brand"]] = brands.get(p["brand"], 0) + 1
        return [b for b, _ in sorted(brands.items(), key=lambda x: -x[1])]

    def get_price_stats(self, category: str, sub_category: Optional[str] = None) -> dict:
        results = [p for p in self.products if category.lower() in p["category"].lower()]
        if sub_category:
            results = [p for p in results if sub_category.lower() in p["sub_category"].lower()]
        prices = [p["discounted_price"] for p in results if p["discounted_price"] > 0]
        if not prices:
            return {}
        return {
            "count":  len(prices),
            "min":    round(min(prices), 2),
            "max":    round(max(prices), 2),
            "avg":    round(sum(prices) / len(prices), 2),
            "median": round(sorted(prices)[len(prices) // 2], 2),
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _sort_params(sort_by: str):
        mapping = {
            "rating":       (lambda p: p["rating"],              True),
            "price_asc":    (lambda p: p["discounted_price"],    False),
            "price_desc":   (lambda p: p["discounted_price"],    True),
            "discount":     (lambda p: p["discount_percentage"], True),
            "rating_count": (lambda p: p["rating_count"],        True),
        }
        return mapping.get(sort_by, mapping["rating"])

    @staticmethod
    def _format(p: dict) -> dict:
        """Return a clean, human-readable product dict for the LLM."""
        price    = p["discounted_price"]
        actual   = p["actual_price"]
        discount = p["discount_percentage"]
        rating   = p["rating"]

        price_str    = f"₹{price:,.0f}"    if price > 0    else "N/A"
        actual_str   = f"₹{actual:,.0f}"   if actual > 0   else ""
        discount_str = f"{discount:.0f}% off" if discount > 0 else ""
        rating_str   = f"{rating:.1f}/5"   if rating > 0   else "No rating"

        return {
            "pid":          p["pid"],
            "name":         p["product_name"],
            "brand":        p["brand"],
            "category":     p["category"],
            "sub_category": p["sub_category"],
            "price":        price_str,
            "actual_price": actual_str,
            "discount":     discount_str,
            "rating":       rating_str,
            "rating_count": f"{p['rating_count']:,} reviews" if p["rating_count"] > 0 else "",
            "description":  p["description"][:300] + "…"
                            if len(p["description"]) > 300 else p["description"],
            "link":         p["product_link"],
            "image":        p["image"],
        }


# ── Singleton loader ──────────────────────────────────────────────────────────

_store_instance: Optional[DataStore] = None


def get_store(json_path: str) -> DataStore:
    global _store_instance
    if _store_instance is None:
        _store_instance = DataStore(json_path)
    return _store_instance