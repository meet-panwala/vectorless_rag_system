"""
catalog_tools.py
----------------
Defines the 5 tools available to the Groq LLM agent + their executors.

413-prevention changes vs original:
  - get_catalog_structure() returns a COMPACT summary only:
      category → { total, sub_categories: [name, name, ...], top_brands[:5] }
    NOT the full price-band breakdown. That level of detail lives in
    get_subcategory_details() which is only called when needed.
  - filter_products() caps limit at 8 (not 20) to keep result payloads small.
  - compare_products() caps at 2 products per segment.
  - Product dicts are slimmed: description capped at 150 chars.
"""

import json
from typing import Optional
from .data_store import DataStore


# ── Tool schemas (Groq / OpenAI function-calling format) ─────────────────────
# Descriptions are intentionally concise — verbose descriptions burn tokens.

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_catalog_structure",
            "description": (
                "Returns a compact catalog overview: categories, sub-category names, "
                "product counts, top brands. Call this FIRST to understand what is available."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_subcategory_details",
            "description": (
                "Returns detailed stats for one sub-category: price bands, all brands, "
                "price range, avg rating. Call after get_catalog_structure to drill in."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "category":     {"type": "string", "description": "Parent category (partial match ok)."},
                    "sub_category": {"type": "string", "description": "Sub-category name (partial match ok)."},
                },
                "required": ["category", "sub_category"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "filter_products",
            "description": (
                "Retrieve products with optional filters. Returns up to 8 products. "
                "All params optional — only pass what is relevant."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "category":     {"type": "string",  "description": "Filter by category (partial ok)."},
                    "sub_category": {"type": "string",  "description": "Filter by sub-category (partial ok)."},
                    "brand":        {"type": "string",  "description": "Filter by brand (partial ok)."},
                    "min_price":    {"type": "number",  "description": "Min price in ₹."},
                    "max_price":    {"type": "number",  "description": "Max price in ₹."},
                    "min_rating":   {"type": "number",  "description": "Min rating (0-5)."},
                    "min_discount": {"type": "number",  "description": "Min discount %."},
                    "keyword":      {"type": "string",  "description": "Search keyword in name/brand/description."},
                    "sort_by": {
                        "type": "string",
                        "enum": ["rating", "price_asc", "price_desc", "discount", "rating_count"],
                        "description": "Sort: rating (default), price_asc, price_desc, discount, rating_count.",
                    },
                    "limit": {"type": "integer", "description": "Results to return (max 8, default 6)."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": (
                "Full-text search by product name, style, or occasion. "
                "Use when user mentions a specific keyword like 'formal shirt' or 'party dress'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search terms."},
                    "limit": {"type": "integer", "description": "Max results (default 6, max 8)."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_products",
            "description": (
                "Compare products in a sub-category across brands or price bands. "
                "Use for 'compare', 'which is better', 'difference between' queries."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sub_category": {"type": "string", "description": "Sub-category to compare within."},
                    "brands": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Brand names to compare. Omit to compare by price band.",
                    },
                    "sort_by": {
                        "type": "string",
                        "enum": ["rating", "price_asc", "price_desc", "discount"],
                    },
                },
                "required": ["sub_category"],
            },
        },
    },
]


# ── ToolExecutor ──────────────────────────────────────────────────────────────

class ToolExecutor:
    """Executes tool calls using DataStore + catalog tree."""

    def __init__(self, data_store: DataStore, catalog_tree: dict):
        self.store = data_store
        self.tree  = catalog_tree

    def execute(self, tool_name: str, tool_args: dict) -> str:
        """Route and execute a tool call. Returns JSON string."""
        handlers = {
            "get_catalog_structure":   self._get_catalog_structure,
            "get_subcategory_details": self._get_subcategory_details,
            "filter_products":         self._filter_products,
            "search_products":         self._search_products,
            "compare_products":        self._compare_products,
        }
        handler = handlers.get(tool_name)
        if not handler:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
        try:
            result = handler(**tool_args)
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e)})

    # ── Tool implementations ──────────────────────────────────────────────────

    def _get_catalog_structure(self) -> dict:
        """
        Returns a COMPACT catalog overview — NOT the full tree.

        Only includes:
          - total_products
          - top 10 global brands
          - per category: total + list of sub-category names + 5 top brands

        This keeps the response under ~2KB instead of ~50KB.
        The LLM can call get_subcategory_details for deeper info.
        """
        compact = {
            "total_products":    self.tree.get("total_products", 0),
            "global_top_brands": self.tree.get("top_brands", [])[:10],
            "categories":        {},
        }
        for cat, cnode in self.tree.get("categories", {}).items():
            compact["categories"][cat] = {
                "total":           cnode.get("total", 0),
                "summary":         cnode.get("summary", ""),
                "sub_categories":  list(cnode.get("sub_categories", {}).keys()),
                "top_brands":      cnode.get("top_brands", [])[:5],
            }
        return compact

    def _get_subcategory_details(self, category: str, sub_category: str) -> dict:
        """Detailed stats for one sub-category (brands, price bands, price range)."""
        cat_key = self._find_key(self.tree.get("categories", {}), category)
        if not cat_key:
            avail = list(self.tree.get("categories", {}).keys())
            return {"error": f"Category '{category}' not found.", "available": avail}

        cnode  = self.tree["categories"][cat_key]
        sc_key = self._find_key(cnode.get("sub_categories", {}), sub_category)
        if not sc_key:
            avail = list(cnode.get("sub_categories", {}).keys())
            return {"error": f"Sub-category '{sub_category}' not found in '{cat_key}'.", "available": avail}

        snode  = cnode["sub_categories"][sc_key]
        brands = self.store.get_brands_in_category(cat_key, sc_key)
        stats  = self.store.get_price_stats(cat_key, sc_key)

        return {
            "category":       cat_key,
            "sub_category":   sc_key,
            "total_products": snode.get("total", 0),
            "price_range":    snode.get("price_range", {}),
            "avg_rating":     snode.get("avg_rating", 0),
            "price_stats":    stats,
            "top_brands":     brands[:20],
            "price_bands":    {
                band: {
                    "count":      bnode.get("count", 0),
                    "avg_price":  bnode.get("avg_price", 0),
                    "avg_rating": bnode.get("avg_rating", 0),
                    "top_brands": bnode.get("top_brands", [])[:4],
                }
                for band, bnode in snode.get("price_bands", {}).items()
            },
            "summary": snode.get("summary", ""),
        }

    def _filter_products(
        self,
        category:     str   = None,
        sub_category: str   = None,
        brand:        str   = None,
        min_price:    float = None,
        max_price:    float = None,
        min_rating:   float = None,
        min_discount: float = None,
        keyword:      str   = None,
        sort_by:      str   = "rating",
        limit:        int   = 6,
    ) -> dict:
        # Hard cap at 8 to keep tool result payload small
        safe_limit = min(int(limit), 8)
        products = self.store.filter_products(
            category=category,
            sub_category=sub_category,
            brand=brand,
            min_price=min_price,
            max_price=max_price,
            min_rating=min_rating,
            min_discount=min_discount,
            keyword=keyword,
            sort_by=sort_by,
            limit=safe_limit,
        )
        # Slim each product for smaller payload
        slim = [self._slim_product(p) for p in products]
        return {
            "count":    len(slim),
            "filters":  {k: v for k, v in {
                "category": category, "sub_category": sub_category,
                "brand": brand, "min_price": min_price, "max_price": max_price,
                "min_rating": min_rating, "sort_by": sort_by,
            }.items() if v is not None},
            "products": slim,
        }

    def _search_products(self, query: str, limit: int = 6) -> dict:
        safe_limit = min(int(limit), 8)
        products = self.store.search_by_name(query, limit=safe_limit)
        slim = [self._slim_product(p) for p in products]
        return {"query": query, "count": len(slim), "products": slim}

    def _compare_products(
        self,
        sub_category: str,
        brands:       list = None,
        sort_by:      str  = "rating",
    ) -> dict:
        if brands:
            segments = {}
            for brand in brands:
                prods = self.store.filter_products(
                    sub_category=sub_category, brand=brand,
                    sort_by=sort_by, limit=2,          # 2 per brand keeps payload tiny
                )
                segments[brand] = [self._slim_product(p) for p in prods]
            return {"sub_category": sub_category, "comparison_by": "brand", "segments": segments}

        # Compare by price band
        segments = {}
        for band_key, band_label, lo, hi in [
            ("budget",  "Budget (Under ₹500)",     0,    499),
            ("mid",     "Mid (₹500–₹1,500)",      500, 1499),
            ("premium", "Premium (Above ₹1,500)", 1500, 999999),
        ]:
            prods = self.store.filter_products(
                sub_category=sub_category,
                min_price=lo if lo > 0 else None,
                max_price=hi if hi < 999999 else None,
                sort_by=sort_by, limit=2,
            )
            if prods:
                segments[band_label] = [self._slim_product(p) for p in prods]

        return {"sub_category": sub_category, "comparison_by": "price_band", "segments": segments}

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _slim_product(p: dict) -> dict:
        """
        Return only essential fields for LLM consumption.
        Caps description at 150 chars to prevent bloat.
        """
        return {
            "name":     p.get("name", ""),
            "brand":    p.get("brand", ""),
            "price":    p.get("price", ""),
            "discount": p.get("discount", ""),
            "rating":   p.get("rating", ""),
            "reviews":  p.get("rating_count", ""),
            "desc":     (p.get("description", "")[:150] + "…")
                        if len(p.get("description", "")) > 150
                        else p.get("description", ""),
        }

    @staticmethod
    def _find_key(d: dict, query: str) -> Optional[str]:
        """Case-insensitive partial-match key lookup."""
        q = query.lower()
        for k in d:
            if k.lower() == q:
                return k
        for k in d:
            if q in k.lower() or k.lower() in q:
                return k
        return None