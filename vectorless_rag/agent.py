"""
agent.py  —  Final working version
------------------------------------
Fix in this version:
  ✅ Off-topic query guard:
       - Keyword-based fast check runs BEFORE any tool call
       - If query is clearly unrelated to fashion/products → instant refusal
       - No API call wasted, no hallucination possible
       - Polite, helpful refusal message shown
  ✅ All previous fixes retained
"""

import json
import os
import re
import time
from groq import Groq
from .catalog_tools import TOOL_SCHEMAS, ToolExecutor
from .data_store import DataStore
from .index_builder import load_catalog_tree

# ─── Constants ────────────────────────────────────────────────────────────────
MAX_TOOL_RESULT_CHARS = 3_500
MAX_HISTORY_PAIRS     = 4
MAX_TURN_CHARS        = 500
MAX_TOKENS_RESPONSE   = 1024
MAX_RETRIES_429       = 3
MAX_STEPS             = 12
FORCE_ANSWER_AFTER    = 3

# ─── Off-topic refusal message ────────────────────────────────────────────────
REFUSAL_MESSAGE = (
    "I'm sorry, I don't have information about that. 🙏\n\n"
    "I am a **Fashion Shopping AI Assistant** and I only have knowledge about:\n"
    "- 👗 Clothing & Accessories (kurtas, sarees, shirts, dresses, jeans, etc.)\n"
    "- 👟 Footwear (sneakers, sandals, formal shoes, etc.)\n"
    "- 👜 Bags, Wallets & Belts\n"
    "- 🛍️ Product prices, ratings, brands, and discounts\n\n"
    "Please ask me something related to fashion products and I'll be happy to help!"
)

# ─── Fashion-related keywords for relevance check ─────────────────────────────
# If the query contains ANY of these → it's relevant → proceed normally
FASHION_KEYWORDS = {
    # product types
    "shirt", "kurta", "saree", "dress", "jeans", "trouser", "pant", "jacket",
    "coat", "blazer", "suit", "waistcoat", "lehenga", "salwar", "dupatta",
    "kurti", "top", "tshirt", "t-shirt", "sweatshirt", "hoodie", "sweater",
    "cardigan", "shrug", "skirt", "shorts", "trackpant", "jogger", "chino",
    "ethnic", "western", "formal", "casual", "partywear", "party wear",
    "innerwear", "lingerie", "bra", "brief", "boxer", "sock", "stocking",
    "raincoat", "windcheater", "tracksuit", "sportswear", "activewear",
    "sleepwear", "nightwear", "pyjama", "kimono", "fabric", "cloth",
    # footwear
    "shoe", "sandal", "slipper", "sneaker", "boot", "heel", "flat", "loafer",
    "moccasin", "flip flop", "flipflop", "chappal", "footwear", "running shoe",
    # accessories & bags
    "bag", "wallet", "belt", "handbag", "purse", "clutch", "backpack",
    "sling", "tote", "watch", "sunglasses", "cap", "hat", "scarf", "stole",
    "jewellery", "jewelry", "necklace", "earring", "bracelet", "ring",
    # shopping terms
    "brand", "price", "rating", "discount", "offer", "cheap", "budget",
    "premium", "buy", "purchase", "shop", "product", "item", "collection",
    "category", "categories", "filter", "sort", "compare", "recommend",
    "best", "top", "popular", "trending", "sale", "deal", "review",
    # specific brands in dataset
    "libas", "biba", "myntra", "flipkart", "fastrack", "van heusen",
    "louis philippe", "allen solly", "peter england", "park avenue",
    "levis", "levi", "puma", "reebok", "adidas", "nike", "wildcraft",
    # colours & styles (product-related)
    "printed", "solid", "striped", "checked", "floral", "embroidered",
    "slim fit", "regular fit", "oversized",
}

# If the query contains ANY of these → clearly off-topic → instant refusal
OFFTOPIC_KEYWORDS = {
    "actor", "actress", "celebrity", "movie", "film", "cricket", "sport",
    "politics", "politician", "president", "prime minister", "minister",
    "science", "history", "geography", "capital", "country", "currency",
    "recipe", "cook", "food", "restaurant", "doctor", "hospital", "medicine",
    "weather", "news", "stock", "share market", "crypto", "bitcoin",
    "programming", "code", "software", "hardware", "phone", "laptop",
    "born", "died", "age", "biography", "who is", "what is",
    "where is", "when was", "how to", "why does", "explain",
    "define", "meaning of", "translate",
}

# ─── System prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a Flipkart Fashion Shopping Assistant with access to 30,000+ fashion products.

YOUR SCOPE — you ONLY answer questions about:
- Fashion products: clothing, footwear, bags, wallets, accessories
- Product prices, ratings, brands, discounts, categories
- Product recommendations, comparisons, filtering

OUT OF SCOPE — if the user asks about anything else (people, movies, politics,
food, science, general knowledge, etc.), respond ONLY with:
"I'm sorry, I don't have information about that. I can only help with fashion products and shopping queries."

TOOL USAGE:
1. Call get_catalog_structure ONCE to see categories.
2. Call filter_products OR search_products to get actual products.
   - If sub-category not found, use search_products with keyword.
   - If filter_products returns 0 results, try search_products.
3. STOP calling tools once you have products and write your final answer.

CRITICAL RULES:
- NEVER invent or make up product names, prices, or ratings.
- ONLY use products that appear in tool results.
- If no products found after searching, say so honestly.
- Never call the same tool with the same arguments twice.

ANSWER FORMAT:
- Numbered list: Name | Price (₹) | Brand
- Include rating only if it is not "No rating".
- Max 5–8 products unless user asks for more.

SORTING:
- "best/top rated" → sort_by=rating
- "cheapest/under ₹X" → sort_by=price_asc + max_price filter
- "most reviewed" → sort_by=rating_count
- "sorted by price" → sort_by=price_asc
- "most discount" → sort_by=discount
"""

FORCE_ANSWER_MSG = {
    "role":    "user",
    "content": (
        "You have called enough tools. "
        "Now write your final answer using ONLY the product data returned by the tools. "
        "Do not add any products not present in the tool results. "
        "If the tools returned 0 products, say so honestly."
    ),
}

DEFAULT_MODEL = "llama-3.3-70b-versatile"


# ─── CatalogAgent ─────────────────────────────────────────────────────────────
class CatalogAgent:
    def __init__(
        self,
        data_store:     DataStore,
        catalog_tree:   dict,
        model:          str   = DEFAULT_MODEL,
        max_tool_calls: int   = MAX_STEPS,
        temperature:    float = 0.3,
    ):
        self.client      = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.executor    = ToolExecutor(data_store, catalog_tree)
        self.model       = model or DEFAULT_MODEL
        self.max_steps   = max_tool_calls
        self.temperature = temperature

    # ─── Public API ───────────────────────────────────────────────────────────
    def get_tool_trace(self, user_message: str, history: list = None) -> dict:
        """
        Main entry point. Runs relevance check first,
        then the full agentic tool loop.
        """
        # ── Step 0: Off-topic guard ───────────────────────────────────────────
        if self._is_offtopic(user_message):
            return {
                "final_answer": REFUSAL_MESSAGE,
                "steps":        0,
                "trace":        [],
            }

        # ── Step 1: Normal agentic loop ───────────────────────────────────────
        messages        = self._build_messages(user_message, history)
        trace           = []
        steps_used      = 0
        tool_calls_made = 0
        force_injected  = False
        products_found  = 0

        while steps_used < self.max_steps:

            if tool_calls_made >= FORCE_ANSWER_AFTER and not force_injected:
                messages.append(FORCE_ANSWER_MSG)
                force_injected = True

            response = self._call_groq_with_retry(messages)

            if isinstance(response, str):
                return {"final_answer": response, "steps": steps_used, "trace": trace}

            msg = response.choices[0].message
            steps_used += 1

            if not getattr(msg, "tool_calls", None):
                answer = msg.content or ""
                if not answer.strip():
                    answer = self._safe_final_answer(messages, trace, products_found, user_message)
                return {
                    "final_answer": answer or "No results found.",
                    "steps":        steps_used,
                    "trace":        trace,
                }

            assistant_tool_calls = []
            tool_result_turns    = []

            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                raw_result  = self.executor.execute(tc.function.name, args)
                safe_result = self._trim_result(raw_result)
                tool_calls_made += 1

                try:
                    result_obj = json.loads(raw_result)
                    if isinstance(result_obj, dict):
                        p = result_obj.get("products", [])
                        if isinstance(p, list):
                            products_found += len(p)
                except Exception:
                    result_obj = raw_result

                trace.append({
                    "step":   steps_used,
                    "tool":   tc.function.name,
                    "args":   args,
                    "result": result_obj,
                })

                assistant_tool_calls.append({
                    "id":   tc.id,
                    "type": "function",
                    "function": {
                        "name":      tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                })
                tool_result_turns.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      safe_result,
                })

            messages.append({
                "role":       "assistant",
                "content":    msg.content or "",
                "tool_calls": assistant_tool_calls,
            })
            messages.extend(tool_result_turns)

        final = self._safe_final_answer(messages, trace, products_found, user_message)
        return {"final_answer": final, "steps": steps_used, "trace": trace}

    # ─── Off-topic relevance check ────────────────────────────────────────────
    @staticmethod
    def _is_offtopic(query: str) -> bool:
        """
        Returns True if the query is clearly NOT about fashion/products.

        Logic:
          1. If query contains any FASHION keyword → relevant → False
          2. If query contains any OFFTOPIC keyword → not relevant → True
          3. If very short and no fashion context → True
          4. Otherwise → assume relevant → False (safe default)
        """
        q = query.lower().strip()

        # Always relevant if it contains a fashion keyword
        for kw in FASHION_KEYWORDS:
            if kw in q:
                return False

        # Clearly off-topic if it contains an off-topic keyword
        for kw in OFFTOPIC_KEYWORDS:
            if kw in q:
                return True

        # Very short queries with no fashion context
        # e.g. "who is xyz", "what is abc" without product context
        if len(q.split()) <= 4 and not any(kw in q for kw in FASHION_KEYWORDS):
            # Check if it starts with question words about non-product things
            offtopic_starts = ("who is", "who was", "what is", "what was",
                               "where is", "when was", "when is", "why is",
                               "how old", "tell me about")
            if any(q.startswith(s) for s in offtopic_starts):
                return True

        return False  # default: assume relevant

    # ─── Safe final answer ────────────────────────────────────────────────────
    def _safe_final_answer(
        self,
        messages:       list,
        trace:          list,
        products_found: int,
        user_message:   str,
    ) -> str:
        if products_found == 0:
            query_subject = user_message.strip().rstrip("?").lower()
            return (
                f"I searched the catalog but could not find any products matching "
                f"**\"{query_subject}\"**.\n\n"
                f"This could be because:\n"
                f"- The sub-category name is different in the catalog\n"
                f"  (e.g. Lehenga Cholis are listed under **Kurtas, Ethnic Sets and Bottoms**)\n"
                f"- Try rephrasing with a broader keyword like **'ethnic wear'**, **'kurta'**, **'saree'**\n"
                f"- Use **🔍 Show reasoning trace** to see what categories exist"
            )

        try:
            forced_messages = list(messages) + [{
                "role":    "user",
                "content": (
                    f"The tools retrieved {products_found} product(s). "
                    "Write a clear final answer using ONLY those products. "
                    "List each with: Name | Price (₹) | Brand. "
                    "Do NOT add any products not in the tool results."
                ),
            }]
            response = self.client.chat.completions.create(
                model       = self.model,
                messages    = forced_messages,
                tool_choice = "none",
                temperature = self.temperature,
                max_tokens  = MAX_TOKENS_RESPONSE,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            return f"⚠️ Could not generate final answer: {e}"

    # ─── Groq call with retry ─────────────────────────────────────────────────
    def _call_groq_with_retry(self, messages: list):
        local_messages = list(messages)

        for attempt in range(MAX_RETRIES_429 + 1):
            try:
                return self.client.chat.completions.create(
                    model       = self.model,
                    messages    = local_messages,
                    tools       = TOOL_SCHEMAS,
                    tool_choice = "auto",
                    temperature = self.temperature,
                    max_tokens  = MAX_TOKENS_RESPONSE,
                )
            except Exception as e:
                err = str(e)

                if "429" in err or "rate_limit" in err.lower():
                    wait = self._parse_retry_after(err)
                    if attempt < MAX_RETRIES_429:
                        time.sleep(wait)
                        continue
                    return (
                        f"⏳ Rate limit reached (429). "
                        f"Please wait ~{wait:.0f}s and try again, "
                        f"or switch to **llama-3.3-70b-versatile** in the sidebar."
                    )

                if "413" in err or "too large" in err.lower() or "payload" in err.lower():
                    non_sys = [i for i, m in enumerate(local_messages) if m["role"] != "system"]
                    if non_sys:
                        local_messages.pop(non_sys[0])
                        continue
                    return (
                        "⚠️ Context too large (413). "
                        "Please press **🗑️ Clear Chat** and ask a shorter question."
                    )

                if "400" in err:
                    return (
                        "⚠️ Bad request (400). "
                        "Please press **🗑️ Clear Chat** and try again."
                    )

                return f"⚠️ Error: {err}"

        return "⚠️ All retries exhausted. Please try again in a moment."

    # ─── Message builder ──────────────────────────────────────────────────────
    def _build_messages(self, user_message: str, history: list = None) -> list:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:
            clean = [
                h for h in history
                if h.get("role") in ("user", "assistant")
                and h.get("content", "").strip()
            ]
            for turn in clean[-(MAX_HISTORY_PAIRS * 2):]:
                messages.append({
                    "role":    turn["role"],
                    "content": self._trim_str(turn.get("content", ""), MAX_TURN_CHARS),
                })
        messages.append({"role": "user", "content": user_message})
        return messages

    # ─── Tool result trimmer ──────────────────────────────────────────────────
    @staticmethod
    def _trim_result(raw: str) -> str:
        if len(raw) <= MAX_TOOL_RESULT_CHARS:
            return raw
        try:
            obj = json.loads(raw)
            if not isinstance(obj, dict):
                raise ValueError

            if "products" in obj and isinstance(obj["products"], list):
                obj["products"] = obj["products"][:4]
                obj.pop("filters", None)
                s = json.dumps(obj, ensure_ascii=False)
                if len(s) <= MAX_TOOL_RESULT_CHARS:
                    return s

            if "categories" in obj and isinstance(obj["categories"], dict):
                slim = {}
                for cat, cdata in obj["categories"].items():
                    slim[cat] = {
                        "total":          cdata.get("total", 0),
                        "sub_categories": (
                            list(cdata["sub_categories"].keys())
                            if isinstance(cdata.get("sub_categories"), dict)
                            else cdata.get("sub_categories", [])
                        ),
                        "top_brands": cdata.get("top_brands", [])[:4],
                    }
                obj["categories"] = slim
                obj.pop("global_top_brands", None)
                s = json.dumps(obj, ensure_ascii=False)
                if len(s) <= MAX_TOOL_RESULT_CHARS:
                    return s

            if "segments" in obj and isinstance(obj["segments"], dict):
                for seg in obj["segments"]:
                    if isinstance(obj["segments"][seg], list):
                        obj["segments"][seg] = obj["segments"][seg][:1]
                s = json.dumps(obj, ensure_ascii=False)
                if len(s) <= MAX_TOOL_RESULT_CHARS:
                    return s

            if "all_brands" in obj:
                obj["all_brands"] = obj["all_brands"][:10]
                s = json.dumps(obj, ensure_ascii=False)
                if len(s) <= MAX_TOOL_RESULT_CHARS:
                    return s

        except Exception:
            pass

        return raw[:MAX_TOOL_RESULT_CHARS] + " ...[truncated]"

    # ─── Helpers ──────────────────────────────────────────────────────────────
    @staticmethod
    def _parse_retry_after(err_str: str) -> float:
        m = re.search(r"try again in\s+([\d.]+)s", err_str, re.IGNORECASE)
        if m:
            return float(m.group(1)) + 1.0
        return 12.0

    @staticmethod
    def _trim_str(text: str, limit: int) -> str:
        return text[:limit] + "…" if len(text) > limit else text


# ─── Singleton factory ────────────────────────────────────────────────────────
_agent_instance: CatalogAgent | None = None
_last_model:     str | None          = None


def get_agent(
    json_path: str,
    tree_path: str,
    model:     str = DEFAULT_MODEL,
) -> CatalogAgent:
    global _agent_instance, _last_model
    model = model or DEFAULT_MODEL
    if _agent_instance is None or _last_model != model:
        from .data_store import get_store
        store = get_store(json_path)
        tree  = load_catalog_tree(tree_path)
        _agent_instance = CatalogAgent(store, tree, model=model)
        _last_model     = model
    return _agent_instance