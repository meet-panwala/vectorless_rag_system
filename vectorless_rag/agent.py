import json
import os
import re
import time
from groq import Groq
from .catalog_tools import TOOL_SCHEMAS, ToolExecutor
from .data_store import DataStore
from .index_builder import load_catalog_tree


MAX_TOOL_RESULT_CHARS = 3_500
MAX_HISTORY_PAIRS     = 4
MAX_TURN_CHARS        = 500
MAX_TOKENS_RESPONSE   = 1024
MAX_RETRIES_429       = 3
MAX_STEPS             = 12
FORCE_ANSWER_AFTER    = 3


SYSTEM_PROMPT = """You are a Flipkart fashion shopping assistant with 30,000+ products.

TOOL USAGE:
1. Call get_catalog_structure ONCE to see categories.
2. Optionally call get_subcategory_details ONCE for details.
3. Call filter_products OR search_products to get products.
4. STOP calling tools and write your final answer immediately.

ANSWER FORMAT:
- Numbered list: Name | Price (₹) | Rating | Brand
- Max 5–8 products unless user asks for more.
- Never call the same tool twice with the same arguments.
- If filter_products returned results, DO NOT call any more tools — answer now.

SORTING:
- "best" / "top" → sort_by=rating
- "cheapest" / "under ₹X" → sort_by=price_asc + max_price filter
- "most reviewed" → sort_by=rating_count
- "most discount" → sort_by=discount
"""

FORCE_ANSWER_MSG = {
    "role":    "user",
    "content": (
        "You now have enough product data from the tools. "
        "STOP calling tools and write your final answer now. "
        "List the products with name, price, and rating."
    ),
}

SUPPORTED_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant"
]
DEFAULT_MODEL = "llama-3.3-70b-versatile"


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
        messages      = self._build_messages(user_message, history)
        trace         = []
        steps_used    = 0
        tool_calls_made = 0          # counts total individual tool calls across all steps
        force_injected  = False      # only inject force-answer once

        while steps_used < self.max_steps:

            # ── Inject force-answer after FORCE_ANSWER_AFTER tool calls ───
            if tool_calls_made >= FORCE_ANSWER_AFTER and not force_injected:
                messages.append(FORCE_ANSWER_MSG)
                force_injected = True

            response = self._call_groq_with_retry(messages)

            # _call_groq_with_retry returns a string on unrecoverable error
            if isinstance(response, str):
                return {"final_answer": response, "steps": steps_used, "trace": trace}

            msg = response.choices[0].message
            steps_used += 1

            # ── No tool call → final answer ───────────────────────────────
            if not getattr(msg, "tool_calls", None):
                answer = msg.content or ""
                if not answer.strip():
                    # Empty response → force a plain answer call
                    if tool_calls_made > 0:
                        answer = self._force_plain_answer(messages)
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

        final = self._force_plain_answer(messages)
        return {
            "final_answer": final or "Sorry, I couldn't summarise the results. Please try again.",
            "steps":        steps_used,
            "trace":        trace,
        }


    def _force_plain_answer(self, messages: list) -> str:
        try:
            forced_messages = list(messages) + [{
                "role":    "user",
                "content": (
                    "Based on the product data you already retrieved, "
                    "write a clear final answer now. List products with name, price (₹), and rating."
                ),
            }]
            response = self.client.chat.completions.create(
                model       = self.model,
                messages    = forced_messages,
                tool_choice = "none",        # ← forces text-only response
                temperature = self.temperature,
                max_tokens  = MAX_TOKENS_RESPONSE,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            return f"⚠️ Could not generate final answer: {e}"


    def _call_groq_with_retry(self, messages: list):
        """
        Handles 429 (rate limit) and 413 (too large) gracefully.
         """
        local_messages = list(messages)   # don't mutate the original

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
                        f"or switch to **llama-3.3-70b-versatile** in the sidebar "
                        f"(it has a much higher TPM limit)."
                    )
                if "413" in err or "too large" in err.lower() or "payload" in err.lower():
                    # Drop the oldest non-system message and retry once
                    non_sys = [i for i, m in enumerate(local_messages) if m["role"] != "system"]
                    if non_sys:
                        local_messages.pop(non_sys[0])
                        continue
                    return (
                        "⚠️ Context too large (413). "
                        "Please press **🗑️ Clear Chat** and ask a shorter question."
                    )

                # ── 400 bad request ───────────────────────────────────────
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

            # filter_products / search_products
            if "products" in obj and isinstance(obj["products"], list):
                obj["products"] = obj["products"][:4]
                obj.pop("filters", None)
                s = json.dumps(obj, ensure_ascii=False)
                if len(s) <= MAX_TOOL_RESULT_CHARS:
                    return s

            # get_catalog_structure
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

            # compare_products
            if "segments" in obj and isinstance(obj["segments"], dict):
                for seg in obj["segments"]:
                    if isinstance(obj["segments"][seg], list):
                        obj["segments"][seg] = obj["segments"][seg][:1]
                s = json.dumps(obj, ensure_ascii=False)
                if len(s) <= MAX_TOOL_RESULT_CHARS:
                    return s

            # get_subcategory_details
            if "all_brands" in obj:
                obj["all_brands"] = obj["all_brands"][:10]
                s = json.dumps(obj, ensure_ascii=False)
                if len(s) <= MAX_TOOL_RESULT_CHARS:
                    return s

        except Exception:
            pass

        return raw[:MAX_TOOL_RESULT_CHARS] + " ...[truncated]"


    @staticmethod
    def _parse_retry_after(err_str: str) -> float:
        m = re.search(r"try again in\s+([\d.]+)s", err_str, re.IGNORECASE)
        if m:
            return float(m.group(1)) + 1.0
        return 12.0

    @staticmethod
    def _trim_str(text: str, limit: int) -> str:
        return text[:limit] + "…" if len(text) > limit else text


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