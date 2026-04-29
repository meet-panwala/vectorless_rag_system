import json
import os
from typing import Generator
from groq import Groq
from .catalog_tools import TOOL_SCHEMAS, ToolExecutor
from .data_store import DataStore
from .index_builder import load_catalog_tree

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a smart, helpful Flipkart fashion shopping assistant.
You have access to a catalog of 30,000+ fashion products.

## How you retrieve information:
1. ALWAYS start by understanding the catalog structure.
2. If the user asks for "best", sort by rating.
3. If the user asks for "cheapest", sort by price_asc.

## Guidelines:
- Format product recommendations as a clear numbered list.
- Always mention price (₹) and rating.
- Do NOT make up products.
"""


class CatalogAgent:
    def __init__(
            self,
            data_store: DataStore,
            catalog_tree: dict,
            model: str = "llama-3.3-70b-versatile",
            max_tool_calls: int = 8,
            temperature: float = 0.3,
    ):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.executor = ToolExecutor(data_store, catalog_tree)
        self.full_tree = catalog_tree
        self.model = model
        self.max_steps = max_tool_calls
        self.temperature = temperature

    def _get_safe_context(self, max_chars=4000):
        """
        MISSING FUNCTION ADDED:
        Safely summarizes the catalog so we don't hit the 413 error.
        """
        # Convert tree to string
        tree_str = json.dumps(self.full_tree)

        if len(tree_str) <= max_chars:
            return tree_str

        # If too big, create a 'Lite' version (just top-level categories)
        lite_tree = {
            "total_products": self.full_tree.get("total_products"),
            "categories": list(self.full_tree.get("categories", {}).keys()),
            "note": "Context truncated. Use tools to find specific products."
        }
        return json.dumps(lite_tree)

    def _get_completion_params(self, messages):
        """Logic to handle model-specific parameter requirements."""
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": 1024,  # Reduced from 4096 to prevent payload bloat
        }

        # groq/compound DOES NOT support the 'tools' parameter
        if "compound" not in self.model:
            params["tools"] = TOOL_SCHEMAS
            params["tool_choice"] = "auto"

        return params

    def get_tool_trace(self, user_message: str, history: list[dict] = None) -> dict:
        # Limit history to prevent 413 error on long chats
        recent_history = history[-3:] if history else []
        messages = self._build_messages(user_message, recent_history)

        # Use the safe context helper
        if "compound" in self.model:
            context = self._get_safe_context(max_chars=5000)
            messages[0]["content"] += f"\n\nCATALOG CONTEXT: {context}"

        trace = []
        steps_used = 0

        while steps_used < self.max_steps:
            try:
                params = self._get_completion_params(messages)
                response = self.client.chat.completions.create(**params)

                msg = response.choices[0].message
                steps_used += 1

                if not hasattr(msg, 'tool_calls') or not msg.tool_calls:
                    return {
                        "final_answer": msg.content or "No products found.",
                        "steps": steps_used,
                        "trace": trace,
                    }

                # Tool Execution Loop
                for tc in msg.tool_calls:
                    args = json.loads(tc.function.arguments)
                    result_str = self.executor.execute(tc.function.name, args)

                    # Truncate tool result if it's massive
                    if len(result_str) > 8000:
                        result_str = result_str[:8000] + "... [Truncated]"

                    trace.append({
                        "step": steps_used,
                        "tool": tc.function.name,
                        "args": args,
                        "result": json.loads(result_str) if "{" in result_str else result_str,
                    })

                    messages.append({"role": "assistant", "tool_calls": [tc]})
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_str})

            except Exception as e:
                return {"final_answer": f"⚠️ Error: {str(e)}", "steps": steps_used, "trace": trace}

        return {"final_answer": "Max steps reached.", "steps": steps_used, "trace": trace}

    @staticmethod
    def _build_messages(user_message: str, history: list[dict] = None) -> list[dict]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:
            for turn in history:
                messages.append({"role": turn.get("role", "user"), "content": turn.get("content", "")})
        messages.append({"role": "user", "content": user_message})
        return messages


# ── Factory ───────────────────────────────────────────────────────────────────
_agent_instance = None
_last_model = None


def get_agent(json_path: str, tree_path: str, model: str = "llama-3.3-70b-versatile") -> CatalogAgent:
    global _agent_instance, _last_model
    if _agent_instance is None or _last_model != model:
        from .data_store import get_store
        store = get_store(json_path)
        tree = load_catalog_tree(tree_path)
        _agent_instance = CatalogAgent(store, tree, model=model)
        _last_model = model
    return _agent_instance