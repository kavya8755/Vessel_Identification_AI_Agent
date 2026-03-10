"""
Conversational AI Vessel Agent
================================
A stateful conversational agent that:
  - Maintains session history
  - Caches repeated vessel queries
  - Parses user intent → structured DB filters
  - Calls the VesselSearchEngine (structured data layer)
  - Feeds ONLY retrieved data to the LLM (RAG pattern to prevent hallucination)

Architecture:
  User message
       │
       ▼
  IntentParser  ──► structured filters
       │
       ▼
  VesselSearchEngine  ──► retrieved records (ground truth)
       │
       ▼
  QueryCache (LRU-style)
       │
       ▼
  LLM Prompt Builder  (injects records as context)
       │
       ▼
  Groq API (llama-3.3-70b-versatile)  ──► grounded answer
       │
       ▼
  SessionManager (appends to history)
       │
       ▼
  User
"""

import hashlib
import json
import time
import re
import os
import urllib.request
import urllib.error
import pandas as pd
import numpy as np
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Groq API config
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", Key)


# ---------------------------------------------------------------------------
# Query Cache (LRU, in-memory)
# ---------------------------------------------------------------------------
class QueryCache:
    """Simple LRU cache for vessel query results."""

    def __init__(self, max_size: int = 200, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl = ttl_seconds
        self._cache: OrderedDict[str, dict] = OrderedDict()

    def _key(self, query: str, filters: dict) -> str:
        payload = json.dumps({"q": query.lower().strip(), "f": filters}, sort_keys=True)
        return hashlib.md5(payload.encode()).hexdigest()

    def get(self, query: str, filters: dict) -> Optional[str]:
        k = self._key(query, filters)
        if k not in self._cache:
            return None
        entry = self._cache[k]
        if time.time() - entry["ts"] > self.ttl:
            del self._cache[k]
            return None
        # LRU: move to end
        self._cache.move_to_end(k)
        logger.debug("Cache HIT for key %s", k[:8])
        return entry["value"]

    def set(self, query: str, filters: dict, value: str) -> None:
        k = self._key(query, filters)
        if k in self._cache:
            self._cache.move_to_end(k)
        self._cache[k] = {"value": value, "ts": time.time()}
        if len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    def stats(self) -> dict:
        return {"size": len(self._cache), "max_size": self.max_size, "ttl_seconds": self.ttl}


# ---------------------------------------------------------------------------
# Session Manager
# ---------------------------------------------------------------------------
@dataclass
class Message:
    role: str          # "user" | "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)
    retrieved_vessels: list = field(default_factory=list)


class SessionManager:
    """Manages per-session conversation history."""

    def __init__(self, session_id: str, max_history: int = 20):
        self.session_id = session_id
        self.max_history = max_history
        self.messages: list[Message] = []
        self.context: dict = {}        # e.g. last mentioned vessel_id, active filters

    def add_message(self, role: str, content: str, retrieved: list = None) -> None:
        self.messages.append(Message(role=role, content=content,
                                     retrieved_vessels=retrieved or []))
        if len(self.messages) > self.max_history * 2:
            self.messages = self.messages[-self.max_history * 2:]

    def get_history_for_llm(self, last_n: int = 6) -> list[dict]:
        """Return last N message pairs formatted for the Anthropic messages API."""
        recent = self.messages[-(last_n * 2):]
        return [{"role": m.role, "content": m.content} for m in recent]

    def update_context(self, **kwargs) -> None:
        self.context.update(kwargs)


# ---------------------------------------------------------------------------
# Intent Parser
# ---------------------------------------------------------------------------
class IntentParser:
    """
    Lightweight rule-based intent parser.
    Converts user messages into structured filter dicts
    that VesselSearchEngine.filter_query() can consume.
    """

    INTENTS = ["lookup", "search", "filter", "position", "history", "compare", "summarise"]

    def parse(self, message: str, session_context: dict) -> dict:
        msg_lower = message.lower()
        result: dict = {"intent": "search", "filters": {}, "raw_query": message}

        # Intent detection
        if any(w in msg_lower for w in ["where is", "position of", "location of", "coordinates"]):
            result["intent"] = "position"
        elif any(w in msg_lower for w in ["history", "changed", "previous name", "flag change"]):
            result["intent"] = "history"
        elif any(w in msg_lower for w in ["compare", "difference between", "vs"]):
            result["intent"] = "compare"
        elif any(w in msg_lower for w in ["list", "show all", "find all", "filter"]):
            result["intent"] = "filter"
        elif any(w in msg_lower for w in ["summarise", "summary", "tell me about", "details on"]):
            result["intent"] = "summarise"

        # Extract entities
        filters = {}

        imo_m = re.search(r"\bimo[:\s#]*(\d{7})\b", msg_lower)
        mmsi_m = re.search(r"\bmmsi[:\s#]*(\d{9})\b", msg_lower)
        if imo_m:
            filters["imo"] = int(imo_m.group(1))
            result["intent"] = "lookup"
        if mmsi_m:
            filters["mmsi"] = int(mmsi_m.group(1))
            result["intent"] = "lookup"

        # Vessel type
        type_kw = {
            "tanker": "tanker", "crude tanker": "Crude Tanker",
            "chemical tanker": "Chemical Tanker", "bulk": "Dry Bulk",
            "container": "Container", "lng": "LNG", "tug": "Tug",
            "fishing": "Fishing", "passenger": "Passenger", "cargo": "General Cargo",
            "ro-ro": "Ro-Ro", "feeder": "Container",
        }
        for kw, vt in type_kw.items():
            if kw in msg_lower:
                filters["vessel_type"] = vt
                break

        # Flag (ISO2 preceded by "flag")
        flag_m = re.search(r"flag\s+([a-z]{2})\b", msg_lower)
        if flag_m:
            filters["flag"] = flag_m.group(1).upper()

        # Year ranges
        year_range_m = re.search(r"built\s+between\s+(\d{4})\s+and\s+(\d{4})", msg_lower)
        year_exact_m = re.search(r"built\s+in\s+(\d{4})", msg_lower)
        if year_range_m:
            filters["builtYear_min"] = int(year_range_m.group(1))
            filters["builtYear_max"] = int(year_range_m.group(2))
        elif year_exact_m:
            filters["builtYear_min"] = filters["builtYear_max"] = int(year_exact_m.group(1))

        # DWT ranges
        dwt_m = re.search(r"deadweight[>\s]+(\d+)", msg_lower)
        if dwt_m:
            filters["deadweight_min"] = float(dwt_m.group(1))

        # Vessel name (quoted or "called X")
        name_m = re.search(r'(?:called|named)\s+"?([a-z0-9 _\-]+)"?', msg_lower)
        if name_m:
            filters["name"] = name_m.group(1).strip()

        # Carry over context filters from session
        ctx_filters = session_context.get("active_filters", {})
        for k, v in ctx_filters.items():
            if k not in filters:
                filters[k] = v

        result["filters"] = filters
        return result


# ---------------------------------------------------------------------------
# Prompt Builder (RAG pattern)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are VesselAI, an expert maritime intelligence assistant powered by Groq.
You answer questions strictly based on RETRIEVED VESSEL DATA provided in each message.
Rules:
1. NEVER invent vessel details not present in the retrieved data.
2. If data is missing or ambiguous, say so explicitly.
3. Use nautical terminology correctly.
4. When asked for positions, provide lat/lon and the last update timestamp.
5. You may provide general maritime knowledge as background context,
   but always distinguish it from retrieved data.
6. If the user's query cannot be answered from the retrieved data,
   say: "I don't have that information in the current dataset."
"""


def build_prompt(user_message: str, retrieved_json: str, intent: dict,
                 session_history: list[dict]) -> list[dict]:
    """Build the messages array for the Anthropic API."""
    system_context = (
        f"INTENT DETECTED: {intent['intent']}\n"
        f"ACTIVE FILTERS: {json.dumps(intent['filters'])}\n\n"
        f"RETRIEVED VESSEL DATA (use this as ground truth):\n{retrieved_json}\n\n"
        "Answer the user based ONLY on the above data."
    )

    messages = []
    # Include recent history
    for m in session_history:
        messages.append(m)

    # Inject retrieved data into the current user turn
    augmented_user = f"{user_message}\n\n---\n{system_context}"
    messages.append({"role": "user", "content": augmented_user})

    return messages


# ---------------------------------------------------------------------------
# Main Agent
# ---------------------------------------------------------------------------
class VesselConversationalAgent:
    """
    Orchestrates the full pipeline:
    parse intent → search → cache → build prompt → call LLM → return answer
    """

    def __init__(self, search_engine, api_key: Optional[str] = None):
        from vessel_search import VesselSearchEngine  # local import for modularity
        self.engine: VesselSearchEngine = search_engine
        self.cache = QueryCache(max_size=200, ttl_seconds=300)
        self.intent_parser = IntentParser()
        self.sessions: dict[str, SessionManager] = {}
        self._api_key = api_key or GROQ_API_KEY

    def get_or_create_session(self, session_id: str) -> SessionManager:
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionManager(session_id)
        return self.sessions[session_id]

    def chat(self, user_message: str, session_id: str = "default") -> str:
        session = self.get_or_create_session(session_id)

        # 1. Parse intent
        intent = self.intent_parser.parse(user_message, session.context)
        logger.info("Intent: %s | Filters: %s", intent["intent"], intent["filters"])

        # 2. Check cache
        cached = self.cache.get(user_message, intent["filters"])
        if cached:
            session.add_message("user", user_message)
            session.add_message("assistant", cached)
            return cached

        # 3. Retrieve from structured engine
        results_df = self._retrieve(intent)
        retrieved_json = self.engine.format_results(results_df, max_rows=5)

        # 4. Build prompt
        history = session.get_history_for_llm(last_n=5)
        messages = build_prompt(user_message, retrieved_json, intent, history)

        # 5. Call LLM (or mock)
        answer = self._call_llm(messages)

        # 6. Cache and persist to session
        self.cache.set(user_message, intent["filters"], answer)
        session.add_message("user", user_message, retrieved=results_df.to_dict("records")[:5])
        session.add_message("assistant", answer)

        # 7. Update session context with new filters
        if intent["filters"]:
            session.update_context(active_filters=intent["filters"])

        return answer

    def _retrieve(self, intent: dict) -> pd.DataFrame:
        filters = intent["filters"]
        query = intent["raw_query"]

        if intent["intent"] == "lookup" and ("imo" in filters or "mmsi" in filters):
            if "imo" in filters:
                return self.engine.lookup_by_imo(filters["imo"])
            return self.engine.lookup_by_mmsi(filters["mmsi"])

        if filters:
            return self.engine.filter_query(filters, top_k=10)

        return self.engine.search_by_name(query, top_k=5)

    def _call_llm(self, messages: list[dict]) -> str:
        if not self._api_key:
            last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
            return (
                f"[MOCK RESPONSE - No GROQ_API_KEY configured]\n"
                f"Query: '{last_user[:80]}...'\n"
                f"Set GROQ_API_KEY to enable live responses."
            )
        try:
            # Groq uses OpenAI-compatible /v1/chat/completions
            # Inject system prompt as first message
            full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

            payload = json.dumps({
                "model": GROQ_MODEL,
                "messages": full_messages,
                "max_tokens": 1024,
                "temperature": 0.2,
            }).encode("utf-8")

            req = urllib.request.Request(
                GROQ_API_URL,
                data=payload,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]

        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            logger.error("Groq HTTP error %s: %s", e.code, body)
            return f"Groq API error {e.code}: {body[:200]}"
        except Exception as e:
            logger.error("LLM call failed: %s", e)
            return f"Error communicating with Groq: {e}"


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from vessel_identity import VesselIdentityResolver
    from vessel_search import VesselSearchEngine

    raw = pd.read_csv("../data/dataset.csv", low_memory=False)
    resolver = VesselIdentityResolver(raw)
    resolver.parse_records()
    resolver.validate_identifiers()
    resolver.detect_conflicts()
    resolved = resolver.resolve_identities()

    engine = VesselSearchEngine(resolved)
    agent = VesselConversationalAgent(engine)  # uses GROQ_API_KEY from env or hardcoded default

    test_queries = [
        "Show me all LNG carriers",
        "Find vessel with IMO 9857365",
        "What container ships were built in 2023?",
        "Where is the MARCO right now?",
        "List dry bulk vessels with deadweight greater than 50000",
    ]

    print("=" * 60)
    print("VESSEL AI AGENT DEMO")
    print("=" * 60)
    for q in test_queries:
        print(f"\n>>> USER: {q}")
        response = agent.chat(q, session_id="demo_session")
        print(f"<<< AGENT: {response}")

    print("\n=== Cache Stats ===")
    print(agent.cache.stats())

    print("\n=== Session Context ===")
    print(agent.sessions["demo_session"].context)
