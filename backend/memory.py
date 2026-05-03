"""
memory.py — Conversation memory for multi-turn incident sessions
NEW feature: Keeps context across follow-up queries in the same incident
"""
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional
from models import ConversationMessage

class SessionMemory:
    """
    In-memory conversation store.
    Each session_id maps to a list of messages.
    Sessions expire after 2 hours of inactivity.
    """

    def __init__(self, ttl_hours: int = 2, max_messages: int = 20):
        self._sessions: dict[str, list[ConversationMessage]] = defaultdict(list)
        self._last_access: dict[str, datetime] = {}
        self.ttl = timedelta(hours=ttl_hours)
        self.max_messages = max_messages

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """Add a message to a session."""
        self._cleanup_expired()
        self._sessions[session_id].append(
            ConversationMessage(role=role, content=content)
        )
        self._last_access[session_id] = datetime.now()
        # Keep only the last N messages (sliding window)
        if len(self._sessions[session_id]) > self.max_messages:
            self._sessions[session_id] = self._sessions[session_id][-self.max_messages:]

    def get_history(self, session_id: str) -> list[ConversationMessage]:
        """Get all messages for a session."""
        self._last_access[session_id] = datetime.now()
        return self._sessions.get(session_id, [])

    def get_context_string(self, session_id: str) -> str:
        """Format conversation history as a string for the LLM prompt."""
        messages = self.get_history(session_id)
        if not messages:
            return ""

        lines = ["\n--- PREVIOUS CONVERSATION CONTEXT ---"]
        for msg in messages[-6:]:  # Last 3 exchanges
            prefix = "DISPATCHER" if msg.role == "user" else "AI"
            lines.append(f"{prefix}: {msg.content[:300]}")
        lines.append("--- END CONTEXT ---\n")
        return "\n".join(lines)

    def clear_session(self, session_id: str) -> None:
        """Clear a session's history."""
        self._sessions.pop(session_id, None)
        self._last_access.pop(session_id, None)

    def get_session_count(self) -> int:
        return len(self._sessions)

    def _cleanup_expired(self) -> None:
        """Remove sessions older than TTL."""
        now = datetime.now()
        expired = [
            sid for sid, last in self._last_access.items()
            if now - last > self.ttl
        ]
        for sid in expired:
            self._sessions.pop(sid, None)
            self._last_access.pop(sid, None)

# Global singleton — shared across all requests
memory_store = SessionMemory()