"""Conversations: context_id -> the Claude session that holds it, its owner, its turns.

Memory-only LRU (cap 256). A context belongs to the user who opened it; another user asking
into it is refused (409 semantics). The Claude session id is what `--resume` needs: the second
turn of a conversation is a cache read, not a re-read of the whole prompt.
"""

import threading
import time
from collections import OrderedDict


class Context:
    __slots__ = ("context_id", "user", "created_at", "session_claude", "turns")

    def __init__(self, context_id, user):
        self.context_id = context_id
        self.user = user
        self.created_at = time.time()
        self.session_claude = None
        self.turns = []  # [{request_id, ts, q, tier, terminal, credits}]


class Contexts:
    def __init__(self, cap=256):
        self.cap = cap
        self._d = OrderedDict()
        self._lock = threading.Lock()

    def get_or_create(self, context_id, user):
        """Returns (context, created). Raises PermissionError when the context belongs to another user."""
        with self._lock:
            c = self._d.get(context_id)
            if c is not None:
                if c.user != user:
                    raise PermissionError(f"context {context_id} belongs to {c.user}")
                self._d.move_to_end(context_id)
                return c, False
            c = Context(context_id, user)
            self._d[context_id] = c
            while len(self._d) > self.cap:
                self._d.popitem(last=False)
            return c, True

    def get(self, context_id):
        with self._lock:
            return self._d.get(context_id)

    def __len__(self):
        return len(self._d)
