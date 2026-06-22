"""Conversation state (in-memory)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class DialogState:
    system_prompt: str
    max_turns: int = 6
    history: list[dict] = field(default_factory=list)

    def build_messages(self, user_text: str) -> list[dict]:
        msgs: list[dict] = [{"role": "system", "content": self.system_prompt}]
        # keep last turns
        if self.max_turns > 0 and self.history:
            msgs.extend(self.history[-2 * self.max_turns :])
        msgs.append({"role": "user", "content": user_text})
        return msgs

    def add_turn(self, user_text: str, assistant_text: str) -> None:
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": assistant_text})

