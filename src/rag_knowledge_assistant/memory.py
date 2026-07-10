from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ConversationTurn:
    question: str
    answer: str


@dataclass
class ConversationMemory:
    turns: dict[str, list[ConversationTurn]] = field(default_factory=dict)

    def add_turn(self, conversation_id: str, question: str, answer: str) -> None:
        self.turns.setdefault(conversation_id, []).append(ConversationTurn(question=question, answer=answer))

    def get_history(self, conversation_id: str) -> list[ConversationTurn]:
        return self.turns.get(conversation_id, [])


conversation_memory = ConversationMemory()
