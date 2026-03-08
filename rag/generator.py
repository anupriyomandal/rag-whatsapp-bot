"""Answer generator that calls the chat model with retrieved context."""

from __future__ import annotations

from pathlib import Path
from typing import List

from openai import OpenAI


class AnswerGenerator:
    """Builds prompts and generates final answers with OpenAI chat."""

    def __init__(self, client: OpenAI, model: str, prompt_path: Path) -> None:
        self._client = client
        self._model = model
        self._prompt_template = self._load_prompt_template(prompt_path)

    def generate(self, question: str, context_chunks: List[str]) -> str:
        """Generate a concise answer grounded in retrieved context."""
        context_block = "\n\n".join(
            f"[{idx + 1}] {chunk}" for idx, chunk in enumerate(context_chunks)
        )

        system_message = self._prompt_template.format(
            context=context_block,
            question=question.strip(),
        )

        completion = self._client.chat.completions.create(
            model=self._model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": question.strip()},
            ],
        )

        answer = completion.choices[0].message.content
        if not answer:
            raise RuntimeError("LLM returned an empty answer")
        return answer.strip()

    @staticmethod
    def _load_prompt_template(prompt_path: Path) -> str:
        """Load prompt file from disk and ensure required placeholders exist."""
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {prompt_path}")

        template = prompt_path.read_text(encoding="utf-8").strip()
        required = ("{context}", "{question}")
        if not all(placeholder in template for placeholder in required):
            raise ValueError(
                "Prompt template must include both {context} and {question} placeholders"
            )
        return template
