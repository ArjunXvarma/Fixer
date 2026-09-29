import os
from dataclasses import dataclass
from typing import Callable

from dotenv import load_dotenv
from langchain_core.utils.function_calling import convert_to_openai_tool

from fixer.agent.tools import TOOLS

load_dotenv()

MAX_RETRIES = 10
DEFAULT_PROVIDER = "gemini"


def _gemini(model: str, max_retries: int):
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model=model, max_retries=max_retries)


def _groq(model: str, max_retries: int):
    from langchain_groq import ChatGroq

    return ChatGroq(model=model, max_retries=max_retries)


def _openrouter(model: str, max_retries: int):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        max_retries=max_retries,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )


@dataclass(frozen=True)
class Provider:
    build: Callable[[str, int], object]
    model: str
    api_key_env: str
    structured_method: str = "json_schema"


PROVIDERS = {
    "gemini": Provider(_gemini, "gemini-3.1-flash-lite", "GOOGLE_API_KEY"),
    "groq": Provider(
        _groq,
        "openai/gpt-oss-20b",
        "GROQ_API_KEY",
        structured_method="function_calling",
    ),
    "openrouter": Provider(
        _openrouter,
        "openrouter/free",
        "OPENROUTER_API_KEY",
        structured_method="function_calling",
    ),
}


class LLM:
    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        max_retries: int = MAX_RETRIES,
    ):
        name = provider or os.getenv("FIXER_PROVIDER") or DEFAULT_PROVIDER

        if name not in PROVIDERS:
            raise ValueError(
                f"Unknown provider {name!r}. Choose from {sorted(PROVIDERS)}."
            )

        spec = PROVIDERS[name]

        if not os.getenv(spec.api_key_env):
            raise RuntimeError(
                f"Provider {name!r} needs {spec.api_key_env} in your .env file."
            )

        self.provider = name
        self.model = model or os.getenv("FIXER_MODEL") or spec.model
        self.max_retries = max_retries
        self.spec = spec

    def __repr__(self):
        return f"LLM({self.provider}:{self.model})"

    def client(self, tools=TOOLS, tool_choice: str | None = None):
        model = self.spec.build(self.model, self.max_retries)

        if not tools:
            return model

        choice = {"tool_choice": tool_choice} if tool_choice else {}

        return model.bind_tools(
            [convert_to_openai_tool(tool) for tool in tools], **choice
        )

    def structured(self, schema, method: str | None = None):
        return self.client(tools=None).with_structured_output(
            schema, method=method or self.spec.structured_method
        )


def build_model(tools=TOOLS, tool_choice: str | None = None):
    return LLM().client(tools=tools, tool_choice=tool_choice)


def build_structured_model(schema):
    return LLM().structured(schema)
