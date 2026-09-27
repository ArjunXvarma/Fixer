from dotenv import load_dotenv
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_google_genai import ChatGoogleGenerativeAI

from fixer.agent.tools import TOOLS

load_dotenv()

MODEL_NAME = "gemini-3.1-flash-lite"


def build_model(tools=TOOLS, tool_choice=None):
    """
    The model every architecture talks to.

    tools=None gives a model that cannot call tools at all. Use it for nodes
    that only think, such as a planner using with_structured_output.

    tool_choice="none" keeps the tools visible but forbids calling them, which
    is how you force a final written answer mid-conversation. Do not use
    tools=None for that: once the history is full of tool calls, Gemini emits
    function calls even with nothing bound.

    bind_tools on Gemini builds its function schemas from the tool's full
    signature, which would show the model the injected repo_path. These specs
    leave injected arguments out; ToolNode still runs the real tools.
    """

    model = ChatGoogleGenerativeAI(model=MODEL_NAME)

    if not tools:
        return model

    choice = {"tool_choice": tool_choice} if tool_choice else {}

    return model.bind_tools(
        [convert_to_openai_tool(tool) for tool in tools], **choice
    )
