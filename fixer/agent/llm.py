from dotenv import load_dotenv
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_google_genai import ChatGoogleGenerativeAI

from fixer.agent.tools import TOOLS

load_dotenv()

MODEL_NAME = "gemini-3.1-flash-lite"


def build_model(tools=TOOLS):
    """
    The model every architecture talks to.

    bind_tools on Gemini builds its function schemas from the tool's full
    signature, which would show the model the injected repo_path. These specs
    leave injected arguments out; ToolNode still runs the real tools.
    """

    return ChatGoogleGenerativeAI(model=MODEL_NAME).bind_tools(
        [convert_to_openai_tool(tool) for tool in tools]
    )
