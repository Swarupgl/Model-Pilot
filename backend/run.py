"""Backend runner for ModelPilot Neuro-SAN multi-agent network."""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load local environment secrets (.env)
load_dotenv()

_backend_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_backend_dir)
_src_dir = os.path.join(_project_root, "src")

# Configure Neuro-SAN environment variables
os.environ["AGENT_MANIFEST_FILE"] = os.path.join(_backend_dir, "registries", "manifest.hocon")
os.environ["AGENT_TOOL_PATH"] = os.path.join(_backend_dir, "coded_tools")

# Ensure backend and src directories are in PYTHONPATH
_existing_pythonpath = os.environ.get("PYTHONPATH", "")
paths_to_add = [_backend_dir, _src_dir]
for p in paths_to_add:
    if p not in _existing_pythonpath.split(os.pathsep):
        _existing_pythonpath = f"{p}{os.pathsep}{_existing_pythonpath}" if _existing_pythonpath else p
os.environ["PYTHONPATH"] = _existing_pythonpath

if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

from neuro_san.client.agent_session_factory import DirectAgentSessionFactory


def invoke_agent(agent_name: str = "modelpilot", user_text: str = "", sly_data=None, chat_context=None):
    """Invoke a Neuro-SAN agent session and return the final response message.

    Args:
        agent_name: Name of the agent (without .hocon suffix).
        user_text: Prompt message from user.
        sly_data: Optional session data dictionary.
        chat_context: Optional conversation context to maintain multi-turn chat.

    Returns:
        Dictionary containing agent response message and chat context.
    """
    factory = DirectAgentSessionFactory()
    session = factory.create_session(
        agent_name=agent_name,
        use_direct=True,
        metadata={},
    )

    request_payload = {
        "user_message": {
            "text": user_text,
        },
        "sly_data": sly_data,
    }

    if chat_context:
        request_payload["chat_context"] = chat_context

    stream = session.streaming_chat(request_payload)
    last_response = {}
    for chat_msg in stream:
        if "response" in chat_msg:
            last_response = chat_msg
        if chat_msg.get("done") is True:
            break

    return last_response if last_response else {"response": {"text": "No response received."}}


if __name__ == "__main__":
    prompt = "Diagnose the built-in iris dataset and recommend the most promising classification algorithms."
    print(f"Invoking ModelPilot agent with prompt: '{prompt}'...")
    response = invoke_agent("modelpilot", prompt)
    print("\n--- ModelPilot Response ---")
    if "response" in response and "text" in response["response"]:
        print(response["response"]["text"])
    else:
        print(response)
