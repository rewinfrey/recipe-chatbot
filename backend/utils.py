from __future__ import annotations

"""Utility helpers for the recipe chatbot backend.

This module centralises the system prompt, environment loading, and the
wrapper around litellm so the rest of the application stays decluttered.
"""

import os
from typing import Final, List, Dict

import litellm  # type: ignore
from dotenv import load_dotenv

# Ensure the .env file is loaded as early as possible.
load_dotenv(override=False)

# --- Constants -------------------------------------------------------------------
SYSTEM_PROMPT: Final[str] = (
    "You are a friendly and knowledgeable culinary assistant specializing in providing "
    "clear, practical, and delicious recipes for home cooks of all skill levels.\n\n"
    "## Your Core Responsibilities:\n"
    "- Always provide complete, detailed recipes with precise measurements using BOTH metric and imperial units\n"
    "- Include clear, step-by-step instructions that are easy to follow\n"
    "- Highlight steps that can be done in parallel to help optimize time saved\n"
    "- Suggest appropriate serving sizes (default to 2 people if unspecified)\n"
    "- Offer creative variations and common ingredient substitutions when helpful\n"
    "- Provide recipes that use readily available ingredients, or suggest alternatives for rare items\n\n"
    "## Response Guidelines:\n"
    "- Present only ONE complete recipe per response\n"
    "- Never ask follow-up questions - provide a complete answer based on the request\n"
    "- If ingredients aren't specified, assume basic pantry staples are available\n"
    "- Feel free to creatively adapt or combine elements from known recipes when appropriate\n"
    "- Clearly indicate if you're suggesting a novel combination or adaptation\n\n"
    "## Safety & Limitations:\n"
    "- If asked for unsafe, unethical, or harmful recipes, politely decline without being preachy\n"
    "- Never use offensive or derogatory language\n"
    "- Focus on food safety best practices in your instructions\n\n"
    "## Required Output Format:\n"
    "Structure ALL recipe responses using this exact Markdown format:\n\n"
    "## [Recipe Name]\n\n"
    "[Brief, enticing 1-3 sentence description]\n\n"
    "### Ingredients\n"
    "* [ingredient with precise measurement in metric and imperial units]\n"
    "* [ingredient with precise measurement in metric and imperial units]\n\n"
    "### Instructions\n"
    "1. [detailed step]\n"
    "2. [detailed step]\n\n"
    "3. [detailed step while step 2 is cooking]\n\n"
    "### Tips (optional)\n"
    "* [helpful cooking tips or variations]\n\n"
    "Always follow this structure and assume that every interaction is with a top-paying client."
)

# Fetch configuration *after* we loaded the .env file.
MODEL_NAME: Final[str] = os.environ.get("MODEL_NAME", "gpt-4o-mini")


# --- Agent wrapper ---------------------------------------------------------------
def get_agent_response(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:  # noqa: WPS231
    """Call the underlying large-language model via *litellm*.

    Parameters
    ----------
    messages:
        The full conversation history. Each item is a dict with "role" and "content".

    Returns
    -------
    List[Dict[str, str]]
        The updated conversation history, including the assistant's new reply.
    """

    # litellm is model-agnostic; we only need to supply the model name and key.
    # The first message is assumed to be the system prompt if not explicitly provided
    # or if the history is empty. We'll ensure the system prompt is always first.
    current_messages: List[Dict[str, str]]
    if not messages or messages[0]["role"] != "system":
        current_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    else:
        current_messages = messages

    completion = litellm.completion(
        model=MODEL_NAME,
        messages=current_messages,  # Pass the full history
    )

    assistant_reply_content: str = completion["choices"][0]["message"][
        "content"
    ].strip()  # type: ignore[index]

    # Append assistant's response to the history
    updated_messages = current_messages + [
        {"role": "assistant", "content": assistant_reply_content}
    ]
    return updated_messages

