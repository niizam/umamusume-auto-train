"""Utilities for delegating decisions to an LLM via the OpenAI API."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

try:
  from openai import OpenAI
except ImportError:  # pragma: no cover - handled gracefully at runtime
  OpenAI = None  # type: ignore


def _get_client(llm_config: Dict[str, Any]) -> Optional[OpenAI]:
  """Create an OpenAI client based on configuration."""

  if not llm_config.get("enabled", False):
    return None

  if OpenAI is None:
    print("[LLM] OpenAI python package not installed. Skipping LLM decision.")
    return None

  api_key = llm_config.get("api_key", "lm-studio")
  if not api_key:
    print(f"[LLM] Environment variable '{api_key}' not set. Skipping LLM decision.")
    return None

  client_kwargs: Dict[str, Any] = {"api_key": api_key}
  base_url = llm_config.get("base_url", "https://127.0.0.1:1234/v1")
  if base_url:
    client_kwargs["base_url"] = base_url

  try:
    return OpenAI(**client_kwargs)  # type: ignore[call-arg]
  except Exception as exc:  # pragma: no cover - depends on environment
    print(f"[LLM] Failed to initialise OpenAI client: {exc}")
    return None


def _format_config_summary(context: Dict[str, Any]) -> str:
  """Create a concise natural-language description of the training config."""

  parts: list[str] = []

  priority = context.get("priority_order")
  if isinstance(priority, list) and priority:
    parts.append(
      "Prioritise training stats in this order when options are similar: "
      + ", ".join(str(stat).upper() for stat in priority)
    )

  max_failure = context.get("maximum_failure")
  if isinstance(max_failure, (int, float)):
    parts.append(
      f"Avoid trainings with failure rates above {max_failure}%."
    )

  min_support = context.get("minimum_support")
  if isinstance(min_support, int) and min_support > 0:
    parts.append(
      f"Prefer trainings with at least {min_support} support cards unless failure chance is 0%."
    )

  do_race = context.get("do_race_when_bad_training")
  if do_race is not None:
    if do_race:
      parts.append(
        "If all viable trainings look poor, racing is allowed instead of resting."
      )
    else:
      parts.append(
        "Avoid scheduling races solely because training options are weak; prefer rest or priority stats."
      )

  stat_caps = context.get("stat_caps")
  if isinstance(stat_caps, dict) and stat_caps:
    cap_desc = ", ".join(
      f"{str(stat).upper()}≤{value}" for stat, value in stat_caps.items()
    )
    parts.append(
      "Do not suggest training stats that have reached their caps (" + cap_desc + ")."
    )

  if not parts:
    return "Follow the provided configuration strictly when making decisions."

  return " ".join(parts)


def _build_messages(context: Dict[str, Any]) -> list[Dict[str, Any]]:
  """Compose conversation messages for the LLM."""

  config_summary = _format_config_summary(context)

  system_prompt = (
    "You help manage training decisions for an automatic Uma Musume trainer. "
    "Use the provided OCR data, configuration thresholds, and current stats to "
    "choose the safest productive action. You must always respond by calling "
    "the `choose_training_action` function. Prefer returning `fallback` when "
    "the supplied data looks inconsistent or insufficient. "
    + config_summary
  )

  user_payload = json.dumps(context, ensure_ascii=False, indent=2)

  return [
    {"role": "system", "content": system_prompt},
    {
      "role": "user",
      "content": (
        "Here is the latest OCR snapshot for the game state. "
        "Return your chosen action through the tool call.\n" + user_payload
      ),
    },
  ]


def _tool_definition() -> Dict[str, Any]:
  """Describe the function tool exposed to the LLM."""

  return {
    "type": "function",
    "function": {
      "name": "choose_training_action",
      "description": (
        "Select the next action for the automation script based on the "
        "observed training options and thresholds."
      ),
      "parameters": {
        "type": "object",
        "properties": {
          "action": {
            "type": "string",
            "enum": ["train", "rest", "race", "fallback"],
            "description": "High-level action to perform.",
          },
          "stat": {
            "type": ["string", "null"],
            "enum": ["spd", "sta", "pwr", "guts", "wit", None],
            "description": "Training stat to focus when action is 'train'.",
          },
          "reason": {
            "type": "string",
            "description": "Short explanation for logging.",
          },
        },
        "required": ["action", "reason"],
        "additionalProperties": False,
      },
    },
  }


def decide_training_action(
  llm_config: Dict[str, Any], context: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
  """Ask the LLM to pick an action given OCR-derived context."""

  client = _get_client(llm_config)
  if client is None:
    return None

  model = llm_config.get("model")
  if not model:
    print("[LLM] No model configured. Skipping LLM decision.")
    return None

  messages = _build_messages(context)
  tools = [_tool_definition()]

  try:
    response = client.chat.completions.create(
      model=model,
      messages=messages,
      tools=tools,
      tool_choice="required",
    )
  except Exception as exc:  # pragma: no cover - depends on external service
    print(f"[LLM] OpenAI response error: {exc}")
    return None

  if not getattr(response, "output", None):
    return None

  for item in response.output:
    if item.type != "tool_calls":
      continue
    for tool_call in item.tool_calls:
      if tool_call.function.name != "choose_training_action":
        continue
      arguments = tool_call.function.arguments or "{}"
      try:
        parsed = json.loads(arguments)
      except json.JSONDecodeError:
        print(f"[LLM] Failed to parse tool arguments: {arguments}")
        return None

      if isinstance(parsed, dict):
        return parsed

  return None