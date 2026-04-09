import json
import os
import textwrap
from typing import Any, Optional

import requests
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = (
    os.getenv("HF_TOKEN")
    or os.getenv("OPENAI_API_KEY")
    or os.getenv("API_KEY", "")
)
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860").rstrip("/")
BENCHMARK = "data-quality-env"
TEMPERATURE = 0.0
MAX_TOKENS = 256
MAX_STEPS = 30
SCORE_EPSILON = 1e-4

TASKS = [
    "task1_missing_values",
    "task2_type_errors_duplicates",
    "task3_outliers_inconsistencies",
]


def clamp_open_score(value: float, epsilon: float = SCORE_EPSILON) -> float:
    return min(max(float(value), epsilon), 1.0 - epsilon)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    action_safe = action.replace("\n", " ").replace("\r", " ")
    print(
        f"[STEP] step={step} action={action_safe} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


SYSTEM_PROMPT = textwrap.dedent(
    """
You are a data quality expert agent.
Return exactly one JSON object with no extra text.
Allowed format examples:
{"action_type": "flag_issue", "issue_type": "missing", "row_index": 1, "column": "age"}
{"action_type": "fix_value", "issue_type": "type_error", "row_index": 2, "column": "price", "fixed_value": 19.99}
{"action_type": "submit"}
"""
).strip()


def _safe_parse_action(raw_text: str) -> dict[str, Any]:
    cleaned = (raw_text or "").strip().replace("```json", "").replace("```", "").strip()
    if not cleaned:
        return {"action_type": "submit"}

    candidates = [cleaned]
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(cleaned[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    return {"action_type": "submit"}


def get_action(client: OpenAI, obs: dict[str, Any], step: int, last_reward: float) -> dict[str, Any]:
    prompt = (
        f"Step: {step}\n"
        f"Last reward: {last_reward:.4f}\n"
        f"Task: {obs.get('task_description', '')}\n"
        f"Dataset: {json.dumps(obs.get('dataset', []))}\n"
        f"Issues found: {obs.get('issues_found_so_far', [])}\n"
        f"Hint: {obs.get('hint', '')}"
    )

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (resp.choices[0].message.content or "").strip()
        action = _safe_parse_action(text)
    except Exception:
        action = {"action_type": "submit"}

    if "action_type" not in action:
        action["action_type"] = "submit"
    return action


def env_reset(task_id: str) -> dict[str, Any]:
    response = requests.post(
        f"{ENV_BASE_URL}/reset",
        json={"task_id": task_id},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def env_step(task_id: str, action: dict[str, Any]) -> dict[str, Any]:
    payload_action = dict(action)
    payload_action["task_id"] = task_id
    response = requests.post(
        f"{ENV_BASE_URL}/step",
        json={"task_id": task_id, "action": payload_action},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def run_task(llm_client: OpenAI, task_id: str) -> None:
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
    rewards: list[float] = []
    steps_taken = 0
    score = 0.0

    try:
        reset_result = env_reset(task_id)
        obs = reset_result.get("observation", {})
        done = bool(reset_result.get("done", False))
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            action_dict = get_action(llm_client, obs, step, last_reward)
            action_str = json.dumps(action_dict, separators=(",", ":"))

            step_result = env_step(task_id, action_dict)
            obs = step_result.get("observation", obs)
            reward = float(step_result.get("reward", 0.0) or 0.0)
            done = bool(step_result.get("done", False))
            info = step_result.get("info") or {}

            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            log_step(step=step, action=action_str, reward=reward, done=done, error=None)

            if done:
                score = float(info.get("final_score", reward) or reward)
                break
    except Exception:
        pass

    score = clamp_open_score(score)
    success = score >= 0.5
    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main() -> None:
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    for task_id in TASKS:
        run_task(llm_client, task_id)


if __name__ == "__main__":
    main()