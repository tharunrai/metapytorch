import os
import json
import asyncio
import textwrap
from typing import Optional
from openai import AsyncOpenAI

from client import DataQualityClient
from models import DataQualityAction

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
BENCHMARK    = "data-quality-env"
TEMPERATURE  = 0.2

TASKS = ["task1_missing_values", "task2_type_errors_duplicates", "task3_outliers_inconsistencies"]

def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    error_val = error if error else "null"
    done_val  = str(done).lower()
    action_safe = action.replace("\n", " ")
    print(f"[STEP] step={step} action={action_safe} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: list[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

SYSTEM_PROMPT = textwrap.dedent("""
You are a data quality expert agent.
Respond with ONLY ONE JSON action object:
{"action_type": "flag_issue", "issue_type": "missing", "row_index": 0, "column": "age"}
When finished finding issues:
{"action_type": "submit"}
""").strip()

async def get_action(client: AsyncOpenAI, obs, step: int, last_reward: float) -> dict:
    prompt = f"Step {step} | Last reward: {last_reward}\nTask: {obs.task_description}\nDataset: {json.dumps(obs.dataset)}\nIssues found: {obs.issues_found_so_far}\nHint: {obs.hint}"
    try:
        resp = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=256,
        )
        text = (resp.choices[0].message.content or "").strip().replace("```json", "").replace("```", "")
        return json.loads(text)
    except Exception:
        return {"action_type": "submit"}

async def run_task(llm_client: AsyncOpenAI, task_id: str):
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
    rewards = []
    steps_taken, score, success = 0, 0.0, False

    try:
        async with DataQualityClient(base_url=ENV_BASE_URL) as env:
            await env.reset()
            result = await env.step(DataQualityAction(action_type="start_task", task_id=task_id))
            obs, done, last_reward = result.observation, result.done, 0.0
            max_steps = 30 # Fallback safety limit

            for step in range(1, max_steps + 1):
                if done: break
                action_dict = await get_action(llm_client, obs, step, last_reward)
                action_str = json.dumps(action_dict)

                result = await env.step(DataQualityAction(**action_dict))
                obs, reward, done = result.observation, result.reward, result.done
                
                rewards.append(reward)
                steps_taken = step
                last_reward = reward

                log_step(step=step, action=action_str, reward=reward, done=done, error=None)
                
                if done:
                    score = reward
                    break

            success = score >= 0.5

    except Exception as e:
        print(f"[DEBUG] Error: {e}", flush=True)

    finally:
        score = min(max(float(score), 0.0), 1.0)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

async def main():
    llm_client = AsyncOpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    for task_id in TASKS:
        await run_task(llm_client, task_id)

if __name__ == "__main__":
    asyncio.run(main())