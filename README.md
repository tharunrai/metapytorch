---
title: Data Quality Env
emoji: "📊"
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: app.py
pinned: false
---
# OpenEnv Data Quality Analyst Environment

This project is a real-world OpenEnv benchmark where an agent performs data quality analysis on tabular datasets. Instead of a game, the agent must find missing values, type errors, duplicates, outliers, and logical inconsistencies.

The environment is fully containerized for Hugging Face Spaces and local Docker execution.

## Why This Environment

Data quality checking is a real production workflow used by data teams, analytics teams, and ML teams. This benchmark evaluates whether an agent can:

- read structured tabular data,
- reason over domain constraints,
- incrementally identify issues,
- avoid false positives,
- and submit a final answer with high precision and recall.

## OpenEnv Interface

The environment follows the OpenEnv style with typed models and a standard API.

- `reset()` -> initial observation
- `step(action)` -> observation, reward, done, info
- `state()` -> current episode state

Implemented HTTP endpoints:

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /tasks`
- `GET /health`
- `GET /docs`

## Observation Space

Each observation contains:

- `task_id`
- `task_description`
- `dataset` (list of row objects)
- `columns`
- `step_number`
- `issues_found_so_far`
- `hint`

## Action Space

Action object fields:

- `action_type` in `{flag_issue, fix_value, submit}`
- `task_id` (optional)
- `issue_type` in `{missing, type_error, duplicate, outlier, inconsistency}`
- `row_index` (0-based)
- `column`
- `fixed_value` (used with `fix_value`)

## Tasks and Difficulty

Three deterministic tasks with increasing difficulty:

1. `task1_missing_values` (easy)
- Goal: identify all null/missing cells in employee records.
- Max steps: 20.

2. `task2_type_errors_duplicates` (medium)
- Goal: find invalid types/values and duplicate records in product sales data.
- Max steps: 25.

3. `task3_outliers_inconsistencies` (hard)
- Goal: detect statistical outliers and logical inconsistencies in world cities data.
- Max steps: 30.

## Reward Design

The reward is shaped across the trajectory:

- positive reward for correct unseen issues,
- negative reward for false positives,
- negative reward for duplicate reports,
- bonus for valid fix actions,
- final score based on precision, recall, and efficiency.

Scores are clamped strictly inside `(0.0, 1.0)`.

## Project Structure

```text
.
├── client.py
├── inference.py
├── models.py
├── openenv.yaml
├── requirements.txt
├── Dockerfile
└── server/
    ├── app.py
    └── environment.py
```

## Local Setup (Docker)

Build image:

```bash
docker build -t data-quality-env .
```

Run container (port 7860):

```bash
docker run -p 7860:7860 data-quality-env
```

Quick checks:

```bash
curl http://localhost:7860/health
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{}'
```

Open API docs in browser:

- `http://localhost:7860/docs`

## Baseline Inference Script

The required baseline script is at project root:

- `inference.py`

Required environment variables:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN` (or `OPENAI_API_KEY`)
- `ENV_BASE_URL` (defaults to `http://localhost:7860`)

Run baseline:

```bash
export HF_TOKEN=<your_token>
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export API_BASE_URL=https://router.huggingface.co/v1
export ENV_BASE_URL=http://localhost:7860

python inference.py
```

The script emits strict structured logs:

- `[START] task=<task> env=<benchmark> model=<model>`
- `[STEP] step=<n> action=<json> reward=<0.00> done=<true|false> error=<msg|null>`
- `[END] success=<true|false> steps=<n> score=<0-1> rewards=<r1,r2,...>`

## Reproducible Baseline Snapshot

Example baseline run (Qwen/Qwen2.5-72B-Instruct):

| Task | Score |
|---|---:|
| task1_missing_values | 0.142 |
| task2_type_errors_duplicates | 0.142 |
| task3_outliers_inconsistencies | 0.142 |

## Hugging Face Spaces Deployment

1. Push this repo to a Docker Space.
2. Ensure Space secrets/variables are set (for inference use):
- `HF_TOKEN`
- `MODEL_NAME`
- `API_BASE_URL`
- `ENV_BASE_URL` (if needed)
3. Confirm health endpoint:
- `https://<space>.hf.space/health`

## Pre-Submission Checklist

- Space responds to `POST /reset` with HTTP 200.
- Docker image builds from repo root.
- `inference.py` runs and logs START/STEP/END format.
- Three tasks available with deterministic scoring strictly in `(0.0, 1.0)`.
- `openenv.yaml` matches implementation.
