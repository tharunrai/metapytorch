---
title: Data Quality Env
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: app.py
pinned: false
---
# 📊 OpenEnv: Data Quality Analyst Environment

An OpenEnv-compliant Reinforcement Learning (RL) environment that simulates a real-world **Data Quality Analyst**. Instead of playing a game, the AI agent must inspect raw, tabular datasets to identify missing values, type errors, duplicates, and statistical outliers.

This environment is fully containerized, deployable to Hugging Face Spaces, and operates within strict compute limits (2 vCPU, 8 GB RAM).

## 🎯 Overview & Motivation
Data cleaning is one of the most time-consuming real-world tasks in data science and software engineering. This environment tests an LLM agent's ability to logically parse JSON-formatted tabular data, understand column context, and deterministically flag or fix data anomalies. 

## 🛠 Tasks & Difficulty Levels
The environment features three programmatic tasks of increasing complexity. Each task features deterministic grading and reproducible scoring.

* **🟢 Easy: `task1_missing_values`**
    * **Objective:** Identify missing (null) values in a small employee dataset (e.g., missing ages or departments).
    * **Grader:** Checks flagged rows/columns against a hidden ground-truth map.
* **🟡 Medium: `task2_type_errors_duplicates`**
    * **Objective:** Inspect a product sales dataset to find incorrect data types (e.g., text in a numerical price column) and exact duplicate rows.
    * **Grader:** Validates the agent's ability to spot systemic formatting issues and repeated data arrays.
* **🔴 Hard: `task3_outliers_inconsistencies`**
    * **Objective:** Analyze a world cities dataset using real-world logic to find statistical outliers (e.g., highly unrealistic populations) and logical contradictions (e.g., "Paris, Germany").
    * **Grader:** Evaluates both mathematical reasoning and real-world geographical/logical context.

## 🧠 Spaces & Specifications

### Observation Space
At each step, the agent receives a state object containing:
* `task_description`: Explicit instructions for the current dataset.
* `dataset`: The raw tabular data formatted as a JSON array of objects.
* `issues_found_so_far`: A list of issues the agent has successfully flagged in previous steps.
* `hint`: Optional contextual clues.

### Action Space
The agent must respond with a strictly formatted JSON action:
* `action_type`: Must be `"flag_issue"`, `"fix_value"`, or `"submit"`.
* `issue_type`: Category of the error (e.g., `"missing"`, `"duplicate"`).
* `row_index`: The 0-based integer index of the problematic row.
* `column`: The name of the specific column containing the error.

### Reward Function
The environment provides **dense, incremental rewards** shaped to guide behavior throughout the trajectory:
* **+0.20 to +0.30:** Successfully flagging a correct, undiscovered issue.
* **-0.10:** False positive (flagging an issue that doesn't exist).
* **-0.05:** Duplicate action (flagging an issue that was already found).
* *Episodes terminate immediately upon issuing the `submit` action, or after a maximum of 30 steps.*

---

## 🚀 Setup & Installation

### Option A: Local Docker (Recommended)
Build and run the OpenEnv server locally:
```bash
# Build the image
docker build -t data-quality-env .

# Run the container (exposes port 7860)
docker run -p 7860:7860 data-quality-env
### Run with Docker (Recommended)
