"""
DataQualityEnv — OpenEnv environment for data quality checking tasks.
An AI agent inspects a dataset and identifies/fixes data quality issues.
"""

import json
import random
from typing import Any, Optional
from pydantic import BaseModel, ValidationError
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import RedirectResponse
import uvicorn

SCORE_EPSILON = 1e-3


def clamp_open_unit_interval(value: float, epsilon: float = SCORE_EPSILON) -> float:
    return min(max(float(value), epsilon), 1.0 - epsilon)

# ─── Pydantic Models ────────────────────────────────────────────────────────

class Observation(BaseModel):
    task_id: str
    task_description: str
    dataset: list[dict]
    columns: list[str]
    step_number: int
    issues_found_so_far: list[str]
    hint: Optional[str] = None

class Action(BaseModel):
    action_type: str  # "flag_issue", "fix_value", "submit"
    task_id: Optional[str] = None
    issue_type: Optional[str] = None   # "missing", "duplicate", "type_error", "outlier", "inconsistency"
    row_index: Optional[int] = None
    column: Optional[str] = None
    description: Optional[str] = None
    fixed_value: Optional[Any] = None

class Reward(BaseModel):
    value: float
    reason: str

class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict

class ResetResult(BaseModel):
    observation: Observation
    done: bool

class StateResult(BaseModel):
    task_id: str
    step_number: int
    issues_found: list[str]
    total_issues_in_task: int
    score_so_far: float
    done: bool


# ─── Dataset Generators ─────────────────────────────────────────────────────

def make_task1_dataset():
    """Easy: Find missing values."""
    data = [
        {"id": 1, "name": "Alice",  "age": 30,   "email": "alice@example.com",  "salary": 50000},
        {"id": 2, "name": "Bob",    "age": None,  "email": "bob@example.com",    "salary": 62000},
        {"id": 3, "name": None,     "age": 25,    "email": "carol@example.com",  "salary": 47000},
        {"id": 4, "name": "David",  "age": 41,    "email": None,                 "salary": 71000},
        {"id": 5, "name": "Eve",    "age": 35,    "email": "eve@example.com",    "salary": None},
        {"id": 6, "name": "Frank",  "age": 28,    "email": "frank@example.com",  "salary": 53000},
        {"id": 7, "name": "Grace",  "age": None,  "email": None,                 "salary": 66000},
        {"id": 8, "name": "Hank",   "age": 33,    "email": "hank@example.com",   "salary": 58000},
    ]
    # Ground truth: all None cells are issues
    issues = []
    for row in data:
        for col, val in row.items():
            if val is None:
                issues.append(f"missing:row{row['id']}:{col}")
    return data, issues

def make_task2_dataset():
    """Medium: Find type errors and duplicate rows."""
    data = [
        {"id": 1,  "product": "Widget A",  "price": 19.99,     "quantity": 100,  "date": "2024-01-15"},
        {"id": 2,  "product": "Widget B",  "price": "twenty",  "quantity": 50,   "date": "2024-01-16"},
        {"id": 3,  "product": "Widget C",  "price": 34.50,     "quantity": "abc","date": "2024-01-17"},
        {"id": 4,  "product": "Widget A",  "price": 19.99,     "quantity": 100,  "date": "2024-01-15"},  # dup of row 1
        {"id": 5,  "product": "Widget D",  "price": 9.99,      "quantity": 200,  "date": "not-a-date"},
        {"id": 6,  "product": "Widget E",  "price": 44.00,     "quantity": 75,   "date": "2024-01-19"},
        {"id": 7,  "product": "Widget B",  "price": "twenty",  "quantity": 50,   "date": "2024-01-16"},  # dup of row 2
        {"id": 8,  "product": "Widget F",  "price": -5.00,     "quantity": 30,   "date": "2024-01-20"},
    ]
    issues = [
        "type_error:row2:price",       # "twenty" not a number
        "type_error:row3:quantity",    # "abc" not a number
        "type_error:row5:date",        # "not-a-date" invalid date
        "type_error:row8:price",       # negative price is invalid
        "duplicate:row4",              # duplicate of row 1
        "duplicate:row7",              # duplicate of row 2
    ]
    return data, issues

def make_task3_dataset():
    """Hard: Find outliers, inconsistencies, and fix them."""
    data = [
        {"id": 1,  "city": "New York",    "country": "USA",     "population": 8336817,  "gdp_per_capita": 85000,  "currency": "USD"},
        {"id": 2,  "city": "London",      "country": "UK",      "population": 9002488,  "gdp_per_capita": 47000,  "currency": "GBP"},
        {"id": 3,  "city": "Tokyo",       "country": "Japan",   "population": 13960000, "gdp_per_capita": 42000,  "currency": "JPY"},
        {"id": 4,  "city": "Paris",       "country": "Germany", "population": 2161000,  "gdp_per_capita": 55000,  "currency": "EUR"},  # Paris is in France not Germany
        {"id": 5,  "city": "Sydney",      "country": "Australia","population": 5312000, "gdp_per_capita": 54000,  "currency": "AUD"},
        {"id": 6,  "city": "Mumbai",      "country": "India",   "population": 99999999, "gdp_per_capita": 2200,   "currency": "INR"},  # outlier population
        {"id": 7,  "city": "Berlin",      "country": "Germany", "population": 3769000,  "gdp_per_capita": 41000,  "currency": "USD"},  # Germany uses EUR not USD
        {"id": 8,  "city": "Toronto",     "country": "Canada",  "population": 2930000,  "gdp_per_capita": 52000,  "currency": "CAD"},
        {"id": 9,  "city": "Dubai",       "country": "UAE",     "population": 3331000,  "gdp_per_capita": 43000,  "currency": "AED"},
        {"id": 10, "city": "São Paulo",   "country": "Brazil",  "population": 12325000, "gdp_per_capita": -1000,  "currency": "BRL"},  # negative gdp is invalid
    ]
    issues = [
        "inconsistency:row4:country",      # Paris → France not Germany
        "outlier:row6:population",         # 99999999 is unrealistic
        "inconsistency:row7:currency",     # Germany uses EUR not USD
        "outlier:row10:gdp_per_capita",    # negative gdp invalid
    ]
    return data, issues


TASKS = {
    "task1_missing_values": {
        "description": (
            "EASY TASK: You are a data quality agent. The dataset below contains employee records. "
            "Some cells have missing (null/None) values. "
            "Your job: flag every missing value by calling action_type='flag_issue' with issue_type='missing', "
            "the row_index (0-based), and the column name. "
            "When done, call action_type='submit'."
        ),
        "generator": make_task1_dataset,
        "max_steps": 20,
        "difficulty": "easy",
    },
    "task2_type_errors_duplicates": {
        "description": (
            "MEDIUM TASK: You are a data quality agent. The dataset below contains product sales records. "
            "There are type errors (wrong data types, invalid values) and duplicate rows. "
            "Flag each issue with action_type='flag_issue', specifying issue_type as 'type_error' or 'duplicate', "
            "the row_index (0-based), and the column (for type errors). "
            "When done, call action_type='submit'."
        ),
        "generator": make_task2_dataset,
        "max_steps": 25,
        "difficulty": "medium",
    },
    "task3_outliers_inconsistencies": {
        "description": (
            "HARD TASK: You are a data quality agent. The dataset below contains world cities data. "
            "There are outliers (statistically extreme values) and inconsistencies (logically contradictory data). "
            "Flag each issue with action_type='flag_issue', specifying issue_type as 'outlier' or 'inconsistency', "
            "the row_index (0-based), and the column. "
            "You may also fix values using action_type='fix_value' with a fixed_value. "
            "When done, call action_type='submit'."
        ),
        "generator": make_task3_dataset,
        "max_steps": 30,
        "difficulty": "hard",
    },
}


# ─── Environment Class ──────────────────────────────────────────────────────

class DataQualityEnv:
    def __init__(self, task_id: str = "task1_missing_values"):
        assert task_id in TASKS, f"Unknown task: {task_id}. Choose from {list(TASKS.keys())}"
        self.task_id = task_id
        self.task_config = TASKS[task_id]
        self._reset_state()

    def _reset_state(self):
        self.dataset, self.ground_truth_issues = self.task_config["generator"]()
        self.step_number = 0
        self.issues_found: list[str] = []
        self.false_positives: list[str] = []
        self.fixes_applied: list[str] = []
        self.done = False
        self.cumulative_reward = 0.0
        self.max_steps = self.task_config["max_steps"]

    def _make_observation(self, hint: str = None) -> Observation:
        return Observation(
            task_id=self.task_id,
            task_description=self.task_config["description"],
            dataset=self.dataset,
            columns=list(self.dataset[0].keys()) if self.dataset else [],
            step_number=self.step_number,
            issues_found_so_far=list(self.issues_found),
            hint=hint,
        )

    def reset(self) -> ResetResult:
        self._reset_state()
        return ResetResult(observation=self._make_observation(), done=False)

    def state(self) -> StateResult:
        return StateResult(
            task_id=self.task_id,
            step_number=self.step_number,
            issues_found=list(self.issues_found),
            total_issues_in_task=len(self.ground_truth_issues),
            score_so_far=round(clamp_open_unit_interval(self.cumulative_reward), 4),
            done=self.done,
        )

    def step(self, action: Action) -> StepResult:
        if self.done:
            final_score = self._compute_final_score()
            return StepResult(
                observation=self._make_observation(hint="Episode already done. Call reset()."),
                reward=final_score,
                done=True,
                info={"error": "already_done", "final_score": final_score},
            )

        self.step_number += 1
        reward = 0.0
        hint = None
        info = {}

        # ── Timeout penalty ──
        if self.step_number > self.max_steps:
            self.done = True
            final_score = self._compute_final_score()
            return StepResult(
                observation=self._make_observation(hint="Max steps reached."),
                reward=final_score,
                done=True,
                info={"reason": "max_steps_exceeded", "final_score": final_score},
            )

        if action.action_type == "submit":
            self.done = True
            final_score = self._compute_final_score()
            recall = len([i for i in self.ground_truth_issues if i in self.issues_found]) / max(len(self.ground_truth_issues), 1)
            precision = len([i for i in self.issues_found if i in self.ground_truth_issues]) / max(len(self.issues_found), 1) if self.issues_found else 0.0
            f1 = 2 * precision * recall / max(precision + recall, 1e-9)
            reward = final_score
            hint = f"Episode complete. Precision={precision:.2f} Recall={recall:.2f} F1={f1:.2f}"
            info = {"final_score": final_score, "precision": precision, "recall": recall, "f1": f1}

        elif action.action_type == "flag_issue":
            issue_key = self._build_issue_key(action)
            if issue_key in self.issues_found:
                reward = -0.05  # already flagged
                hint = "Already flagged this issue."
                info = {"error": "duplicate_flag"}
            elif issue_key in self.ground_truth_issues:
                self.issues_found.append(issue_key)
                progress = len(self.issues_found) / len(self.ground_truth_issues)
                reward = 0.2 + 0.1 * progress  # partial reward, grows as more found
                hint = f"Correct! ({len(self.issues_found)}/{len(self.ground_truth_issues)} issues found)"
                info = {"correct": True, "issue_key": issue_key}
            else:
                self.false_positives.append(issue_key)
                reward = -0.1  # false positive penalty
                hint = "Incorrect — that is not a real issue."
                info = {"correct": False, "false_positive": issue_key}

        elif action.action_type == "fix_value":
            fix_key = f"fix:row{(action.row_index or 0)+1}:{action.column}"
            relevant_issue = self._build_issue_key(action)
            if relevant_issue in self.ground_truth_issues:
                self.fixes_applied.append(fix_key)
                reward = 0.15  # bonus for actually fixing
                hint = "Fix applied successfully."
                info = {"fixed": True}
            else:
                reward = -0.05
                hint = "Fix attempted on a non-issue cell."
                info = {"fixed": False}
        else:
            reward = -0.02
            hint = f"Unknown action_type '{action.action_type}'. Use: flag_issue, fix_value, submit."
            info = {"error": "unknown_action"}

        self.cumulative_reward = clamp_open_unit_interval(self.cumulative_reward + reward)
        return StepResult(
            observation=self._make_observation(hint=hint),
            reward=round(reward, 4),
            done=self.done,
            info=info,
        )

    def _build_issue_key(self, action: Action) -> str:
        row_num = (action.row_index or 0) + 1  # convert 0-based to 1-based id
        if action.issue_type in ("duplicate",):
            return f"duplicate:row{row_num}"
        return f"{action.issue_type}:row{row_num}:{action.column}"

    def _compute_final_score(self) -> float:
        true_positives = [i for i in self.issues_found if i in self.ground_truth_issues]
        recall = len(true_positives) / max(len(self.ground_truth_issues), 1)
        precision = len(true_positives) / max(len(self.issues_found), 1) if self.issues_found else 0.0
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        # Step efficiency bonus: fewer steps = slightly higher score
        efficiency = max(0.0, 1.0 - self.step_number / (self.max_steps * 2))
        score = 0.85 * f1 + 0.15 * efficiency
        return round(clamp_open_unit_interval(score), 4)


# ─── FastAPI App ─────────────────────────────────────────────────────────────

app = FastAPI(title="DataQualityEnv", description="OpenEnv data quality checking environment")

_envs: dict[str, DataQualityEnv] = {}

def get_env(task_id: str) -> DataQualityEnv:
    if task_id not in TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task_id '{task_id}'. Choose from {list(TASKS.keys())}.",
        )
    if task_id not in _envs:
        _envs[task_id] = DataQualityEnv(task_id)
    return _envs[task_id]

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)

@app.post("/reset")
def reset(body: Optional[dict] = None):
    payload = body or {}
    task_id = payload.get("task_id", "task1_missing_values")
    if not isinstance(task_id, str):
        raise HTTPException(status_code=422, detail="`task_id` must be a string.")
    env = get_env(task_id)
    result = env.reset()
    return result.model_dump()

@app.post("/step")
def step(body: Optional[dict] = None):
    payload = body or {}
    action_payload = payload.get("action", payload)

    if not isinstance(action_payload, dict):
        raise HTTPException(status_code=422, detail="`action` must be an object.")

    task_id = payload.get("task_id") or action_payload.get("task_id") or "task1_missing_values"
    if not isinstance(task_id, str):
        raise HTTPException(status_code=422, detail="`task_id` must be a string.")

    env = get_env(task_id)
    try:
        action = Action(**action_payload)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc

    result = env.step(action)
    return result.model_dump()

@app.get("/state")
def state(task_id: str = "task1_missing_values"):
    env = get_env(task_id)
    return env.state().model_dump()

@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {"task_id": tid, "difficulty": cfg["difficulty"], "description": cfg["description"][:120] + "..."}
            for tid, cfg in TASKS.items()
        ]
    }

@app.get("/health")
def health():
    return {"status": "ok", "env": "DataQualityEnv"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
