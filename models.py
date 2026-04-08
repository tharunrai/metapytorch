from typing import Optional, Any, List, Dict
from pydantic import BaseModel
from openenv_core.env_server import Action, Observation, State

class DataQualityAction(BaseModel, Action):
    action_type: str 
    task_id: Optional[str] = None
    issue_type: Optional[str] = None
    row_index: Optional[int] = None
    column: Optional[str] = None

class DataQualityObservation(BaseModel, Observation):
    task_id: str
    task_description: str
    dataset: List[Dict[str, Any]]
    columns: List[str]
    step_number: int
    issues_found_so_far: List[str]
    hint: Optional[str] = None

class DataQualityState(BaseModel, State):
    task_id: str
    step_number: int
    issues_found: List[str]
    total_issues_in_task: int
    score_so_far: float
    done: bool