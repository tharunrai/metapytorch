from openenv.core.http_env_client import HTTPEnvClient
from openenv.core.types import StepResult
from models import DataQualityAction, DataQualityObservation, DataQualityState

class DataQualityClient(HTTPEnvClient[DataQualityAction, DataQualityObservation]):
    def _step_payload(self, action: DataQualityAction) -> dict:
        return action.model_dump()
    
    def _parse_result(self, payload: dict) -> StepResult:
        obs_data = payload.get("observation", payload)
        return StepResult(
            observation=DataQualityObservation(**obs_data),
            reward=float(payload.get("reward", 0.0)),
            done=bool(payload.get("done", False))
        )
        
    def _parse_state(self, payload: dict) -> DataQualityState:
        return DataQualityState(**payload)