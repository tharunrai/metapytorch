from openenv.core.env_server import create_fastapi_app
from server.environment import DataQualityEnv

app = create_fastapi_app(DataQualityEnv())