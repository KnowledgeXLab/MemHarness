from contextlib import contextmanager
import os
import time
from agentenv.envs import (
    AcademiaEnvClient,
    AlfWorldEnvClient,
    BabyAIEnvClient,
    MazeEnvClient,
    MovieEnvClient,
    SciworldEnvClient,
    SheetEnvClient,
    SqlGymEnvClient,
    TextCraftEnvClient,
    TodoEnvClient,
    WeatherEnvClient,
    WebarenaEnvClient,
    WebshopEnvClient,
    WordleEnvClient,
    SearchQAEnvClient,
)

def init_env_client(args):
    # task_name - task dict
    envclient_classes = {
        "webshop": WebshopEnvClient,
        "alfworld": AlfWorldEnvClient,
        "babyai": BabyAIEnvClient,
        "sciworld": SciworldEnvClient,
        "textcraft": TextCraftEnvClient,
        "webarena": WebarenaEnvClient,
        "sqlgym": SqlGymEnvClient,
        "maze": MazeEnvClient,
        "wordle": WordleEnvClient,
        "weather": WeatherEnvClient,
        "todo": TodoEnvClient,
        "movie": MovieEnvClient,
        "sheet": SheetEnvClient,
        "academia": AcademiaEnvClient,
        "searchqa": SearchQAEnvClient,
    }
    # select task according to the name
    envclient_class = envclient_classes.get(args.task_name.lower(), None)
    if envclient_class is None:
        raise ValueError(f"Unsupported task name: {args.task_name}")
    retry = 0
    while True:
        try:
            env_client = envclient_class(env_server_base=args.env_addr, data_len=1, timeout=2400)
            break
        except Exception as e:
            retry += 1
            print(f"Failed to connect to env server, retrying...({retry}/{args.max_retries})")
            if retry > args.max_retries:
                raise e
            time.sleep(5)
    return env_client