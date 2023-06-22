from src.utils.singleton import Singleton
from src.utils.globals import Globals

from src.services.config_service import ConfigService
from src.services.data_service import DataService

from fine_tune.trainer import Trainer

from dataclasses import dataclass

@dataclass
class AppContainer(metaclass=Singleton):
    config_service = ConfigService(
        configs=Globals.project_path.joinpath("src", "configs")
    )

    data_service = DataService(config_service=config_service)

    trainer = Trainer(config_service=config_service, data_service=data_service)