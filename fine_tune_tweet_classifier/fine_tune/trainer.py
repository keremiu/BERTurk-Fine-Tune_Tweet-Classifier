from src.services.config_service import ConfigService
from src.services.data_service import DataService

import torch

class Trainer():
    def __init__(self, config_service: ConfigService, data_service: DataService):
        self.config_service = config_service
        self.data_service = data_service

    def train(self):
        ...