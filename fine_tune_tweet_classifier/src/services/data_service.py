from src.utils.singleton import Singleton
from src.utils.globals import Globals

from .config_service import ConfigService

import pandas

class DataService(metaclass=Singleton):
    def __init__(self, config_service: ConfigService):
        self.config_service = config_service
    
    def read_csv(self):
        tweet_data = pandas.read_csv(self.config_service.tweets_csv_path)

        return tweet_data