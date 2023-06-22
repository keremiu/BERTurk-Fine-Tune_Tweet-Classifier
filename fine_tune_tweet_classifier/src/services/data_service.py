from src.utils.singleton import Singleton
from src.utils.globals import Globals

from .config_service import ConfigService

import pandas

class DataService(metaclass=Singleton):
    def __init__(self, config_service: ConfigService):
        self.config_service = config_service
    
    def read_training_tweets(self) -> pandas.DataFrame:
        labeled_tweets = pandas.read_csv(self.config_service.training_tweets_csv_path)

        if not {"text", "label"}.issubset(labeled_tweets.columns):
            raise Exception("The dataset must contain the columns 'text' and 'label'")

        return labeled_tweets
    
    def read_test_tweets(self) -> pandas.DataFrame:
        labeled_tweets = pandas.read_csv(self.config_service.test_tweets_csv_path)

        if not {"text", "label"}.issubset(labeled_tweets.columns):
            raise Exception("The dataset must contain the columns 'text' and 'label'")

        return labeled_tweets