from src.utils.singleton import Singleton

import pathlib
import yaml
import os

class ConfigService(metaclass=Singleton):
    def __init__(self, configs: pathlib.Path):
        self.config = {} 

        # Append the contents of every .yaml file in configs directory into self.config dictionary 
        for config_file in os.listdir(configs):
            config_name = config_file[:-5]  # Remove .yaml suffix
            config_path = configs.joinpath(config_file) 
            
            with open(config_path, "r") as cf:
                self.config[config_name] = yaml.safe_load(cf)

    @property
    def training_tweets_csv_path(self):
        return pathlib.Path(self.config["config"]["training_tweets_csv_path"])

    @property
    def test_tweets_csv_path(self):
        return pathlib.Path(self.config["config"]["test_tweets_csv_path"])
    
    @property
    def model_path(self):
        try:
            return pathlib.Path(self.config["config"]["model_path"])
        except TypeError:
            return None
    
    
    @property
    def device(self):
        return self.config["config"]["device"]

    @property
    def bert_model_name(self):
        return self.config["config"]["model_parameters"]["bert_model_name"]

    @property
    def layers(self):
        return self.config["config"]["model_parameters"]["layers"]

    @property
    def validation_size(self):
        return self.config["config"]["training_parameters"]["validation_size"]
    
    @property
    def batch_size(self):
        return self.config["config"]["training_parameters"]["batch_size"]
    
    @property
    def learning_rate(self):
        return self.config["config"]["training_parameters"]["learning_rate"]

    @property
    def class_weights(self):
        return self.config["config"]["training_parameters"]["class_weights"]

    @property
    def optimizer(self):
        return self.config["config"]["training_parameters"]["optimizer"]
    
    @property
    def lr_scheduler(self):
        return self.config["config"]["training_parameters"]["scheduler"]
    
    @property
    def num_epochs(self):
        return self.config["config"]["training_parameters"]["num_epochs"]
    
    @property
    def threshold(self):
        return self.config["config"]["evaluation_parameters"]["threshold"]
