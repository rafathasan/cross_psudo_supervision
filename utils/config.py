import yaml
from omegaconf import DictConfig, OmegaConf

class Config:
    def __init__(self, config_file_path):
        self.config_file_path = config_file_path
        with open(self.config_file_path, 'r') as file:
            self.config_object = yaml.safe_load(file)        

    @property
    def config_dict(self) -> DictConfig:
        return OmegaConf.create(self.config_object)
    @property
    def train_config(self):
        return self.config_dict.trainer.train
    @property
    def test_config(self):
        return self.config_dict.trainer.test
    @property
    def datasets_config(self):
        return self.config_dict.datasets