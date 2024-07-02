import yaml
import torch

class BaseConfig:
    """Base Encoder Decoder config"""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class Config(BaseConfig):
    def __init__(self, config_file):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check and set device
        # config['device'] = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        config['device'] = torch.device("cpu")
        config['ckpt_dir'] = config['src_model_name'] + '_to_' + config['tgt_model_name']
        super().__init__(**config)

