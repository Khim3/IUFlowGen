import yaml
import os

def load_config(file_path = '/home/tttung/Khiem/thesis/config.yaml'):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)