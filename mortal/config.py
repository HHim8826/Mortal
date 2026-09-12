import toml
import os

config_file = os.environ.get('MORTAL_CFG', 'config.toml')
with open(config_file, encoding='utf-8') as f:
    config = toml.load(f)

# Online training runs the trainer and the self-play workers side by side on
# one box, from one config but not on one GPU. This is how each says which.
if device := os.environ.get('MORTAL_DEVICE'):
    config['control']['device'] = device
