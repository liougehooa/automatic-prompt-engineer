import os
import yaml


def update_config(config, base_config='configs/default.yaml'):
    # Get default config from yaml
    with open(os.path.join(os.path.dirname(__file__), base_config)) as f:
        default_config = yaml.safe_load(f)

    # Update default config with user config
    # Note that the config is a nested dictionary, so we need to update it recursively
    def update(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    return update(default_config, config)


def simple_config(config_file='configs/bandits.yaml'):
    """Returns a yaml config."""
    with open(os.path.join(os.path.dirname(__file__), config_file)) as f:
        default_config = yaml.safe_load(f)
    return default_config
