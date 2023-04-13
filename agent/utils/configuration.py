from omegaconf import OmegaConf


def load_config(config_path):
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Load the config file
            cfg = OmegaConf.load(config_path)
            return func(cfg, *args, **kwargs)

        return wrapper

    return decorator
