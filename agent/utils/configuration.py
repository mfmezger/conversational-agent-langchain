from omegaconf import OmegaConf


def load_config(location):
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Load the config file
            cfg = OmegaConf.load(location)
            return func(cfg, *args, **kwargs)

        return wrapper

    return decorator
