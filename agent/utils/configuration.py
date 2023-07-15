"""Configuration Wrapper."""
from omegaconf import OmegaConf


def load_config(location):
    """Method to load the configuration file.

    :param location: _description_
    :type location: _type_
    """

    def decorator(func):
        """Decorator to load the configuration file."""

        def wrapper(*args, **kwargs):
            """Wrapper to load the configuration file."""
            # Load the config file
            cfg = OmegaConf.load(location)
            return func(cfg=cfg, *args, **kwargs)

        return wrapper

    return decorator
