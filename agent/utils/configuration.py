"""Configuration Wrapper."""
from omegaconf import OmegaConf


def load_config(location):
    """Loads the configuration file.

    Args:
        location (str): The location of the configuration file.

    Returns:
        dict: A dictionary containing the configuration settings.
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
