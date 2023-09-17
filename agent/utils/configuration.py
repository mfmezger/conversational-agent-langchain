"""Configuration Wrapper."""
from omegaconf import OmegaConf
from typing import Dict, Callable, Tuple
from functools import wraps

def load_config(location: str) -> Callable[[Callable], Callable]:
    """Loads the configuration file.

    Args:
        location (str): The location of the configuration file.

    Returns:
        dict: A dictionary containing the configuration settings.
    """

    def decorator(func: Callable) -> Callable:
        """Decorator to load the configuration file."""
        @wraps(func)
        def wrapper(*args, **kwargs) -> Dict:
            """Wrapper to load the configuration file."""
            # Load the config file
            cfg = OmegaConf.load(location)
            return func(cfg=cfg, *args, **kwargs)

        return wrapper

    return decorator
