"""Configuration Wrapper."""
from omegaconf import OmegaConf

def load_config(config_file_path: str):
    """Method to load the configuration file.

    :param config_file_path: The path to the configuration file
    :type config_file_path: str
    :return: A decorator that loads the configuration file
    :rtype: function
    """
    def decorator(func):
        """Decorator to load the configuration file."""
        def wrapper(*args, **kwargs):
            """Wrapper to load the configuration file."""
            try:
                # Load the config file
                cfg = OmegaConf.load(config_file_path)
            except Exception as e:
                raise ValueError(f"Error loading configuration file: {e}")
            return func(cfg, *args, **kwargs)

        return wrapper

    return decorator

    