"""This is the utility module."""
from loguru import logger


def combine_text_from_list(input_list: list) -> str:
    """Combines all strings in a list to one string.

    Args:
        input_list (list): List of strings

    Raises:
        TypeError: Input list must contain only strings

    Returns:
        str: Combined string
    """
    # iterate through list and combine all strings to one
    combined_text = ""

    logger.info(f"List: {input_list}")

    for text in input_list:
        # verify that text is a string
        if isinstance(text, str):
            # combine the text in a new line
            combined_text += "\n".join(text)

        else:
            raise TypeError("Input list must contain only strings")

    return combined_text
