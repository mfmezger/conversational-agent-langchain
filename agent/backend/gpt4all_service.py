"""GPT4ALL Backend Service."""
from gpt4all import Embed4All

text = """"""
embedder = Embed4All()
output = embedder.embed(text)
print(output)


def embedd_text_gpt4all():
    """Embedd text with GPT4ALL."""
    pass


def summarize_text_gpt4all():
    """Summarize text with GPT4ALL."""
    pass


def completion_gpt4all():
    """Complete text with GPT4ALL."""
    pass
