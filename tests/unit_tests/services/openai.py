import unittest
from unittest.mock import patch, MagicMock
from open_ai_service import OpenAIService
from omegaconf import DictConfig

class TestOpenAIService(unittest.TestCase):

    def setUp(self):
        self.cfg_mock = MagicMock(spec=DictConfig)
        self.collection_name = "test_collection"
        self.openai_service = OpenAIService(cfg=self.cfg_mock, collection_name=self.collection_name)

    def test_init(self):
        # Test initialization logic here
        self.assertEqual(self.openai_service.collection_name, self.collection_name)
        # Add more assertions as needed

    @patch('open_ai_service.generate_collection')
    @patch('open_ai_service.logger')
    def test_create_collection(self, logger_mock, generate_collection_mock):
        collection_name = "new_collection"
        self.assertTrue(self.openai_service.create_collection(name=collection_name))
        generate_collection_mock.assert_called_once_with(collection_name, self.cfg_mock.openai_embeddings.size)
        logger_mock.info.assert_called_once()

    @patch('open_ai_service.DirectoryLoader')
    @patch('open_ai_service.logger')
    def test_embed_documents(self, logger_mock, directory_loader_mock):
        # Mock the loader and its return value
        directory_loader_mock.return_value.load_and_split.return_value = []
        self.openai_service.embed_documents(directory="dummy_path", file_ending=".pdf")
        directory_loader_mock.assert_called_once()
        logger_mock.info.assert_called()

    @patch('open_ai_service.openai')
    @patch('open_ai_service.generate_prompt')
    def test_summarize_text(self, generate_prompt_mock, openai_mock):
        # Setup mock return values
        generate_prompt_mock.return_value = "Mock Prompt"
        openai_mock.api_key = "dummy_key"
        openai_mock.chat.completions.create.return_value = MagicMock(choices=[MagicMock(messages=MagicMock(content="Mock Summary"))])

        summary = self.openai_service.summarize_text(text="Dummy text")
        generate_prompt_mock.assert_called_once()
        openai_mock.chat.completions.create.assert_called_once()
        self.assertEqual(summary, "Mock Summary")

if __name__ == '__main__':
    unittest.main()
