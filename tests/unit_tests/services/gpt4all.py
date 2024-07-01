import unittest
from unittest.mock import patch, MagicMock
from gpt4all_service import GPT4AllService
from omegaconf import DictConfig

class TestGPT4AllService(unittest.TestCase):

    def setUp(self):
        self.cfg_mock = MagicMock(spec=DictConfig)
        self.collection_name = "test_collection"
        self.gpt4all_service = GPT4AllService(cfg=self.cfg_mock, collection_name=self.collection_name)

    def test_init(self):
        self.assertEqual(self.gpt4all_service.collection_name, self.collection_name)
        # Add more assertions as needed

    @patch('gpt4all_service.generate_collection')
    def test_create_collection(self, generate_collection_mock):
        collection_name = "new_collection"
        self.assertTrue(self.gpt4all_service.create_collection(name=collection_name))
        generate_collection_mock.assert_called_once_with(collection_name, self.cfg_mock.gpt4all_embeddings.size)

    @patch('gpt4all_service.DirectoryLoader')
    @patch('gpt4all_service.logger')
    def test_embed_documents(self, logger_mock, directory_loader_mock):
        directory_loader_mock.return_value.load_and_split.return_value = []
        self.gpt4all_service.embed_documents(directory="dummy_path", file_ending=".pdf")
        directory_loader_mock.assert_called_once()
        logger_mock.info.assert_called()

    @patch('gpt4all_service.GPT4All')
    @patch('gpt4all_service.generate_prompt')
    def test_summarize_text(self, generate_prompt_mock, gpt4all_mock):
        generate_prompt_mock.return_value = "Mock Prompt"
        gpt4all_instance = gpt4all_mock.return_value
        gpt4all_instance.generate.return_value = "Mock Summary"
        summary = self.gpt4all_service.summarize_text(text="Dummy text")
        generate_prompt_mock.assert_called_once()
        gpt4all_instance.generate.assert_called_once_with("Mock Prompt", max_tokens=300)
        self.assertEqual(summary, "Mock Summary")

if __name__ == '__main__':
    unittest.main()
