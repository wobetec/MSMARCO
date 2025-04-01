import unittest
from src.datasets import MSMarcoDataset, PreProcessor


class TestMSMarcoDataset(unittest.TestCase):
    def test_load_data(self):
        dataset = MSMarcoDataset(data_folder='data/subset_msmarco_train_0')
        dataset.load_data(input_file='subset_msmarco_train_0.01_9.pkl')
        
        self.assertTrue(len(dataset.queries) > 0, "Queries should be loaded.")
        self.assertTrue(len(dataset.documents) > 0, "Documents should be loaded.")
        self.assertTrue(len(dataset.qrels) > 0, "Qrels should be loaded.")

        self.assertIsInstance(dataset.queries, dict, "Queries should be a dictionary.")
        self.assertIsInstance(dataset.queries[list(dataset.queries.keys())[0]], str, "Query should be a str.")

        self.assertIsInstance(dataset.documents, dict, "Documents should be a dictionary.")
        self.assertIsInstance(dataset.documents[list(dataset.documents.keys())[0]], str, "Document should be a str.")

    def test_split_data(self):
        dataset = MSMarcoDataset(data_folder='data/subset_msmarco_train_0')
        dataset.load_data(input_file='subset_msmarco_train_0.01_9.pkl')
        dataset.split_data(split_ratio=0.8, seed=42)
        
        self.assertTrue(len(dataset.train_query_ids) > 0, "Train query IDs should be split.")
        self.assertTrue(len(dataset.test_query_ids) > 0, "Test query IDs should be split.")
        self.assertNotEqual(dataset.train_query_ids, dataset.test_query_ids, "Train and test query IDs should be different.")

        self.assertEqual(len(dataset.train_query_ids) + len(dataset.test_query_ids), len(dataset.queries), "Train and test query IDs should sum to total queries.")
    

class TestPreProcessor(unittest.TestCase):
    #------------------------------------------------------------#
    # Main methods
    #------------------------------------------------------------#
    def test_preprocess_text(self):
        processor = PreProcessor(pipeline=[PreProcessor.lowercase])
        text = "Hello World!"
        processed_text = processor.preprocess_text(text)
        
        self.assertEqual(processed_text, "hello world!", "Text should be lowercased.")
    
    def test_preprocess(self):
        processor = PreProcessor(pipeline=[PreProcessor.lowercase])
        texts = ["Hello World!", "Python is great!"]
        processed_texts = processor.preprocess(texts)
        
        self.assertEqual(processed_texts, ["hello world!", "python is great!"], "Texts should be lowercased.")

    #------------------------------------------------------------#
    # Processing methods
    #------------------------------------------------------------#
    def test_lowercase(self):
        text = "Hello World!"
        processed_text = PreProcessor.lowercase(text)
        
        self.assertEqual(processed_text, "hello world!", "Text should be lowercased.")
    
    def test_remove_punctuation(self):
        text = "Hello, World!"
        processed_text = PreProcessor.remove_punctuation(text)
        
        self.assertEqual(processed_text, "Hello World", "Punctuation should be removed.")

    def test_remove_stopwords(self):
        text = "This is a test sentence."
        processed_text = PreProcessor.lowercase(text)
        processed_text = PreProcessor.remove_stopwords(processed_text)
        
        self.assertEqual(processed_text, "this test sentence.", "Stopwords should be removed.")

    def test_stem(self):
        text = "running runners ran"
        processed_text = PreProcessor.stem(text)
        
        self.assertEqual(processed_text, "run runner ran", "Stemming should be applied.")
