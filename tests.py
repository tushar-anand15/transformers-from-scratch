import unittest
import torch
from torch.utils.data import DataLoader, Dataset
import nltk
from utils_torch import DataLoaderFactory, TiktokenTokenizer

# Ensure the Brown corpus is downloaded
nltk.download('brown')

class TestDataLoaderOverlap(unittest.TestCase):
    def setUp(self):
        # Prepare the text
        self.text = ' '.join(' '.join(sentence) for sentence in nltk.corpus.brown.sents()[:1000])
        
        # Initialize the tokenizer
        self.tokenizer = TiktokenTokenizer()
        
        # Create DataLoader
        factory = DataLoaderFactory()
        self.dataloader = factory.create_dataloader_v1(self.text, batch_size=4, max_length=256, stride=128)

    def check_overlap(self, input_ids, target_ids):
        # Decode the first input and target sequences from the batch
        decoded_input = self.tokenizer.decode(input_ids.tolist())
        decoded_target = self.tokenizer.decode(target_ids.tolist())
        
        # Print the full decoded input and target texts
        print("Decoded Input:", decoded_input)
        print("Decoded Target:", decoded_target)
        
        # Find the overlap using encoded arrays
        overlap_length = 128  # As defined by the stride
        input_subsequence = input_ids[-overlap_length:].tolist()
        target_subsequence = target_ids[:overlap_length].tolist()
        
        # Check if the subsequence matches
        self.assertEqual(input_subsequence, target_subsequence, "The input and target subsequences do not match.")
        
        # Decode the overlapping portion
        overlap_encoded = input_subsequence
        overlap_decoded = self.tokenizer.decode(overlap_encoded)
        
        # Print the overlapping portion
        print("Overlapping Portion (Encoded):", overlap_encoded)
        print("Overlapping Portion (Decoded):", overlap_decoded)

    def test_overlap(self):
        # Process the first 5 batches
        for i, batch in enumerate(self.dataloader):
            if i >= 5:
                break
            input_ids, target_ids = batch
            self.check_overlap(input_ids[0], target_ids[0])

if __name__ == '__main__':
    unittest.main()