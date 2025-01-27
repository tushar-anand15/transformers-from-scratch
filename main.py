from utils_torch import DataLoaderFactory, TiktokenTokenizer
import nltk

def create_vocabulary(corpus_name='brown'):
    nltk.download(corpus_name)
    
    if corpus_name == 'brown':
        words = nltk.corpus.brown.words()
    elif corpus_name == 'reuters':
        nltk.download('reuters')
        words = nltk.corpus.reuters.words()
    elif corpus_name == 'gutenberg':
        nltk.download('gutenberg')
        words = nltk.corpus.gutenberg.words()
    else:
        raise ValueError(f"Unsupported corpus: {corpus_name}")
    
    unique_words = sorted(set(word.lower() for word in words if word.isalpha()))
    vocabulary = {idx + 1: word for idx, word in enumerate(unique_words)}
    vocabulary[0] = '<UNK>'  
    
    word_to_index = {word: idx for idx, word in vocabulary.items()}
    
    return word_to_index

def main():
    # Example usage

    text = ' '.join(' '.join(sentence) for sentence in nltk.corpus.brown.sents()[:1000])
    
    # Using tiktoken tokenizer
    tiktoken_tokenizer = TiktokenTokenizer()
    encoded_text = tiktoken_tokenizer.encode(text)
    print("Tiktoken Tokenizer Encoding:", encoded_text)

    # Create DataLoader using tiktoken tokenizer
    factory = DataLoaderFactory()
    dataloader = factory.create_dataloader_v1(text, batch_size=4, max_length=256, stride=128)
    
    # Process only the first batch
    for i, batch in enumerate(dataloader):
        input_ids, target_ids = batch
        
        # Decode the first input and target sequences from the first batch
        decoded_input = tiktoken_tokenizer.decode(input_ids[0].tolist())
        decoded_target = tiktoken_tokenizer.decode(target_ids[0].tolist())
        
        # Print the full decoded input and target texts
        print("Decoded Input 0:", decoded_input)
        print("Decoded Target 0:", decoded_target)
        
        # Find the overlap using encoded arrays
        overlap_length = 128  # As defined by the stride
        input_subsequence = input_ids[0][-overlap_length:].tolist()
        target_subsequence = target_ids[0][:overlap_length].tolist()
        
        # Check if the subsequence matches
        if input_subsequence == target_subsequence:
            overlap_encoded = input_subsequence
            overlap_decoded = tiktoken_tokenizer.decode(overlap_encoded)
            print("Overlapping Portion (Encoded):", overlap_encoded)
            print("Overlapping Portion (Decoded):", overlap_decoded)
        else:
            print("No exact overlap found between input and target subsequences.")
        
        break  # Only process the first batch

if __name__ == "__main__":
    main()