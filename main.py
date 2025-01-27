
# main.py

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
    vocabulary[0] = '<UNK>'  # Ensure <UNK> is at index 0
    
    # Create a reverse mapping for word to index
    word_to_index = {word: idx for idx, word in vocabulary.items()}
    
    return word_to_index

def main():
    # Example usage

    text = ' '.join(' '.join(sentence) for sentence in nltk.corpus.brown.sents()[:1000])
    
    # Using tiktoken tokenizer
    tiktoken_tokenizer = TiktokenTokenizer()
    encoded_text = tiktoken_tokenizer.encode(text)
    print("Tiktoken Tokenizer Encoding:", encoded_text)

    # Decode the encoded text
    decoded_text = tiktoken_tokenizer.decode(encoded_text)
    print("Decoded Text:", decoded_text[:500])  # Print the first 500 characters for brevity

    # Create DataLoader using tiktoken tokenizer
    factory = DataLoaderFactory()
    dataloader = factory.create_dataloader_v1(text, batch_size=4, max_length=256, stride=128)
    
    # Process only the first batch
    for i, batch in enumerate(dataloader):
        input_ids, target_ids = batch
        print("Batch from DataLoader (input_ids):", input_ids)
        print("Batch from DataLoader (target_ids):", target_ids)
        
        # Decode the first two sequences from the first batch
        for j in range(2):
            decoded_batch = tiktoken_tokenizer.decode(input_ids[j].tolist())
            print(f"Decoded Batch {j}:", decoded_batch[:500])  # Print the first 500 characters for brevity
        
        break  # Only process the first batch

if __name__ == "__main__":
    main()