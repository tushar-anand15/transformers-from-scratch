# utils_custom.py

from collections import defaultdict

class CustomTokenizer:
    def __init__(self, vocabulary):
        self.word_to_index = vocabulary
        self.index_to_word = {idx: word for word, idx in vocabulary.items()}
        self.bpe_merges = []  # Store the merge operations for BPE

    def encode(self, text):
        words = text.split()
        return [self.word_to_index.get(word.lower(), self.word_to_index['<UNK>']) for word in words]

    def decode(self, indices):
        return ' '.join(self.index_to_word.get(idx, '<UNK>') for idx in indices)

    
    
    
    
    def byte_pair_encoding(self, text, num_merges=10):
        vocab = defaultdict(int)
        for word in text.split():
            word = ' '.join(list(word)) + ' </w>'
            vocab[word] += 1

        for _ in range(num_merges):
            pairs = self.get_stats(vocab)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            vocab = self.merge_vocab(best, vocab)
            self.bpe_merges.append(best)

        return vocab

    def get_stats(self, vocab):
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def merge_vocab(self, pair, vocab):
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        new_vocab = {}
        for word in vocab:
            new_word = word.replace(bigram, replacement)
            new_vocab[new_word] = vocab[word]
        return new_vocab

    def tokenize_vocabulary(self, num_merges=10):
        bpe_vocab = defaultdict(int)
        for word in self.word_to_index.keys():
            if word != '<UNK>':
                bpe_word = ' '.join(list(word)) + ' </w>'
                bpe_vocab[bpe_word] += 1

        for _ in range(num_merges):
            pairs = self.get_stats(bpe_vocab)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            bpe_vocab = self.merge_vocab(best, bpe_vocab)
            self.bpe_merges.append(best)

        token_dict = {token: idx + 1 for idx, token in enumerate(bpe_vocab.keys())}
        token_dict['<UNK>'] = 0  # Ensure <UNK> is at index 0

        return token_dict

    def encode_bpe(self, text, token_dict):
        bpe_tokens = []
        for word in text.split():
            bpe_word = ' '.join(list(word)) + ' </w>'
            for merge in self.bpe_merges:
                bpe_word = bpe_word.replace(' '.join(merge), ''.join(merge))
            bpe_tokens.append(bpe_word)
        
        return [token_dict.get(token, token_dict['<UNK>']) for token in bpe_tokens]

    def decode_bpe(self, bpe_indices, token_dict):
        reverse_token_dict = {idx: token for token, idx in token_dict.items()}
        decoded_words = []
        for idx in bpe_indices:
            token = reverse_token_dict.get(idx, '<UNK>')
            word = token.replace(' ', '').replace('</w>', '')
            decoded_words.append(word)
        return ' '.join(decoded_words)