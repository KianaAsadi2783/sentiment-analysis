# data
import re 
from pathlib import Path
from collections import Counter
import torch 
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'

# clear split texts
def tokenizer(text):
    text = text.lower()
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.strip().split()
    return tokens

class Vocabulary:
    # dict for words(0: pad, 1:unknown words)
    def __init__(self):
        # (word to Id)
        self.word2idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        # revers (ID to word)
        self.idx2word = {0:PAD_TOKEN, 1:UNK_TOKEN}

        self.pad_idx = 0
        self.unk_idx = 1
    
    def add_word(self, word):
        # if it was new then add
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def __len__(self):
        # return len of all current words in dict
        return len(self.word2idx)
    
# pytorch dataset class
class IMDBDataset(Dataset):
    def __init__(self, reviews, labels, vocab):
        self.reviews = reviews # all texts(tokenize befor)
        self.labels = labels
        self.vocab = vocab

    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, idx):
        text = self.reviews[idx]
        label = self.labels[idx]

        word_ids = [self.vocab.word2idx.get(w, self.vocab.unk_idx) for w in text]
        return torch.tensor(word_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)
    
def load_imdb_data(base_path, split):
    reviews = []
    labels = []
    # combine base adress &file name
    split_path = Path(base_path) / split

    # loop for pos/neg 
    for label_type, label_val in [('pos',1),('neg',0)]:
        dir_path = split_path / label_type # neg or pos file?
        if not dir_path.exists():
            continue

        for file_path in dir_path.glob('*.txt'):
            with open(file_path, 'r', encoding='utf_8') as f:
                tokens = tokenizer(f.read())
                if len(tokens) >0: # not empty
                    reviews.append(tokens)
                    labels.append(label_val)

    return reviews, labels

# create vocab base on train(only)
def build_vocab(train_reviews, max_vocab_size=25000):
    vocab = Vocabulary()
    word_counts = Counter()

    for review in train_reviews:
        word_counts.update(review) # count words for each review
    
    for word, count in word_counts.most_common(max_vocab_size):
        vocab.add_word(word)

    return vocab

# batch & pad
def collate_fn(batch):
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    # lists of seq & label of batch
    sequences, labels = zip(*batch)
    # original lenghts of each sentences for RNN
    lenghts = torch.tensor([len(seq) for seq in sequences])

    padded_seq = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.stack(labels)

    return padded_seq, lenghts, labels

def get_dataloaders(base_path, batch_size=32, max_vocab_size=25000):
    # reed data
    train_reviews, train_labels = load_imdb_data(base_path, "train")
    test_reviews, test_labels = load_imdb_data(base_path, 'test')

    # build dict
    vocab = build_vocab(train_reviews, max_vocab_size)

    # pytorch dataset
    train_dataset = IMDBDataset(train_reviews, train_labels, vocab)
    test_dataset = IMDBDataset(test_reviews, test_labels, vocab)

    # Dataloader 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, test_loader, vocab
