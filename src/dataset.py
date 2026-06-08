import re
import torch
from torch.utils.data import DataLoader, Dataset
from collections import Counter
from datasets import load_dataset
import config


def tokenize(text):
    tokens = re.findall(r"\w+|[^\w\s]", text.lower())
    return tokens


class Vocab:
    def __init__(self, tokens, max_size):
        counter = Counter(tokens)
        most_common = counter.most_common(max_size - 2)
        self.stoi = {"<unk>": 0, "<pad>": 1}
        self.itos = ["<unk>", "<pad>"]
        for word, _ in most_common:
            self.stoi[word] = len(self.itos)
            self.itos.append(word)

    def __len__(self):
        return len(self.itos)

    def __call__(self, words):
        return [self.stoi.get(w, 0) for w in words]

    def get_itos(self):
        return self.itos


class WikiText2Dataset(Dataset):
    def __init__(self, text, vocab, context_window):
        tokens = vocab(tokenize(text))
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.context_window = context_window

    def __len__(self):
        return len(self.tokens) - self.context_window

    def __getitem__(self, idx):
        context = self.tokens[idx : idx + self.context_window]
        target = self.tokens[idx + self.context_window]
        return context, target


def load_data():
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=["train", "validation", "test"])
    train_text = "\n".join(dataset[0]["text"])
    valid_text = "\n".join(dataset[1]["text"])
    test_text = "\n".join(dataset[2]["text"])

    all_tokens = tokenize(train_text)
    vocab = Vocab(all_tokens, max_size=config.VOCAB_SIZE)
    print(f"Vocab size: {len(vocab)}")

    train_dataset = WikiText2Dataset(train_text, vocab, config.CONTEXT_WINDOW)
    valid_dataset = WikiText2Dataset(valid_text, vocab, config.CONTEXT_WINDOW)
    test_dataset = WikiText2Dataset(test_text, vocab, config.CONTEXT_WINDOW)

    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True
    )
    valid_loader = DataLoader(valid_dataset, batch_size=config.BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE)

    return train_loader, valid_loader, test_loader, vocab
