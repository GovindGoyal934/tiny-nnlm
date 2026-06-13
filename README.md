# Tiny NNLM

This project reproduces Bengio et al.'s neural probabilistic language model in PyTorch and trains it on WikiText-2.

## What the model does

The model predicts the next token from a fixed 5-token context. The implementation follows the paper's feedforward design:

- `nn.Embedding` turns each context token into a learned vector.
- The 5 embeddings are concatenated and passed through a `Linear -> Tanh -> Dropout` hidden block.
- A second linear head maps the hidden representation to vocabulary logits.
- A direct embedding-to-output path is added to match the paper's skip connection idea from Equation 3.

The output of the network is a vocabulary-sized logit vector, and training uses cross-entropy loss against the next token.

## Data pipeline

The dataset loader in [src/dataset.py](src/dataset.py) does three things:

1. Downloads WikiText-2 raw text from Hugging Face Datasets.
2. Builds a fixed vocabulary from the training split only.
3. Converts each split into `(5-token context, next-token target)` training examples.

Padding is only used when a prompt is shorter than the context window during generation. Training examples are formed by sliding a 5-word window over the tokenized corpus.

## Training

The training loop in [src/train.py](src/train.py) includes:

- Adam optimization.
- Learning-rate reduction on validation plateaus.
- Gradient clipping.
- Early stopping based on validation loss.
- Saving both the best checkpoint and the canonical `nnlm.pt` file.

The best model snapshot is copied when validation improves, so the saved checkpoint is a true epoch snapshot rather than a live reference that can change later.

## Generation

The text generation script in [src/generate.py](src/generate.py) uses controlled sampling:

- Temperature rescales logits before sampling.
- Top-p / nucleus filtering keeps the smallest token set whose cumulative probability reaches the configured threshold.
- Padding tokens are excluded from sampling.

Run it with a custom prompt to generate text autoregressively from the 5-word context window.

## Files

- [src/model.py](src/model.py): NNLM architecture.
- [src/dataset.py](src/dataset.py): tokenization, vocabulary, and dataset creation.
- [src/train.py](src/train.py): training, validation, checkpointing, and early stopping.
- [src/generate.py](src/generate.py): nucleus sampling and text generation.
- [config.py](config.py): hyperparameters.

## Typical usage

Train the model:

```bash
python src/train.py
```

Generate text after training:

```bash
python src/generate.py --seed "he portrayed" --max_length 20 --temperature 1.0
```

## Notes

- WikiText-2 is loaded through the Hugging Face Datasets package, so the first run may download the dataset.
- Checkpoints are saved as `.pt` files and are ignored by git in this repository.