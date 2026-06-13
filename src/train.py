import sys
import os
import copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from tqdm import tqdm
import config
from model import NNLM
from dataset import load_data


def train():
    device = config.DEVICE
    train_loader, valid_loader, test_loader, vocab = load_data()

    model = NNLM(vocab_size=len(vocab)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=config.LR_SCHEDULER_PATIENCE,
        factor=config.LR_SCHEDULER_FACTOR,
    )

    # Early stopping state
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state = None

    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0
        n_batches = 0
        for context, target in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{config.EPOCHS}"
        ):
            context, target = context.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(context)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        current_lr = optimizer.param_groups[0]["lr"]

        model.eval()
        val_loss = 0
        val_batches = 0
        with torch.no_grad():
            for context, target in valid_loader:
                context, target = context.to(device), target.to(device)
                output = model(context)
                loss = criterion(output, target)
                val_loss += loss.item()
                val_batches += 1

        train_ppl = torch.exp(torch.tensor(total_loss / n_batches))
        val_avg_loss = val_loss / val_batches
        val_ppl = torch.exp(torch.tensor(val_avg_loss))
        print(
            f"Epoch {epoch + 1}: train loss={total_loss / n_batches:.4f} "
            f"ppl={train_ppl:.2f} | val loss={val_avg_loss:.4f} "
            f"ppl={val_ppl:.2f} | lr={current_lr:.2e}"
        )

        scheduler.step(val_avg_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # Early stopping: check for improvement in validation loss
        if best_val_loss - val_avg_loss > config.EARLY_STOPPING_MIN_DELTA:
            best_val_loss = val_avg_loss
            epochs_no_improve = 0
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, config.BEST_MODEL_PATH)
            print(f"Validation loss improved; saved best model to {config.BEST_MODEL_PATH}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s)")

        if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered (no improvement for {config.EARLY_STOPPING_PATIENCE} epochs)")
            break

    # Load best state if available before final evaluation
    if best_state is not None:
        model.load_state_dict(best_state)
        # also write the canonical model file
        torch.save(best_state, "nnlm.pt")
        print("Saved final best model to nnlm.pt")

    model.eval()
    test_loss = 0
    test_batches = 0
    with torch.no_grad():
        for context, target in test_loader:
            context, target = context.to(device), target.to(device)
            output = model(context)
            loss = criterion(output, target)
            test_loss += loss.item()
            test_batches += 1
    test_ppl = torch.exp(torch.tensor(test_loss / test_batches))
    print(f"Test perplexity: {test_ppl:.2f}")

    torch.save(model.state_dict(), "nnlm.pt")
    print("Model saved to nnlm.pt")

    return model, vocab


if __name__ == "__main__":
    model, vocab = train()
