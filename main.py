import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from data import get_dataloaders
from model import SentimentRNN
from tqdm import tqdm

# Hyperparameters
BASE_DATA_PATH = Path(__file__).resolve().parent / "aclImdb"
BATCH_SIZE = 64
MAX_VOCAB_SIZE = 25000
EMBED_SIZE = 100
HIDDEN_SIZE = 256
NUM_LAYERS = 2
DROPOUT = 0.5
LEARNING_RATE = 0.001
EPOCHS = 10
CELL_TYPE = 'rnn'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


def create_model(vocab):
    """Create the sentiment analysis model."""
    model = SentimentRNN(
        vocab_size=len(vocab),
        embed_size=EMBED_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        pad_idx=vocab.pad_idx,
        cell_type=CELL_TYPE
    ).to(device)

    return model


def get_model_path():
    """Return the checkpoint path for the selected model type."""
    save_dir = Path(__file__).resolve().parent / "checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)

    return save_dir / f"sentiment_{CELL_TYPE}.pth"


def train_model(train_loader, vocab):
    """Train the model and save the trained weights."""
    model = create_model(vocab)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training Loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        accuracy = 0.0

        batch_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{EPOCHS}"
        )

        for batch_idx, (inputs, lengths, labels) in enumerate(batch_bar):
            inputs = inputs.to(device)
            labels = labels.to(device)
            lengths = torch.clamp(lengths, min=1)

            optimizer.zero_grad()

            outputs = model(inputs, lengths)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)

            batch_acc = (
                (predicted == labels).sum().item()
                / labels.size(0)
            )

            accuracy = (
                (accuracy * batch_idx + batch_acc)
                / (batch_idx + 1)
            )

            total_loss += loss.item()

            batch_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{100 * accuracy:.2f}%"
            )

        print(
            f"Epoch {epoch + 1} finished. "
            f"Avg Loss: {total_loss / len(train_loader):.4f}"
        )

    # Save trained model
    model_path = get_model_path()
    torch.save(model.state_dict(), model_path)

    print(f"Model saved to {model_path}")

    return model


def load_model(vocab):
    """Load the saved model weights."""
    model = create_model(vocab)
    model_path = get_model_path()

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device,
            weights_only=True
        )
    )

    print(f"Model loaded from {model_path}")

    return model


def evaluate_model(model, test_loader):
    """Evaluate the model on the test dataset."""
    print("Evaluating on test data...")

    model.eval()

    total_correct = 0
    total_samples = 0

    with torch.inference_mode():
        for inputs, lengths, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            lengths = torch.clamp(lengths, min=1)

            outputs = model(inputs, lengths)
            _, predicted = torch.max(outputs, 1)

            total_correct += (
                (predicted == labels).sum().item()
            )

            total_samples += labels.size(0)

    accuracy = 100 * total_correct / total_samples

    print(f"Final Test Accuracy: {accuracy:.2f}%")


def main():
    # Check dataset
    if not BASE_DATA_PATH.exists():
        raise FileNotFoundError(
            f"IMDb dataset not found at: {BASE_DATA_PATH}\n"
            "Please download the dataset and place the "
            "'aclImdb' folder in the project root."
        )

    # Load dataset and build vocabulary
    train_loader, test_loader, vocab = get_dataloaders(
        BASE_DATA_PATH,
        BATCH_SIZE,
        MAX_VOCAB_SIZE
    )

    print(f"Vocab size: {len(vocab)}")

    model_path = get_model_path()

    # Load existing model or train a new one
    if model_path.exists():
        print("Saved model found. Loading model...")
        model = load_model(vocab)
    else:
        print("No saved model found. Training a new model...")
        model = train_model(train_loader, vocab)

    # Evaluate the model
    evaluate_model(model, test_loader)


if __name__ == "__main__":
    main()