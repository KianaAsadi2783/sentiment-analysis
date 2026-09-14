import torch
import torch.nn as nn
import torch.optim as optim
from data import get_dataloaders
from model import SentimentRNN
from tqdm import tqdm

# Hyperparameters
BASE_DATA_PATH = r"C:\Users\Lenovo\Desktop\aclImdb" 
BATCH_SIZE = 64 
MAX_VOCAB_SIZE = 25000 
EMBED_SIZE = 100 
HIDDEN_SIZE = 256 
NUM_LAYERS = 2 
DROPOUT = 0.5 
LEARNING_RATE = 0.001 
EPOCHS = 10 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device:{device}")

def train_model():
    train_loader, test_loader, vocab = get_dataloaders(BASE_DATA_PATH, BATCH_SIZE, MAX_VOCAB_SIZE)
    vocab_size = len(vocab)
    print(f"Vocab size:{vocab_size}")

    model = SentimentRNN(
        vocab_size=vocab_size,
        embed_size=EMBED_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        pad_idx=vocab.pad_idx,
        cell_type='lstm'
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training Loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        accuracy = 0.0
        
        batch_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch_idx, (inputs, lengths, labels) in enumerate(batch_bar):
            inputs, labels = inputs.to(device), labels.to(device)
            lengths = torch.clamp(lengths, min=1)

            optimizer.zero_grad()
            outputs = model(inputs, lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            batch_acc = (predicted == labels).sum().item() / labels.size(0)
            accuracy = (accuracy * batch_idx + batch_acc) / (batch_idx + 1)
            total_loss += loss.item()

            batch_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100 * accuracy:.2f}%")
        
        print(f"Epoch {epoch+1} finished. Avg Loss: {total_loss/len(train_loader):.4f}")

    # Evaluating
    print("Evaluating on test data...")
    model.eval()
    total_acc = 0.0
    num_batches = 0
    
    with torch.inference_mode():
        for inputs, lengths, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            lengths = torch.clamp(lengths, min=1)
            
            outputs = model(inputs, lengths)
            _, predicted = torch.max(outputs.data, 1)
            
            total_acc += (predicted == labels).sum().item() / labels.size(0)
            num_batches += 1

    print(f"\nFinal Test Accuracy: {100 * (total_acc / num_batches):.2f}%")
    
if __name__ == "__main__":
    train_model()