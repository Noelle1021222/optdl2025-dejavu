import torch
import torch.optim as optim
import numpy as np
from tqdm.auto import trange

from .data_utils import set_seed
from .evaluation import evaluate_model

def train_model(model, train_loader, val_loader, learning_rate, epochs, patience, stop_criteria, save_path):
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_metric = -1 * np.inf
    best_epoch = 0
    wait = 0

    for epoch in trange(epochs):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if val_loader is not None:
            val_metric = evaluate_model(model, val_loader, criterion, stop_criteria, device)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}, Validation {stop_criteria}: {val_metric:.4f}")

            if val_metric >= best_metric:
                best_metric = val_metric
                best_epoch = epoch + 1 #epoch start from zero
                wait = 0
                torch.save(model.state_dict(), save_path)
            else:
                wait += 1

            if wait >= patience:
                print("Early stopping triggered.")
                break
        else:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")
            if epoch == epochs - 1:
                torch.save(model.state_dict(), save_path)

    return model, best_epoch, best_metric