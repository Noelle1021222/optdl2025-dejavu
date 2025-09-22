import torch
from sklearn.metrics import f1_score

def evaluate_model(model, loader, loss, metric, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        total_loss = 0
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            if outputs.size(1) > 1:  # Multiclass classification
                preds = torch.argmax(outputs, dim=1)
            else:  # Binary classification
                outputs = outputs.squeeze(1)
                preds = outputs > 0.
                labels = labels.float()
            total_loss += loss(outputs, labels).item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    if metric == 'Accuracy':
        correct = sum(p == l for p, l in zip(all_preds, all_labels))
        return correct / len(all_labels)
    elif metric == 'MacroF1':
        return f1_score(all_labels, all_preds, average='macro')
    else:
        return -1 * total_loss / len(loader)