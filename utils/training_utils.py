import torch

def train(model, loader, criterion, optimizer):
    # Set the model to training mode
    model.train()
    train_loss = 0
    for data, target in loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    return train_loss / len(loader)

def evaluate(model, loader, criterion):
    # Set the model to evaluation mode
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, target in loader:
            output = model(data)
            val_loss += criterion(output, target).item()
    
    return val_loss / len(loader)
