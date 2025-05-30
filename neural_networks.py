import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class NeuralNet(nn.Module):
    # TODO: Implémenter l'architecture du réseau de neurones
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.net(x)


def train(model, loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0

    # TODO: Implémenter l'entraînement de votre modèle
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)

        preds = model(xb)
        loss = criterion(preds, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * xb.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    print(f"[Epoch {epoch}] train loss: {epoch_loss:.4f}")


def eval(model, loader, criterion):
    model.eval()
    losses, all_preds, all_targets = [], [], []

    # TODO: Implémenter la validation de votre modèle
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)

        logits = model(xb)
        loss = criterion(logits, yb)
        losses.append(loss.item() * xb.size(0))

        preds = logits.argmax(1)
        all_preds.append(preds.cpu())
        all_targets.append(yb.cpu())

    loss = np.sum(losses) / len(loader.dataset)
    y_true = torch.cat(all_targets).numpy()
    y_pred = torch.cat(all_preds).numpy()
    acc = accuracy_score(y_true, y_pred)

    print(f"Validation - loss: {loss:.4f} | acc: {acc:.4f}")
    return y_true, y_pred

def plot_examples(images, labels, preds=None, n=6):
    plt.figure(figsize=(10, 2))

    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(images[i][0], cmap="gray")
        title = f"{labels[i]}"

        if preds is not None:
            title += f"→{preds[i]}"

        plt.title(title)
        plt.axis("off")

    plt.tight_layout()
    plt.show()

def main(batch_size=128, epochs=5, lr=1e-3, random_state=0):
    torch.manual_seed(random_state)

    train_ds = datasets.MNIST(root="data", train=True, download=True)
    test_ds  = datasets.MNIST(root="data", train=False, download=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2)

    # TODO: Instancier votre modèle, votre perte et votre optimiseur
    model = NeuralNet().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # TODO: Effectuer l'entraînement et la validation de votre modèle
    for epoch in range(1, epochs + 1):
        train(model, train_loader, criterion, optimizer, epoch)
        y_true, y_pred = eval(model, test_loader, criterion)

    sample_imgs, sample_labels = next(iter(test_loader))[:6]
    sample_preds = model(sample_imgs.to(DEVICE)).argmax(1).cpu()
    plot_examples(sample_imgs, sample_labels, sample_preds)


if __name__ == "__main__":
    main()
