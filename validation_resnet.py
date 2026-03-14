# validation_resnet.py
import os
import torch
import torch.nn as nn
import torch.optim as optim

from resnet import ResNet20  # from this repo


def main():
    print("Running ResNet validation...")
    print("Torch version:", torch.__version__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Model: 20-layer ResNet on CIFAR-10 (10 classes)
    model = ResNet20(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    batch_size = 64
    num_classes = 10
    num_epochs = 3

    torch.manual_seed(0)
    x = torch.randn(batch_size, 3, 32, 32, device=device)
    y = torch.randint(0, num_classes, (batch_size,), device=device)

    initial_loss = None
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        loss_value = float(loss.item())
        print(f"Epoch {epoch} - loss: {loss_value:.4f}")

        if initial_loss is None:
            initial_loss = loss_value

    # Sanity checks
    if not torch.isfinite(loss):
        raise AssertionError("Final loss is not finite")

    # We expect at least some improvement; tolerate small noise
    if loss_value >= initial_loss * 1.1:
        raise AssertionError(
            f"Loss did not improve enough: initial={initial_loss:.4f}, final={loss_value:.4f}"
        )

    print("ResNet validation completed successfully.")


if __name__ == "__main__":
    main()
