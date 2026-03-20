import torch
import aimet_torch
import torch.nn as nn

def test_device():
    print("CUDA available:", torch.cuda.is_available())
    print("GPU count:", torch.cuda.device_count())

    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.randn(1000, 1000).to(device)
    y = torch.randn(1000, 1000).to(device)

    z = torch.matmul(x, y)

    print("Tensor device:", z.device)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = nn.Sequential(
        nn.Conv2d(3,16,3),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(576,10)
    ).to(device)

    print("Model device:", next(model.parameters()).device)

if __name__ == "__main__":
    test_device()