import os
import torch
import torch.nn as nn
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def setup():
    dist.init_process_group(
        backend="nccl",
        init_method="env://"
    )


def cleanup():
    dist.destroy_process_group()


def main():
    setup()

    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Dataset
    X = torch.randn(10000, 100)
    y = torch.randint(0, 10, (10000,))

    dataset = TensorDataset(X, y)

    sampler = DistributedSampler(dataset)

    loader = DataLoader(
        dataset,
        batch_size=64,
        sampler=sampler
    )

    # Model
    model = nn.Sequential(
        nn.Linear(100, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )

    model = model.to(device)

    model = DDP(
        model,
        device_ids=[local_rank]
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    criterion = nn.CrossEntropyLoss()

    for epoch in range(5):

        sampler.set_epoch(epoch)

        for inputs, targets in loader:

            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, targets)

            loss.backward()

            optimizer.step()

        if rank == 0:
            print(
                f"Epoch {epoch} "
                f"Loss {loss.item():.4f}"
            )

    cleanup()


if __name__ == "__main__":
    main()
