import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.datasets import FakeData
from torchvision.transforms import ToTensor


def setup():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, local_rank


def cleanup():
    dist.destroy_process_group()


def main():
    rank, local_rank = setup()

    # Model
    model = torch.nn.Sequential(
        torch.nn.Linear(784, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 10),
    ).cuda(local_rank)

    # Wrap with DDP
    model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    # Dataset
    dataset = FakeData(
        size=10000,
        image_size=(1, 28, 28),
        num_classes=10,
        transform=ToTensor(),
    )

    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=64,
        sampler=sampler,
        num_workers=2,
    )

    for epoch in range(5):
        sampler.set_epoch(epoch)

        for images, labels in loader:
            images = images.cuda(local_rank)
            labels = labels.cuda(local_rank)

            images = images.view(images.size(0), -1)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            # DDP gradient synchronization occurs here
            loss.backward()

            optimizer.step()

    cleanup()


if __name__ == "__main__":
    main()
