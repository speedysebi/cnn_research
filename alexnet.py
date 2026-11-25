import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import functional as F


def model():
    def norm():
        # PyTorch divides alpha by the window size.
        return nn.LocalResponseNorm(5, alpha=5e-4, beta=0.75, k=2)

    def pool():
        return nn.MaxPool2d(3, stride=2)

    net = nn.Sequential(
        nn.Conv2d(3, 96, 11, stride=4, padding=2), nn.ReLU(), norm(), pool(),
        nn.Conv2d(96, 256, 5, padding=2, groups=2), nn.ReLU(), norm(), pool(),
        nn.Conv2d(256, 384, 3, padding=1), nn.ReLU(),
        nn.Conv2d(384, 384, 3, padding=1, groups=2), nn.ReLU(),
        nn.Conv2d(384, 256, 3, padding=1, groups=2), nn.ReLU(), pool(),
        nn.Flatten(),
        nn.Linear(256 * 6 * 6, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 1000),
    )
    layers = [layer for layer in net if isinstance(layer, (nn.Conv2d, nn.Linear))]
    for index, layer in enumerate(layers):
        nn.init.normal_(layer.weight, std=0.01)
        nn.init.constant_(layer.bias, float(index in (1, 3, 4, 5, 6)))
    return net


def base_image(image):
    return F.pil_to_tensor(F.center_crop(F.resize(image, 256), 256))


class Images(Dataset):
    def __init__(self, samples, mean, pca, training):
        self.samples = samples
        self.mean = mean
        self.pca = pca
        self.training = training

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            x = base_image(image.convert("RGB")).float()
        x = (x - self.mean) / 255
        if self.training:
            y, z = torch.randint(33, (2,)).tolist()
            x = x[:, y:y + 224, z:z + 224]
            if torch.rand(()) < 0.5:
                x = x.flip(-1)
            return x + (self.pca @ (torch.randn(3) * 0.1))[:, None, None], label
        crops = F.five_crop(x, 224)
        return torch.stack((*crops, *(crop.flip(-1) for crop in crops))), label


def loader(data, batch, workers, shuffle=False):
    return DataLoader(data, batch_size=batch, shuffle=shuffle, num_workers=workers,
                      persistent_workers=workers > 0)


def stats(training, path, workers):
    training.transform = base_image
    mean = torch.zeros(3, 256, 256, dtype=torch.float64)
    pixel_sum = torch.zeros(3, dtype=torch.float64)
    products = torch.zeros(3, 3, dtype=torch.float64)
    for i, (images, _) in enumerate(loader(training, 32, workers)):
        mean += images.sum(0)
        # Sample one pixel per 8x8 cell for RGB covariance.
        pixels = images[:, :, 4::8, 4::8].permute(1, 0, 2, 3).reshape(3, -1).double() / 255
        pixel_sum += pixels.sum(1)
        products += pixels @ pixels.T
        if i % 1000 == 0:
            print(f"Stats: {i * 32:,}/{len(training):,} images", flush=True)
    count = len(training) * 32 * 32
    rgb = pixel_sum / count
    values, vectors = torch.linalg.eigh(products / count - torch.outer(rgb, rgb))
    torch.save({"mean": (mean / len(training)).float(),
                "pca": (vectors * values).float()}, path)


def labeled_samples(root, split, classes):
    from scipy.io import loadmat

    meta = loadmat(root / "labels" / "meta.mat", squeeze_me=True, struct_as_record=False)
    synsets = {}
    for item in np.atleast_1d(meta["synsets"]).flat:
        id_name = next(name for name in item._fieldnames if name.startswith("ILSVRC"))
        synsets[int(getattr(item, id_name))] = item.WNID
    labels = (root / "labels" / f"{split}.txt").read_text().split()
    images = sorted(path for path in (root / split).rglob("*")
                    if path.suffix.lower() in {".jpeg", ".jpg"})
    if not labels or len(images) != len(labels):
        raise ValueError(f"{split}: found {len(images)} images and {len(labels)} labels")
    for index, image in enumerate(images, 1):
        if int(image.stem.rsplit("_", 1)[1]) != index:
            raise ValueError(f"Unexpected image numbering in {image}")
    return [(image, classes[synsets[int(label)]]) for image, label in zip(images, labels)]


@torch.no_grad()
def evaluate(net, batches, device):
    net.eval()
    hits = torch.zeros(2, device=device)
    count = 0
    for images, labels in batches:
        labels = labels.to(device)
        # Run each crop separately to keep peak GPU memory bounded.
        probabilities = [net(crop.to(device)).softmax(1) for crop in images.unbind(1)]
        predictions = torch.stack(probabilities).mean(0).topk(5, dim=1).indices
        correct = predictions.eq(labels[:, None])
        hits += torch.stack((correct[:, 0].sum(), correct.any(1).sum()))
        count += labels.numel()
    return (100 * (1 - hits / count)).tolist()


def update(net, optimizer, images, labels, device):
    optimizer.zero_grad(set_to_none=True)
    loss = nn.functional.cross_entropy(net(images.to(device)), labels.to(device))
    loss.backward()
    optimizer.step()
    return loss.detach()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("smoke", "stats", "train", "test"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("run"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--steps", type=int, default=1, help="Synthetic smoke steps")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(2012)
    args.out.mkdir(parents=True, exist_ok=True)
    stats_path = args.out / "stats.pt"
    checkpoint = args.out / "latest.pt"

    if args.mode != "smoke":
        training = ImageFolder(args.data / "train")
        if len(training.classes) != 1000:
            raise ValueError(f"Expected 1000 training classes, found {len(training.classes)}")
        if args.mode == "stats":
            stats(training, stats_path, args.workers)
            return

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("PyTorch cannot see a GPU. Use the ROCm environment or --device cpu.")
        print(f"{torch.cuda.get_device_name(device)} | HIP {torch.version.hip}", flush=True)
    net = model().to(device)
    weights = [p for p in net.parameters() if p.ndim > 1]
    biases = [p for p in net.parameters() if p.ndim == 1]
    optimizer = torch.optim.SGD(
        [{"params": weights, "weight_decay": 0.0005}, {"params": biases}],
        lr=0.01, momentum=0.9,
    )

    if args.mode == "smoke":
        batch = args.batch or 2
        images = torch.randn(batch, 3, 224, 224, device=device) * 0.2
        labels = torch.arange(batch, device=device) % 1000
        start = time.perf_counter()
        for _ in range(args.steps):
            loss = update(net, optimizer, images, labels, device)
        loss = loss.item()
        if not torch.isfinite(torch.tensor(loss)):
            raise RuntimeError(f"Nonfinite smoke loss: {loss}")
        elapsed = time.perf_counter() - start
        print(f"{sum(p.numel() for p in net.parameters()):,} parameters | "
              f"loss {loss:.4f} | {batch * args.steps / elapsed:.1f} images/sec", flush=True)
        return

    info = torch.load(stats_path, map_location="cpu", weights_only=True)
    eval_split = "test" if args.mode == "test" else "val"
    eval_set = Images(labeled_samples(args.data, eval_split, training.class_to_idx),
                      info["mean"], info["pca"], False)
    eval_batches = loader(eval_set, min(args.batch or 128, 32), args.workers)
    epoch = 0
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=4, min_lr=1e-5,
    )
    if args.resume or args.mode == "test":
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state["classes"] != training.classes:
            raise ValueError("Checkpoint class mapping differs from the training data")
        net.load_state_dict(state["model"])
        if args.mode == "train":
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            epoch = state["epoch"]
        del state
    if args.mode == "test":
        top1, top5 = evaluate(net, eval_batches, device)
        print(f"Test top-1 error {top1:.2f}% | top-5 error {top5:.2f}%")
        return

    train_set = Images(training.samples, info["mean"], info["pca"], True)
    train_batches = loader(train_set, args.batch or 128, args.workers, shuffle=True)
    for epoch in range(epoch + 1, args.epochs + 1):
        net.train()
        start = time.perf_counter()
        for i, (images, labels) in enumerate(train_batches, 1):
            loss = update(net, optimizer, images, labels, device)
            if i == 1 or i % 100 == 0:
                value = loss.item()
                if not torch.isfinite(torch.tensor(value)):
                    raise RuntimeError(f"Nonfinite loss at epoch {epoch}, batch {i}")
                print(f"Epoch {epoch} | batch {i}/{len(train_batches)} | loss {value:.4f}", flush=True)
        duration = time.perf_counter() - start
        top1, top5 = evaluate(net, eval_batches, device)
        scheduler.step(top1)
        print(f"Epoch {epoch} | {duration:.1f}s training | val top-1 error {top1:.2f}% | "
              f"val top-5 error {top5:.2f}%", flush=True)
        temporary = checkpoint.with_suffix(".tmp")
        torch.save({"model": net.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "classes": training.classes,
                    "epoch": epoch}, temporary)
        temporary.replace(checkpoint)


if __name__ == "__main__":
    main()
