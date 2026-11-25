import argparse
import re
import tarfile
from pathlib import Path

from scipy.io import loadmat


WNID = re.compile(r"n\d{8}")
JPEG = {".jpg", ".jpeg"}


def unpack(archive, split, target):
    with tarfile.open(archive, "r|*") as source:
        for item in source:
            if not item.isfile():
                continue
            original = Path(item.name)
            if split == "train" and original.suffix == ".tar" and WNID.fullmatch(original.stem):
                folder = target / original.stem
                folder.mkdir(exist_ok=True)
                with tarfile.open(fileobj=source.extractfile(item), mode="r|*") as nested:
                    for member in nested:
                        name = Path(member.name)
                        if member.isfile() and name.suffix.lower() in JPEG:
                            member.name = name.name
                            nested.extract(member, folder, filter="data")
            elif original.suffix.lower() in JPEG:
                folder = target
                if split == "train":
                    match = WNID.match(original.name) or WNID.fullmatch(original.parent.name)
                    if match is None:
                        raise ValueError(f"Cannot identify class for {item.name}")
                    folder = target / match.group()
                    folder.mkdir(exist_ok=True)
                item.name = original.name
                source.extract(item, folder, filter="data")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("split", choices=("train", "val", "test", "devkit"))
    parser.add_argument("archive", type=Path)
    parser.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args()
    if not args.archive.is_file():
        parser.error(f"Archive not found: {args.archive}")
    target = args.data / args.split
    target.mkdir(parents=True, exist_ok=True)
    if args.split == "devkit":
        with tarfile.open(args.archive) as source:
            source.extractall(target, filter="data")
        labels = args.data / "labels"
        labels.mkdir(exist_ok=True)
        metas = sorted(target.rglob("meta.mat"))
        if len(metas) != 1:
            raise ValueError("Expected one meta.mat in the devkit")
        meta = loadmat(metas[0], squeeze_me=True)
        if "synsets" not in meta:
            raise ValueError("Expected synsets in meta.mat")
        (labels / "meta.mat").write_bytes(metas[0].read_bytes())
        for split in ("val", "test"):
            matches = sorted(target.rglob(f"*{split}*ground*truth*.txt"))
            if len(matches) == 1:
                (labels / f"{split}.txt").write_bytes(matches[0].read_bytes())
            if not (labels / f"{split}.txt").exists():
                print(f"Place official {split} ground-truth labels at {labels / (split + '.txt')}")
    else:
        unpack(args.archive, args.split, target)
        count = sum(path.suffix.lower() in JPEG for path in target.rglob("*"))
        if count == 0:
            raise ValueError(f"No images extracted to {target}")
        print(f"{count:,} images in {target}")


if __name__ == "__main__":
    main()
