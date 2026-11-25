# AlexNet on the Radeon AI PRO R9700

This project targets Table 1 of the original AlexNet paper. [Krizhevsky, Sutskever, and Hinton (2012)](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
The paper reports 39.7% top-1 and 18.9% top-5 error on ILSVRC 2010 test.
The 15.3% figure comes from the separate ILSVRC 2012 competition ensemble.

## Get the data

- [ImageNet ILSVRC 2010 downloads](https://www.image-net.org/challenges/LSVRC/2010/2010-downloads.php).
  Get the original train, validation, and test images, the development kit, and the separate test ground truth. The organizers also list an image patch for the legacy archives.
- [Shanghai Jiao Tong University train and validation archives](https://scidata.sjtu.edu.cn/records/b2ft9-xh152).
  This record lists train (133.1 GB) and validation (5.4 GB), with MD5 hashes.
- [ImageNet ILSVRC 2012 downloads](https://www.image-net.org/challenges/LSVRC/2012/2012-downloads.php).
  For a separate 2012 run, use the Task 1 & 2 train images, validation images and devkit. Use a separate data and run directory. For local scoring use validation, since 2012 test labels are held back.
- [ImageNet account and data access](https://www.image-net.org/download).

Allow roughly 350 GB free if keeping both the archives and extracted 2010 images on one drive. Store archives on another drive or extract to a larger SSD using the same --data path throughout. Windows PowerShell examples below assume the archives are on D:. Edit the actual names and drive to match your downloads.

## Run in PowerShell

```powershell
# Run from the repository root with uv installed.
uv venv --python 3.12 .venv-rocm
uv pip install --python .venv-rocm/Scripts/python.exe --index https://stable.repo.amd.com/rocm/whl-next/ "torch[device-gfx1201]==2.15.0+rocm10.2.0" "torchvision[device-gfx1201]==0.29.2+rocm10.2.0" "scipy==1.18.1"
$py = ".\.venv-rocm\Scripts\python.exe"
& $py alexnet.py smoke --batch 128
& $py prepare.py train "D:\ImageNet\ILSVRC2010_images_train.tar"
& $py prepare.py val "D:\ImageNet\ILSVRC2010_images_val.tar"
& $py prepare.py test "D:\ImageNet\ILSVRC2010_images_test.tar"
& $py prepare.py devkit "D:\ImageNet\ILSVRC2010_devkit-1.0.tar.gz"
```

Put ILSVRC2010_test_ground_truth.txt in data/labels/test.txt with one numeric class ID per line in image-number order. The devkit command copies meta.mat and any recognizably named validation or test ground-truth text files into data/labels. If the validation file has a different name, copy it to data/labels/val.txt. Training images go under data/train/<WordNet ID>. Validation and test images stay in numbered order under data/val and data/test.

```powershell
& $py alexnet.py stats --workers 4
& $py alexnet.py train --batch 128 --workers 4 --epochs 90
& $py alexnet.py test --batch 32 --workers 4
```

The first command computes the training-set mean image and RGB covariance. Training prints validation errors and atomically replaces run/latest.pt after each completed epoch. To continue after an interruption, rerun the training command with --resume. The --epochs value is the total target, including completed epochs. For an initial one-epoch run, use --epochs 1 and then resume with --epochs 90.

This is a single-GPU PyTorch reimplementation. It keeps the grouped convolutions, local response normalization, overlapping pooling, dropout, 224×224 crops, PCA color jitter, SGD settings and ten-crop probability averaging. Implementation choices include dividing the mean-centered RGB values by 255, using every eighth row and column to estimate the PCA covariance, padding the first convolution by two pixels, and using a validation plateau of four epochs to trigger the learning-rate reductions described manually in the paper. Accuracy will depend on those choices and the actual data. Use the first real epoch's logged training duration to estimate runtime, then allow for validation and preprocessing.

The GPU environment uses Python 3.12, AMD's PyTorch 2.15.0 with ROCm 10.2.0 and torchvision 0.29.2. PyTorch calls its ROCm device API cuda. On the R9700, the GPU smoke test completed 10 synthetic training steps at batch size 128. Full training accuracy is pending the dataset run. [Original 2012 CUDA source](https://github.com/computerhistory/AlexNet-Source-Code)
