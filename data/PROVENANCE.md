# 1. PA2 Data and Model Provenance

## 2. Required recognition dataset

CIFAR-10 was created by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton. The official source describes 60,000 32×32 color images in ten classes, split into 50,000 training and 10,000 test images: [University of Toronto CIFAR page](https://www.cs.toronto.edu/~kriz/cifar.html).

The notebook downloads CIFAR-10 through `torchvision.datasets.CIFAR10`; the repository does not copy the image archive. `split_spec.json` specifies the deterministic selection algorithm at seed `16720`, and `generate_splits.py` can save the split indices after the official archive is present. Review the dataset's known web-image provenance and do not treat it as free of social, privacy, or labeling limitations.

## 3. Pretrained Atlas models

Approved frozen extractors are torchvision `ResNet18_Weights.IMAGENET1K_V1` and `Swin_T_Weights.IMAGENET1K_V1` (the approved DEFAULT weights). Their weight metadata supplies the preprocessing recipe and original ImageNet categories. Students must name the exact weight enum in their manifest/report and may not redistribute the cached weight file in the submission.

## 4. Personal Atlas data

Use `atlas_manifest.csv`. Record filename, category, development/held-out split, whether it is the student's own photo, creator/source URL, license or permission, and consent/privacy notes. Strip EXIF from submitted copies. Generated content never counts as an original photograph.

The release includes a deterministic, course-authored accessibility collection
under `accessibility_fallback/`. It has three categories with six development
and two held-out images per category, a file manifest, SHA-256 record, license,
and staff approval template. Run `setup_accessibility_fallback.py` to reproduce
it exactly. It may be issued only with advance staff approval for access,
privacy, safety, or accessibility, and the completed approval states whether
the original-photo rule is waived. It is not a general alternative to the
assigned collection; all held-out analysis and Atlas requirements remain.

## 5. License and redistribution rule

Do not commit downloaded CIFAR images, ImageNet weights, or personal Atlas
photos to the course repository. The course-authored accessibility images, split specifications, manifests,
templates, and code are included in the release.
