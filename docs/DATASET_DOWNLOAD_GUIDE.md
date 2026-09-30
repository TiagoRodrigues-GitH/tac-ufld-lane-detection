# Dataset download guide

This guide is for the person downloading the lane-detection datasets for the TAC-UFLD project. You only need to download and extract the files; you don't need to train anything or understand the code. When a dataset is extracted, run the check at the end and send the printed result to Tiago.

Sizes were read from the download pages on 30 Sep 2026. Plan for about twice the archive size in free space while extracting; after extraction you can delete the archives.

| Dataset | Download from | Archives to download | Space needed |
|---|---|---|---|
| CULane | [Google Drive](https://drive.google.com/drive/folders/1mSLgwVTiaUMAb4AVOWwlCD5JcWdrwpvu) | 9 files | about 41 GB (85 GB during extraction) |
| TuSimple | [Kaggle](https://www.kaggle.com/datasets/manideep1108/tusimple) (free account needed) | 1 zip | about 24 GB (50 GB during extraction) |
| OpenLane | [Google Drive](https://drive.google.com/drive/folders/18upnDfB-VVuQf3GPiv_JQn1-BUOcAotk) | 22 files (images in 20 parts + 2 annotation files) | about 124 GB (250 GB during extraction) |

Use one parent folder for all datasets, for example `D:\datasets\`, on a drive with enough space (an SSD makes training faster). Don't rename the folders that come out of the archives. On Windows, [7-Zip](https://www.7-zip.org/) opens `.tar`, `.tar.gz` and `.zip` files. A `.tar.gz` needs two steps in 7-Zip: first extract the `.tar` from it, then extract that `.tar`.

## 1. CULane

Download these files from the CULane Drive folder:

| File | Size | Contents |
|---|---|---|
| `driver_23_30frame.tar.gz` | 20.44 GB | training/validation images |
| `driver_161_90frame.tar.gz` | 4.69 GB | training/validation images |
| `driver_182_30frame.tar.gz` | 5.49 GB | training/validation images |
| `driver_37_30frame.tar.gz` | 1.04 GB | test images |
| `driver_100_30frame.tar.gz` | 4.75 GB | test images |
| `driver_193_90frame.tar.gz` | 4.2 GB | test images |
| `laneseg_label_w16.tar.gz` | 249.7 MB | lane label images |
| `list.tar.gz` | 1.1 MB | lists of training/validation/test images |
| `annotations_new.tar.gz` | 38 MB | corrected lane annotations (see step 3) |

Not needed: `video_example.zip` and `laneseg_label_w16_test.zip`.

Steps:

1. Create `D:\datasets\CULane\`.
2. Extract every archive **into** `D:\datasets\CULane\`.
3. Extract `annotations_new.tar.gz` **last**, into the same folder, and let it **overwrite** files when asked. The CULane authors say the original training/validation annotations are wrong and this archive replaces them.

Expected result:

```
D:\datasets\CULane\
├── driver_23_30frame\        (folders like 05151649_0422.MP4\ with 00000.jpg + 00000.lines.txt ...)
├── driver_161_90frame\
├── driver_182_30frame\
├── driver_37_30frame\
├── driver_100_30frame\
├── driver_193_90frame\
├── laneseg_label_w16\
└── list\
    ├── train_gt.txt          (88,880 lines)
    ├── val_gt.txt            (9,675 lines)
    ├── test.txt              (34,680 lines)
    └── test_split\           (9 files: test0_normal.txt ... test8_night.txt)
```

## 2. TuSimple

1. Log in to Kaggle (a free account is enough) and click **Download** on the [TuSimple page](https://www.kaggle.com/datasets/manideep1108/tusimple). You get one zip of about 24 GB.
2. Extract it into `D:\datasets\`. It creates a folder called `TUSimple`.

Expected result:

```
D:\datasets\TUSimple\
├── train_set\
│   ├── clips\                (one folder per drive, then one per clip with 20 frames: 1.jpg ... 20.jpg)
│   ├── label_data_0313.json
│   ├── label_data_0531.json
│   └── label_data_0601.json
├── test_set\
│   └── clips\
└── test_label.json           (test annotations, 2,782 lines)
```

If `test_label.json` ends up inside `test_set\`, that is also fine.

## 3. OpenLane (version 1)

Download these files from the OpenLane Drive folder:

- Everything inside the `split_tar` sub-folder: `images_training_0.tar` … `images_training_15.tar` (about 5 GB each) and `images_validation_0.tar` … `images_validation_3.tar`. Together these 20 parts hold the same images as the single 103.4 GB `images.tar`. Download **either** the parts **or** `images.tar`, not both; the parts are easier to resume if the connection drops.
- `lane3d_1000_training.tar` (13.64 GB).
- `lane3d_1000_validation_test.tar` (6.49 GB).

Not needed: `lane3d_300.tar` (a smaller subset of the same data), `cipo.zip`, `scene.zip`.

Steps:

1. Create `D:\datasets\OpenLane\`.
2. Extract every `.tar` into it.

Expected result:

```
D:\datasets\OpenLane\
├── images\
│   ├── training\             (798 folders segment-...\ with .jpg frames)
│   └── validation\           (202 folders segment-...\)
└── lane3d_1000\
    ├── training\             (798 folders segment-...\ with one .json per frame)
    ├── validation\           (202 folders segment-...\)
    └── test\                 (six scenario folders, e.g. ..._case\; optional)
```

## 4. ELAS

ELAS is already on Tiago's computer (`LaneDetection\datasets\dataset_elas_v1`), so there is nothing to download. Scene `GRI_S02` still has its images zipped (`images.zip`). The current experiments don't use it; unzip it into `GRI_S02\images\` only if you are asked to.

## Check each dataset

Python 3 is all you need. From the project folder (`tac-ufld-lane-detection`), run one line per dataset:

```
python scripts/check_dataset.py culane   D:\datasets\CULane
python scripts/check_dataset.py tusimple D:\datasets\TUSimple
python scripts/check_dataset.py openlane D:\datasets\OpenLane
```

The check only reads files; it never changes anything. Each line starts with one of these marks:

- `[ OK ]`: that part is present.
- `[WARN]`: that part is usable, but something differs from the published numbers. For example, some images may be missing after an interrupted download.
- `[MISS]`: that part is missing. The line says which archive to extract, and where.

The last line reads **READY** or **NOT READY**. Send the whole output to Tiago, and fix any `[MISS]` line before training starts.
