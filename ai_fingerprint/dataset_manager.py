from __future__ import annotations

import hashlib
import os
import random
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .dataset_catalog import DATASETS, DatasetSpec, get_dataset_spec
from .data_partition import make_partition_assignment


UCI_HAR_URL = (
    "https://archive.ics.uci.edu/static/public/240/"
    "human+activity+recognition+using+smartphones.zip"
)


class DatasetError(RuntimeError):
    pass


def _safe_extract_zip(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            target = (destination / member.filename).resolve()
            if destination not in target.parents and target != destination:
                raise DatasetError(
                    f"Unsafe path in archive: {member.filename}"
                )
        zf.extractall(destination)


def _download_file(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "ai-fingerprint-experiment/0.2"},
    )
    temporary = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(request, timeout=120) as response:
        with temporary.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    temporary.replace(target)


def _stable_token_id(token: str, vocab_size: int) -> int:
    digest = hashlib.blake2b(
        token.encode("utf-8", errors="ignore"),
        digest_size=8,
    ).digest()
    value = int.from_bytes(digest, "big")
    return 2 + (value % max(vocab_size - 2, 1))


def _tokenize_text(
    text: str,
    length: int,
    vocab_size: int,
) -> np.ndarray:
    tokens = text.lower().split()
    values = np.zeros(length, dtype=np.int64)
    for index, token in enumerate(tokens[:length]):
        values[index] = _stable_token_id(token, vocab_size)
    return values


def _image_to_chw_float(image: Any, size: int) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise DatasetError(
            "Image dataset support requires Pillow. Install it with: pip install pillow"
        ) from exc

    if hasattr(image, "detach"):
        array = image.detach().cpu().numpy()
        if array.ndim == 3 and array.shape[0] in {1, 3, 4}:
            if array.shape[0] == 1:
                array = np.repeat(array, 3, axis=0)
            if array.shape[0] == 4:
                array = array[:3]
            return array.astype(np.float32, copy=False)

    if isinstance(image, np.ndarray):
        array = image
        if array.ndim == 2:
            array = np.repeat(array[:, :, None], 3, axis=2)
        if array.ndim == 3 and array.shape[0] in {1, 3, 4}:
            if array.shape[0] == 1:
                array = np.repeat(array, 3, axis=0)
            if array.shape[0] == 4:
                array = array[:3]
            return array.astype(np.float32, copy=False)

        image = Image.fromarray(array)

    if not isinstance(image, Image.Image):
        try:
            image = Image.fromarray(np.asarray(image))
        except Exception as exc:
            raise DatasetError(
                f"Cannot convert image type {type(image)!r}"
            ) from exc

    image = image.convert("RGB")
    image = image.resize((size, size))
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.transpose(array, (2, 0, 1))


class BaseSource:
    def __len__(self) -> int:
        raise NotImplementedError

    def get(self, index: int) -> np.ndarray:
        raise NotImplementedError

    def get_with_target(
        self,
        index: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        raise DatasetError(
            f"{type(self).__name__} does not expose training targets"
        )


class SyntheticSource(BaseSource):
    def __init__(self, config: Dict[str, Any], seed: int) -> None:
        self.config = config
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.length = max(
            int(config["execution"]["repetitions"])
            + int(config["execution"]["warmup"]),
            100,
        )

    def __len__(self) -> int:
        return self.length

    def get(self, index: int) -> np.ndarray:
        ai = self.config["ai"]
        application = ai["application"]

        if application in {
            "image_classification",
            "object_detection",
            "image_segmentation",
            "reconstruction",
            "anomaly_detection",
            "image_generation",
            "image_denoising",
        }:
            size = int(ai["input_size"])
            return self.rng.random(
                (3, size, size),
                dtype=np.float32,
            )

        if application == "activity_recognition":
            seq = int(ai["sequence_length"])
            dim = int(ai["input_dim"])
            return self.rng.normal(
                size=(seq, dim)
            ).astype(np.float32)

        if application == "text_classification":
            length = int(ai["max_text_length"])
            vocab = int(ai["vocab_size"])
            return self.rng.integers(
                low=0,
                high=vocab,
                size=(length,),
                dtype=np.int64,
            )

        if application in {
            "node_classification",
            "graph_classification",
        }:
            nodes = int(ai.get("graph_nodes", 32))
            features = int(ai.get("graph_features", 16))
            adjacency = self.rng.integers(
                low=0,
                high=2,
                size=(nodes, nodes),
                dtype=np.int64,
            ).astype(np.float32)
            adjacency = np.triu(adjacency, 1)
            adjacency = adjacency + adjacency.T
            node_features = self.rng.normal(
                size=(nodes, features),
            ).astype(np.float32)
            return np.concatenate(
                [adjacency, node_features],
                axis=1,
            )

        raise DatasetError(
            f"Synthetic source does not support application={application!r}"
        )

    def partition_label(self, index: int) -> int:
        classes = max(int(self.config.get("ai", {}).get("num_classes", 2)), 1)
        # Deterministic label used only for reproducible data partitioning.
        return int((int(index) * 1103515245 + self.seed) % classes)

    def get_with_target(
        self,
        index: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        sample = self.get(index)
        ai = self.config["ai"]
        application = ai["application"]

        if application in {
            "image_classification",
            "activity_recognition",
            "text_classification",
            "node_classification",
            "graph_classification",
        }:
            label = self.rng.integers(
                low=0,
                high=max(int(ai.get("num_classes", 2)), 1),
                size=(),
                dtype=np.int64,
            )
            return sample, np.asarray(label, dtype=np.int64)

        if application in {
            "reconstruction",
            "anomaly_detection",
            "image_denoising",
        }:
            return sample, np.asarray(sample, dtype=np.float32)

        raise DatasetError(
            f"Synthetic training targets are not implemented for "
            f"application={application!r}"
        )


class TorchvisionImageSource(BaseSource):
    def __init__(
        self,
        name: str,
        root: Path,
        split: str,
        input_size: int,
        auto_download: bool,
    ) -> None:
        try:
            from torchvision import datasets
        except ImportError as exc:
            raise DatasetError(
                "Torchvision dataset requested but torchvision is not installed"
            ) from exc

        self.name = name
        self.input_size = input_size
        self.dataset = self._build(
            datasets=datasets,
            root=root,
            split=split,
            download=auto_download,
        )

    def _build(self, datasets, root: Path, split: str, download: bool):
        root.mkdir(parents=True, exist_ok=True)
        is_train = split == "train"

        if self.name == "mnist":
            return datasets.MNIST(
                root=str(root),
                train=is_train,
                download=download,
            )
        if self.name == "fashion_mnist":
            return datasets.FashionMNIST(
                root=str(root),
                train=is_train,
                download=download,
            )
        if self.name == "cifar10":
            return datasets.CIFAR10(
                root=str(root),
                train=is_train,
                download=download,
            )
        if self.name == "cifar100":
            return datasets.CIFAR100(
                root=str(root),
                train=is_train,
                download=download,
            )
        if self.name == "svhn":
            actual_split = "train" if is_train else "test"
            return datasets.SVHN(
                root=str(root),
                split=actual_split,
                download=download,
            )
        if self.name == "stl10":
            actual_split = "train" if is_train else "test"
            return datasets.STL10(
                root=str(root),
                split=actual_split,
                download=download,
            )
        if self.name == "food101":
            actual_split = "train" if is_train else "test"
            return datasets.Food101(
                root=str(root),
                split=actual_split,
                download=download,
            )
        if self.name == "oxford_iiit_pet":
            actual_split = "trainval" if is_train else "test"
            return datasets.OxfordIIITPet(
                root=str(root),
                split=actual_split,
                target_types="category",
                download=download,
            )
        if self.name == "flowers102":
            actual_split = "train" if is_train else "test"
            return datasets.Flowers102(
                root=str(root),
                split=actual_split,
                download=download,
            )
        if self.name == "dtd":
            actual_split = "train" if is_train else "test"
            return datasets.DTD(
                root=str(root),
                split=actual_split,
                download=download,
            )
        if self.name == "eurosat":
            return datasets.EuroSAT(
                root=str(root),
                download=download,
            )
        if self.name == "caltech101":
            return datasets.Caltech101(
                root=str(root),
                download=download,
            )

        raise DatasetError(
            f"No torchvision image source implementation for {self.name!r}"
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def get(self, index: int) -> np.ndarray:
        item = self.dataset[index]
        image = item[0] if isinstance(item, tuple) else item
        return _image_to_chw_float(image, self.input_size)

    def get_with_target(
        self,
        index: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        item = self.dataset[index]
        if not isinstance(item, tuple) or len(item) < 2:
            raise DatasetError(
                f"Dataset {self.name!r} does not expose class labels"
            )
        image, target = item[0], item[1]
        return (
            _image_to_chw_float(image, self.input_size),
            np.asarray(int(target), dtype=np.int64),
        )


class VOCDetectionSource(BaseSource):
    def __init__(
        self,
        year: str,
        root: Path,
        split: str,
        input_size: int,
        auto_download: bool,
    ) -> None:
        try:
            from torchvision.datasets import VOCDetection
        except ImportError as exc:
            raise DatasetError(
                "VOC dataset support requires torchvision"
            ) from exc

        if year == "2007":
            image_set = "trainval" if split == "train" else "test"
        else:
            image_set = "train" if split == "train" else "val"

        self.dataset = VOCDetection(
            root=str(root),
            year=year,
            image_set=image_set,
            download=auto_download,
        )
        self.input_size = input_size

    def __len__(self) -> int:
        return len(self.dataset)

    def get(self, index: int) -> np.ndarray:
        image, _ = self.dataset[index]
        return _image_to_chw_float(image, self.input_size)


class TorchvisionSegmentationSource(BaseSource):
    def __init__(
        self,
        name: str,
        root: Path,
        split: str,
        input_size: int,
        auto_download: bool,
    ) -> None:
        try:
            from torchvision import datasets
        except ImportError as exc:
            raise DatasetError(
                "Segmentation dataset support requires torchvision"
            ) from exc

        self.input_size = input_size

        if name == "voc2012_segmentation":
            image_set = "train" if split == "train" else "val"
            self.dataset = datasets.VOCSegmentation(
                root=str(root),
                year="2012",
                image_set=image_set,
                download=auto_download,
            )
        elif name == "oxford_iiit_pet_segmentation":
            actual_split = "trainval" if split == "train" else "test"
            self.dataset = datasets.OxfordIIITPet(
                root=str(root),
                split=actual_split,
                target_types="segmentation",
                download=auto_download,
            )
        else:
            raise DatasetError(
                f"No segmentation source implementation for {name!r}"
            )

    def __len__(self) -> int:
        return len(self.dataset)

    def get(self, index: int) -> np.ndarray:
        image, _ = self.dataset[index]
        return _image_to_chw_float(image, self.input_size)


class LocalImageFolderSource(BaseSource):
    IMAGE_EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
        ".JPEG",
    }

    def __init__(
        self,
        base_path: Path,
        split: str,
        input_size: int,
    ) -> None:
        split_candidates = [
            base_path / split,
            base_path / f"{split}2017",
        ]
        self.root = next(
            (candidate for candidate in split_candidates if candidate.exists()),
            base_path,
        )
        if not self.root.exists():
            raise DatasetError(
                f"Local dataset path does not exist: {self.root}"
            )

        self.files = sorted(
            path
            for path in self.root.rglob("*")
            if path.is_file() and path.suffix in self.IMAGE_EXTENSIONS
        )
        if not self.files:
            raise DatasetError(
                f"No image files found under {self.root}"
            )

        self.input_size = input_size
        class_names = sorted(
            {
                path.parent.name
                for path in self.files
                if path.parent != self.root
            }
        )
        self.class_to_index = {
            name: index
            for index, name in enumerate(class_names)
        }

    def __len__(self) -> int:
        return len(self.files)

    def get(self, index: int) -> np.ndarray:
        try:
            from PIL import Image
        except ImportError as exc:
            raise DatasetError(
                "Local image dataset support requires Pillow"
            ) from exc

        with Image.open(self.files[index]) as image:
            return _image_to_chw_float(image.copy(), self.input_size)

    def get_with_target(
        self,
        index: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        path = self.files[index]
        if path.parent.name not in self.class_to_index:
            raise DatasetError(
                "Local image training requires class subdirectories"
            )
        return (
            self.get(index),
            np.asarray(
                self.class_to_index[path.parent.name],
                dtype=np.int64,
            ),
        )


class UCIHARSource(BaseSource):
    def __init__(
        self,
        root: Path,
        split: str,
        auto_download: bool,
    ) -> None:
        dataset_root = root / "uci_har"
        extracted = dataset_root / "UCI HAR Dataset"

        if not extracted.exists():
            if not auto_download:
                raise DatasetError(
                    f"UCI HAR is not available at {extracted}. "
                    "Enable data.auto_download or run the dataset download command."
                )
            dataset_root.mkdir(parents=True, exist_ok=True)
            archive = dataset_root / "uci_har.zip"
            _download_file(UCI_HAR_URL, archive)
            _safe_extract_zip(archive, dataset_root)

        actual = "train" if split == "train" else "test"
        y_path = extracted / actual / f"y_{actual}.txt"
        signal_dir = extracted / actual / "Inertial Signals"

        signal_names = [
            f"body_acc_x_{actual}.txt",
            f"body_acc_y_{actual}.txt",
            f"body_acc_z_{actual}.txt",
            f"body_gyro_x_{actual}.txt",
            f"body_gyro_y_{actual}.txt",
            f"body_gyro_z_{actual}.txt",
            f"total_acc_x_{actual}.txt",
            f"total_acc_y_{actual}.txt",
            f"total_acc_z_{actual}.txt",
        ]

        signal_paths = [signal_dir / name for name in signal_names]
        if all(signal_path.exists() for signal_path in signal_paths):
            signals = [
                np.loadtxt(signal_path, dtype=np.float32)
                for signal_path in signal_paths
            ]
            # Shape: samples, timesteps, sensor channels = N x 128 x 9.
            self.features = np.stack(signals, axis=2)
        else:
            # Fallback for incomplete mirrors that contain only the official
            # 561 engineered feature vectors.
            x_path = extracted / actual / f"X_{actual}.txt"
            if not x_path.exists():
                raise DatasetError(
                    f"UCI HAR files are incomplete under {extracted}"
                )
            flat = np.loadtxt(x_path, dtype=np.float32)
            self.features = flat[:, None, :]

        self.labels = np.loadtxt(y_path, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.features)

    def get(self, index: int) -> np.ndarray:
        return self.features[index]

    def get_with_target(
        self,
        index: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        # Official UCI HAR labels are 1..6; convert to 0..5.
        target = int(self.labels[index]) - 1
        return (
            self.features[index],
            np.asarray(target, dtype=np.int64),
        )


HF_DATASETS: Dict[str, Dict[str, Any]] = {
    "imdb": {
        "paths": ["stanfordnlp/imdb", "imdb"],
        "config": None,
        "text_fields": ["text"],
        "split_map": {"train": "train", "test": "test", "validation": "test"},
    },
    "sst2": {
        "paths": ["stanfordnlp/sst2", "glue"],
        "configs": [None, "sst2"],
        "text_fields": ["sentence", "text"],
        "split_map": {
            "train": "train",
            "test": "validation",
            "validation": "validation",
        },
    },
    "ag_news": {
        "paths": ["fancyzhx/ag_news", "ag_news"],
        "config": None,
        "text_fields": ["text"],
        "split_map": {"train": "train", "test": "test", "validation": "test"},
    },
    "yelp_polarity": {
        "paths": ["fancyzhx/yelp_polarity", "yelp_polarity"],
        "config": None,
        "text_fields": ["text"],
        "split_map": {"train": "train", "test": "test", "validation": "test"},
    },
    "dbpedia_14": {
        "paths": ["fancyzhx/dbpedia_14", "dbpedia_14"],
        "config": None,
        "text_fields": ["content", "text"],
        "split_map": {"train": "train", "test": "test", "validation": "test"},
    },
    "amazon_polarity": {
        "paths": ["fancyzhx/amazon_polarity", "amazon_polarity"],
        "config": None,
        "text_fields": ["content", "text", "title"],
        "split_map": {"train": "train", "test": "test", "validation": "test"},
    },
    "tweet_eval_sentiment": {
        "paths": ["cardiffnlp/tweet_eval"],
        "config": "sentiment",
        "text_fields": ["text"],
        "split_map": {
            "train": "train",
            "test": "test",
            "validation": "validation",
        },
    },
}


class HuggingFaceTextSource(BaseSource):
    def __init__(
        self,
        name: str,
        root: Path,
        split: str,
        max_text_length: int,
        vocab_size: int,
        auto_download: bool,
    ) -> None:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise DatasetError(
                "Hugging Face dataset support requires the datasets package. "
                "Install it with: pip install datasets"
            ) from exc

        if not auto_download:
            os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

        entry = HF_DATASETS[name]
        actual_split = entry["split_map"].get(split, entry["split_map"]["test"])
        errors: List[str] = []

        configs: List[Optional[str]]
        if "configs" in entry:
            configs = list(entry["configs"])
        else:
            configs = [entry.get("config")]

        dataset = None
        for path in entry["paths"]:
            for config_name in configs:
                try:
                    kwargs = {
                        "path": path,
                        "split": actual_split,
                        "cache_dir": str(root / "huggingface"),
                    }
                    if config_name is not None:
                        kwargs["name"] = config_name
                    dataset = load_dataset(**kwargs)
                    break
                except Exception as exc:
                    errors.append(
                        f"{path}/{config_name}: {type(exc).__name__}: {exc}"
                    )
            if dataset is not None:
                break

        if dataset is None:
            joined = "\n".join(errors[-4:])
            raise DatasetError(
                f"Unable to load Hugging Face dataset {name!r}. "
                f"Recent attempts:\n{joined}"
            )

        self.dataset = dataset
        self.text_fields = entry["text_fields"]
        self.max_text_length = max_text_length
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        return len(self.dataset)

    def get(self, index: int) -> np.ndarray:
        item = self.dataset[index]
        texts = [
            str(item[field])
            for field in self.text_fields
            if field in item and item[field] is not None
        ]
        if not texts:
            raise DatasetError(
                f"No supported text field found in dataset item keys: {list(item)}"
            )

        text = " ".join(texts)
        return _tokenize_text(
            text=text,
            length=self.max_text_length,
            vocab_size=self.vocab_size,
        )

    def get_with_target(
        self,
        index: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        item = self.dataset[index]
        if "label" not in item:
            raise DatasetError(
                "Text dataset item does not contain a label field"
            )
        return (
            self.get(index),
            np.asarray(int(item["label"]), dtype=np.int64),
        )


class DatasetManager:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.name = config["ai"]["dataset"]
        self.spec = get_dataset_spec(self.name)
        self.rng = np.random.default_rng(int(config["execution"]["seed"]))
        self.source = self._build_source()
        self._indices = self._build_partition_indices()
        if bool(config["data"].get("shuffle", True)):
            self.rng.shuffle(self._indices)
        self._cursor = 0

    def _anomaly_labels(self) -> set[int]:
        values = self.config.get("anomaly_detection", {}).get("anomaly_labels", [9])
        return {int(value) for value in values}

    def _is_anomaly_label(self, label: int) -> bool:
        return int(label) in self._anomaly_labels()


    def _partition_label(self, index: int) -> int:
        # Partition labels are ground-truth data-management metadata only;
        # they never enter proxy fingerprinting predictors.
        method = getattr(self.source, "partition_label", None)
        if callable(method):
            return int(method(index))
        dataset = getattr(self.source, "dataset", None)
        if dataset is not None:
            for attr in ("targets", "labels"):
                values = getattr(dataset, attr, None)
                if values is not None:
                    value = values[index]
                    try:
                        return int(value.item())
                    except AttributeError:
                        return int(value)
            try:
                item = dataset[index]
                if isinstance(item, (tuple, list)) and len(item) >= 2:
                    value = item[1]
                    try:
                        return int(value.item())
                    except AttributeError:
                        return int(value)
            except Exception:
                pass
        labels = getattr(self.source, "labels", None)
        if labels is not None:
            value = labels[index]
            try:
                return int(value.item())
            except AttributeError:
                return int(value)
        hf = getattr(self.source, "dataset", None)
        if hf is not None:
            try:
                item = hf[index]
                if isinstance(item, dict) and "label" in item:
                    return int(item["label"])
            except Exception:
                pass
        try:
            _, target = self.source.get_with_target(index)
            array = np.asarray(target)
            if array.size == 1:
                return int(array.reshape(-1)[0])
        except Exception:
            pass
        raise DatasetError(
            f"Dataset {self.name!r} does not expose scalar class labels required "
            "for Dirichlet non-IID partitioning"
        )

    def _build_partition_indices(self) -> np.ndarray:
        all_indices = np.arange(len(self.source), dtype=np.int64)
        partition = self.config.get("data", {}).get("partition", {}) or {}
        deployment = str(self.config.get("execution", {}).get("deployment", ""))
        role = str(self.config.get("node", {}).get("role", ""))
        split = str(self.config.get("data", {}).get("split", "train"))
        application = str(self.config.get("ai", {}).get("application", ""))

        labels_all: np.ndarray | None = None
        anomaly_data_role = str(
            self.config.get("anomaly_detection", {}).get("data_role", "train")
        ).strip().lower()
        self._anomaly_calibration_indices = np.asarray([], dtype=np.int64)
        if application == "anomaly_detection" and split == "train":
            # Build one deterministic normal-only validation holdout before
            # client partitioning. These samples are never used for local
            # model training and the test split is never used to tune tau.
            labels_all = np.asarray(
                [self._partition_label(int(i)) for i in all_indices],
                dtype=np.int64,
            )
            anomaly_labels = self._anomaly_labels()
            normal_mask = np.asarray(
                [int(label) not in anomaly_labels for label in labels_all],
                dtype=bool,
            )
            normal_indices = all_indices[normal_mask]
            if normal_indices.size < 2:
                raise DatasetError(
                    "Anomaly-detection training requires at least two normal samples after "
                    f"excluding anomaly labels {sorted(anomaly_labels)}"
                )
            anomaly_cfg = self.config.get("anomaly_detection", {}) or {}
            fraction = float(anomaly_cfg.get("calibration_fraction", 0.10))
            base_seed = int(
                partition.get("seed")
                if partition.get("seed") is not None
                else self.config.get("execution", {}).get("seed", 42)
            )
            calibration_seed = base_seed + int(
                anomaly_cfg.get("calibration_seed_offset", 73001)
            )
            rng = np.random.default_rng(calibration_seed)
            shuffled = normal_indices.copy()
            rng.shuffle(shuffled)
            calibration_count = max(1, int(round(normal_indices.size * fraction)))
            calibration_count = min(calibration_count, normal_indices.size - 1)
            calibration_indices = np.sort(shuffled[:calibration_count])
            training_indices = np.sort(shuffled[calibration_count:])
            self._anomaly_calibration_indices = calibration_indices
            partition["anomaly_calibration_seed"] = calibration_seed
            partition["anomaly_calibration_count"] = int(calibration_indices.size)
            partition["anomaly_training_normal_count"] = int(training_indices.size)

            if anomaly_data_role == "calibration":
                self._partition_assignment = None
                return calibration_indices
            all_indices = training_indices

        # Partition only client-side federated model-training data. A held-out
        # anomaly calibration generator intentionally bypasses the client
        # partition so every evaluator sees the same fixed validation set.
        if not (
            deployment == "federated"
            and role == "client"
            and split == "train"
            and anomaly_data_role != "calibration"
        ):
            self._partition_assignment = None
            return all_indices

        client_count = max(int(partition.get("client_count", 1)), 1)
        client_index = int(partition.get("client_index", 0))
        seed = int(
            partition.get("seed")
            if partition.get("seed") is not None
            else self.config["execution"].get("seed", 42)
        )
        kind = str(partition.get("type", "iid")).strip().lower().replace("-", "_")
        if kind == "noniid":
            kind = "non_iid"
        alpha = float(partition.get("alpha", 0.5))

        if labels_all is None:
            labels_all = np.asarray(
                [self._partition_label(int(i)) for i in np.arange(len(self.source), dtype=np.int64)],
                dtype=np.int64,
            )
        partition_labels = labels_all[all_indices]
        assignment = make_partition_assignment(
            labels=partition_labels,
            partition_type=kind,
            client_index=client_index,
            client_count=client_count,
            seed=seed,
            alpha=alpha,
        )
        self._partition_assignment = assignment
        partition["type"] = assignment.partition_type
        partition["alpha"] = assignment.alpha if assignment.alpha is not None else alpha
        partition["seed"] = seed
        partition["client_count"] = client_count
        partition["client_index"] = client_index
        partition["assignment_id"] = assignment.assignment_id
        partition["disjoint"] = True
        return np.asarray(all_indices[assignment.indices], dtype=np.int64)

    def partition_summary(self) -> Dict[str, Any]:
        partition = self.config.get("data", {}).get("partition", {}) or {}
        assignment = getattr(self, "_partition_assignment", None)
        if assignment is not None:
            summary = assignment.summary()
            summary.update(
                {
                    "type": summary.pop("partition_type"),
                    "client_id": partition.get("client_id"),
                    "disjoint": True,
                    "source": partition.get("source", "server"),
                    "anomaly_labels": (
                        sorted(self._anomaly_labels())
                        if self.config.get("ai", {}).get("application") == "anomaly_detection"
                        else None
                    ),
                    "anomaly_training_excludes_labels": (
                        self.config.get("ai", {}).get("application") == "anomaly_detection"
                    ),
                    "anomaly_data_role": str(self.config.get("anomaly_detection", {}).get("data_role", "train")),
                    "anomaly_calibration_count": int(len(getattr(self, "_anomaly_calibration_indices", []))),
                }
            )
            return summary

        return {
            "type": str(partition.get("type", "iid")),
            "alpha": partition.get("alpha"),
            "seed": partition.get("seed"),
            "client_count": int(partition.get("client_count", 1)),
            "client_index": int(partition.get("client_index", 0)),
            "client_id": partition.get("client_id"),
            "disjoint": bool(partition.get("disjoint", True)),
            "sample_count": int(len(self._indices)),
            "class_histogram": {},
            "assignment_id": partition.get("assignment_id"),
            "anomaly_labels": (
                sorted(self._anomaly_labels())
                if self.config.get("ai", {}).get("application") == "anomaly_detection"
                else None
            ),
            "anomaly_training_excludes_labels": (
                self.config.get("ai", {}).get("application") == "anomaly_detection"
            ),
            "anomaly_data_role": str(self.config.get("anomaly_detection", {}).get("data_role", "train")),
            "anomaly_calibration_count": int(len(getattr(self, "_anomaly_calibration_indices", []))),
        }

    def _local_path_for(self, name: str) -> Optional[Path]:
        local_paths = self.config["data"].get("local_paths", {}) or {}
        value = local_paths.get(name)
        return Path(value).expanduser() if value else None

    def _build_source(self) -> BaseSource:
        spec = self.spec
        data = self.config["data"]
        ai = self.config["ai"]
        root = Path(data["root"]).expanduser()
        split = data.get("split") or spec.default_split
        auto_download = bool(data.get("auto_download", True))
        input_size = int(ai["input_size"])

        if spec.acquisition == "synthetic":
            return SyntheticSource(
                config=self.config,
                seed=int(self.config["execution"]["seed"]),
            )

        if self.name in {
            "mnist",
            "fashion_mnist",
            "cifar10",
            "cifar100",
            "svhn",
            "stl10",
            "food101",
            "oxford_iiit_pet",
            "flowers102",
            "dtd",
            "eurosat",
            "caltech101",
        }:
            return TorchvisionImageSource(
                name=self.name,
                root=root / self.name,
                split=split,
                input_size=input_size,
                auto_download=auto_download,
            )

        if self.name == "voc2007":
            return VOCDetectionSource(
                year="2007",
                root=root / self.name,
                split=split,
                input_size=input_size,
                auto_download=auto_download,
            )

        if self.name == "voc2012":
            return VOCDetectionSource(
                year="2012",
                root=root / self.name,
                split=split,
                input_size=input_size,
                auto_download=auto_download,
            )

        if self.name in {
            "voc2012_segmentation",
            "oxford_iiit_pet_segmentation",
        }:
            return TorchvisionSegmentationSource(
                name=self.name,
                root=root / self.name,
                split=split,
                input_size=input_size,
                auto_download=auto_download,
            )

        if self.name == "uci_har":
            return UCIHARSource(
                root=root,
                split=split,
                auto_download=auto_download,
            )

        if self.name in HF_DATASETS:
            return HuggingFaceTextSource(
                name=self.name,
                root=root,
                split=split,
                max_text_length=int(ai["max_text_length"]),
                vocab_size=int(ai["vocab_size"]),
                auto_download=auto_download,
            )

        if self.name in {"imagenet", "coco2017"}:
            local_path = self._local_path_for(self.name)
            if local_path is None:
                raise DatasetError(
                    f"{self.name} requires data.local_paths.{self.name} "
                    f"according to this layout: {spec.manual_layout}"
                )
            return LocalImageFolderSource(
                base_path=local_path,
                split=split,
                input_size=input_size,
            )

        raise DatasetError(
            f"No dataset source implementation for {self.name!r}"
        )

    def __len__(self) -> int:
        max_samples = self.config["data"].get("max_samples")
        available = len(self._indices)
        if max_samples is None:
            return available
        return min(available, int(max_samples))

    def _next_index(self) -> int:
        usable = len(self)
        if usable <= 0:
            raise DatasetError(f"Dataset {self.name!r} is empty")

        if self._cursor >= usable:
            self._cursor = 0
            if bool(self.config["data"].get("shuffle", True)):
                self.rng.shuffle(self._indices[:usable])

        index = int(self._indices[self._cursor])
        self._cursor += 1
        return index

    def reset(self) -> None:
        """Reset sampling to the beginning of the prepared index order."""
        self._cursor = 0

    def sample(self) -> np.ndarray:
        batch_size = int(self.config["execution"]["batch_size"])
        samples = [
            self.source.get(self._next_index())
            for _ in range(batch_size)
        ]
        try:
            return np.stack(samples, axis=0)
        except ValueError as exc:
            shapes = [tuple(sample.shape) for sample in samples]
            raise DatasetError(
                f"Cannot batch samples with shapes {shapes}"
            ) from exc

    def _next_index_for_anomaly(self, normal_only: bool | None) -> int:
        usable = len(self)
        if usable <= 0:
            raise DatasetError(f"Dataset {self.name!r} is empty")
        attempts = max(usable * 2, 16)
        for _ in range(attempts):
            index = self._next_index()
            binary = 1 if self._is_anomaly_label(self._partition_label(index)) else 0
            if normal_only is None or (normal_only and binary == 0) or (normal_only is False):
                return index
        requirement = "normal" if normal_only else "eligible"
        raise DatasetError(
            f"Unable to sample {requirement} anomaly-evaluation examples from {self.name!r}"
        )

    def sample_anomaly_batch(
        self,
        *,
        normal_only: bool | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return inputs, reconstruction targets, and binary anomaly labels.

        Labels are ground-truth evaluation metadata only and must never enter
        proxy-side fingerprinting predictors. 0=normal, 1=anomaly.
        """
        batch_size = int(self.config["execution"]["batch_size"])
        samples: list[np.ndarray] = []
        binary_labels: list[int] = []
        for _ in range(batch_size):
            index = self._next_index_for_anomaly(normal_only)
            sample = np.asarray(self.source.get(index), dtype=np.float32)
            class_label = self._partition_label(index)
            samples.append(sample)
            binary_labels.append(1 if self._is_anomaly_label(class_label) else 0)
        try:
            inputs = np.stack(samples, axis=0)
        except ValueError as exc:
            raise DatasetError(
                f"Cannot batch anomaly-evaluation samples with shapes "
                f"{[tuple(sample.shape) for sample in samples]}"
            ) from exc
        return (
            inputs,
            np.asarray(inputs, dtype=np.float32).copy(),
            np.asarray(binary_labels, dtype=np.int64),
        )

    def sample_training_batch(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:
        batch_size = int(self.config["execution"]["batch_size"])
        pairs = [
            self.source.get_with_target(self._next_index())
            for _ in range(batch_size)
        ]
        samples = [pair[0] for pair in pairs]
        application = str(self.config.get("ai", {}).get("application", ""))
        if application in {
            "reconstruction",
            "anomaly_detection",
            "image_denoising",
        }:
            # Autoencoder-style workloads reconstruct the input. Public image
            # datasets normally expose class labels as their second item;
            # those labels are not valid reconstruction targets.
            targets = [np.asarray(sample, dtype=np.float32) for sample in samples]
        else:
            targets = [pair[1] for pair in pairs]
        try:
            return (
                np.stack(samples, axis=0),
                np.stack(targets, axis=0),
            )
        except ValueError as exc:
            sample_shapes = [
                tuple(sample.shape)
                for sample in samples
            ]
            target_shapes = [
                tuple(np.asarray(target).shape)
                for target in targets
            ]
            raise DatasetError(
                f"Cannot batch training samples {sample_shapes} "
                f"with targets {target_shapes}"
            ) from exc


def prepare_dataset(
    name: str,
    root: str = "datasets",
    split: Optional[str] = None,
    input_size: int = 224,
) -> Dict[str, Any]:
    if name not in DATASETS:
        raise DatasetError(f"Unknown dataset: {name}")

    spec = DATASETS[name]
    if spec.acquisition == "manual":
        return {
            "name": name,
            "status": "manual",
            "message": spec.manual_layout or "Manual acquisition required.",
        }

    if spec.acquisition == "synthetic":
        return {
            "name": name,
            "status": "synthetic",
            "message": "No download required.",
        }

    application = spec.applications[0]
    family = (
        "rnn"
        if application == "activity_recognition"
        else "transformer"
        if application == "text_classification"
        else "gnn"
        if application in {"node_classification", "graph_classification"}
        else "cnn"
    )
    architecture = (
        "lstm"
        if application == "activity_recognition"
        else "tiny_transformer"
        if application == "text_classification"
        else "gcn"
        if application in {"node_classification", "graph_classification"}
        else "resnet"
    )
    variant = (
        "lstm_2layer"
        if application == "activity_recognition"
        else "tiny_transformer_2layer"
        if application == "text_classification"
        else "gcn_2layer"
        if application in {"node_classification", "graph_classification"}
        else "resnet18"
    )

    config: Dict[str, Any] = {
        "experiment": {
            "experiment_id": "DATASET_PREPARE",
            "output_dir": "experiments/results",
        },
        "node": {
            "role": "client",
            "host": "127.0.0.1",
            "port": 5000,
        },
        "transport": {
            "kind": "tcp",
            "certfile": None,
            "keyfile": None,
            "cafile": None,
            "verify_peer": False,
            "server_hostname": None,
        },
        "ai": {
            "framework": "pytorch",
            "runtime": "native",
            "family": family,
            "architecture": architecture,
            "variant": variant,
            "application": application,
            "dataset": name,
            "model_artifact": None,
            "num_classes": spec.num_classes or 10,
            "input_size": input_size,
            "sequence_length": 128,
            "input_dim": 9,
            "vocab_size": 10000,
            "max_text_length": 128,
            "graph_nodes": 32,
            "graph_features": 16,
        },
        "data": {
            "root": root,
            "split": split or spec.default_split,
            "auto_download": True,
            "shuffle": False,
            "max_samples": 1,
            "local_paths": {},
        },
        "execution": {
            "task": "inference",
            "deployment": "local",
            "operation": "workload",
            "repetitions": 1,
            "warmup": 0,
            "interval_ms": 0,
            "batch_size": 1,
            "precision": "fp32",
            "seed": 42,
            "epochs": 1,
            "steps_per_epoch": 1,
            "learning_rate": 0.001,
        },
        "federated": {
            "rounds": 1,
            "local_epochs": 1,
            "steps_per_epoch": 1,
            "expected_clients": 1,
            "client_id": "dataset_prepare",
            "aggregation": "fedavg",
        },
        "device": {
            "label": "custom",
            "operating_system": "unknown",
        },
    }

    manager = DatasetManager(config)
    sample = manager.sample()

    return {
        "name": name,
        "status": "ready",
        "samples": len(manager),
        "sample_shape": list(sample.shape),
        "root": str(Path(root).expanduser()),
    }
