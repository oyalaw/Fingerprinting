from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    applications: Tuple[str, ...]
    modality: str
    source: str
    acquisition: str
    size_tier: str
    num_classes: Optional[int]
    default_split: str
    description: str
    manual_layout: Optional[str] = None
    aliases: Tuple[str, ...] = field(default_factory=tuple)


DATASETS: Dict[str, DatasetSpec] = {
    "synthetic_image": DatasetSpec(
        name="synthetic_image",
        applications=("image_classification",),
        modality="image",
        source="generated",
        acquisition="synthetic",
        size_tier="tiny",
        num_classes=10,
        default_split="test",
        description="Generated RGB images for smoke tests and controlled baselines.",
    ),
    "mnist": DatasetSpec(
        name="mnist",
        applications=("image_classification", "reconstruction"),
        modality="image",
        source="torchvision",
        acquisition="automatic",
        size_tier="small",
        num_classes=10,
        default_split="test",
        description="Handwritten digit images.",
    ),
    "fashion_mnist": DatasetSpec(
        name="fashion_mnist",
        applications=("image_classification", "reconstruction"),
        modality="image",
        source="torchvision",
        acquisition="automatic",
        size_tier="small",
        num_classes=10,
        default_split="test",
        description="Fashion article images in the MNIST format.",
    ),
    "cifar10": DatasetSpec(
        name="cifar10",
        applications=("image_classification", "reconstruction", "image_generation"),
        modality="image",
        source="torchvision",
        acquisition="automatic",
        size_tier="small",
        num_classes=10,
        default_split="test",
        description="Ten class natural image dataset.",
    ),
    "cifar100": DatasetSpec(
        name="cifar100",
        applications=("image_classification",),
        modality="image",
        source="torchvision",
        acquisition="automatic",
        size_tier="small",
        num_classes=100,
        default_split="test",
        description="One hundred class natural image dataset.",
    ),
    "svhn": DatasetSpec(
        name="svhn",
        applications=("image_classification",),
        modality="image",
        source="torchvision",
        acquisition="automatic",
        size_tier="medium",
        num_classes=10,
        default_split="test",
        description="Street View House Numbers.",
    ),
    "stl10": DatasetSpec(
        name="stl10",
        applications=("image_classification",),
        modality="image",
        source="torchvision",
        acquisition="automatic",
        size_tier="medium",
        num_classes=10,
        default_split="test",
        description="Natural images for representation learning and classification.",
    ),
    "food101": DatasetSpec(
        name="food101",
        applications=("image_classification",),
        modality="image",
        source="torchvision",
        acquisition="automatic",
        size_tier="large",
        num_classes=101,
        default_split="test",
        description="Food image classification dataset.",
    ),
    "oxford_iiit_pet": DatasetSpec(
        name="oxford_iiit_pet",
        applications=("image_classification",),
        modality="image",
        source="torchvision",
        acquisition="automatic",
        size_tier="medium",
        num_classes=37,
        default_split="test",
        description="Oxford IIIT Pet breed classification dataset.",
    ),
    "flowers102": DatasetSpec(
        name="flowers102",
        applications=("image_classification",),
        modality="image",
        source="torchvision",
        acquisition="automatic",
        size_tier="medium",
        num_classes=102,
        default_split="test",
        description="Oxford 102 category flower dataset.",
    ),
    "dtd": DatasetSpec(
        name="dtd",
        applications=("image_classification",),
        modality="image",
        source="torchvision",
        acquisition="automatic",
        size_tier="medium",
        num_classes=47,
        default_split="test",
        description="Describable Textures Dataset.",
    ),
    "eurosat": DatasetSpec(
        name="eurosat",
        applications=("image_classification",),
        modality="image",
        source="torchvision",
        acquisition="automatic",
        size_tier="medium",
        num_classes=10,
        default_split="test",
        description="Satellite image land use classification dataset.",
    ),
    "caltech101": DatasetSpec(
        name="caltech101",
        applications=("image_classification",),
        modality="image",
        source="torchvision",
        acquisition="automatic",
        size_tier="medium",
        num_classes=101,
        default_split="all",
        description="Object category image dataset.",
    ),
    "imagenet": DatasetSpec(
        name="imagenet",
        applications=("image_classification",),
        modality="image",
        source="local_imagefolder",
        acquisition="manual",
        size_tier="very_large",
        num_classes=1000,
        default_split="val",
        description="ImageNet ILSVRC style image classification dataset.",
        manual_layout="<path>/train/<class>/*.JPEG and <path>/val/<class>/*.JPEG",
    ),
    "synthetic_detection": DatasetSpec(
        name="synthetic_detection",
        applications=("object_detection",),
        modality="image",
        source="generated",
        acquisition="synthetic",
        size_tier="tiny",
        num_classes=10,
        default_split="test",
        description="Generated RGB frames for detection pipeline smoke tests.",
    ),
    "voc2007": DatasetSpec(
        name="voc2007",
        applications=("object_detection",),
        modality="image",
        source="torchvision",
        acquisition="automatic",
        size_tier="medium",
        num_classes=20,
        default_split="test",
        description="PASCAL VOC 2007 object detection.",
    ),
    "voc2012": DatasetSpec(
        name="voc2012",
        applications=("object_detection",),
        modality="image",
        source="torchvision",
        acquisition="automatic",
        size_tier="medium",
        num_classes=20,
        default_split="val",
        description="PASCAL VOC 2012 object detection.",
    ),
    "coco2017": DatasetSpec(
        name="coco2017",
        applications=("object_detection",),
        modality="image",
        source="local_image_directory",
        acquisition="manual",
        size_tier="very_large",
        num_classes=80,
        default_split="val",
        description="COCO 2017 object detection images.",
        manual_layout="<path>/train2017/*.jpg or <path>/val2017/*.jpg",
    ),
    "synthetic_segmentation": DatasetSpec(
        name="synthetic_segmentation",
        applications=("image_segmentation",),
        modality="image",
        source="generated",
        acquisition="synthetic",
        size_tier="tiny",
        num_classes=2,
        default_split="test",
        description="Generated RGB frames for segmentation pipeline smoke tests.",
    ),
    "voc2012_segmentation": DatasetSpec(
        name="voc2012_segmentation",
        applications=("image_segmentation",),
        modality="image",
        source="torchvision",
        acquisition="automatic",
        size_tier="medium",
        num_classes=21,
        default_split="val",
        description="PASCAL VOC 2012 semantic segmentation.",
    ),
    "oxford_iiit_pet_segmentation": DatasetSpec(
        name="oxford_iiit_pet_segmentation",
        applications=("image_segmentation",),
        modality="image",
        source="torchvision",
        acquisition="automatic",
        size_tier="medium",
        num_classes=3,
        default_split="test",
        description="Oxford IIIT Pet trimap segmentation.",
    ),
    "synthetic_sequence": DatasetSpec(
        name="synthetic_sequence",
        applications=("activity_recognition",),
        modality="sequence",
        source="generated",
        acquisition="synthetic",
        size_tier="tiny",
        num_classes=6,
        default_split="test",
        description="Generated multivariate sensor sequences.",
    ),
    "uci_har": DatasetSpec(
        name="uci_har",
        applications=("activity_recognition",),
        modality="sequence",
        source="uci",
        acquisition="automatic",
        size_tier="small",
        num_classes=6,
        default_split="test",
        description="Human Activity Recognition Using Smartphones.",
    ),
    "synthetic_graph": DatasetSpec(
        name="synthetic_graph",
        applications=("node_classification", "graph_classification"),
        modality="graph_tensor",
        source="generated",
        acquisition="synthetic",
        size_tier="tiny",
        num_classes=3,
        default_split="test",
        description=(
            "Generated packed graph tensors for artifact-backed GNN "
            "fingerprinting smoke tests."
        ),
    ),
    "synthetic_text": DatasetSpec(
        name="synthetic_text",
        applications=("text_classification",),
        modality="text",
        source="generated",
        acquisition="synthetic",
        size_tier="tiny",
        num_classes=2,
        default_split="test",
        description="Generated token sequences for smoke tests.",
    ),
    "imdb": DatasetSpec(
        name="imdb",
        applications=("text_classification",),
        modality="text",
        source="huggingface",
        acquisition="automatic",
        size_tier="medium",
        num_classes=2,
        default_split="test",
        description="IMDB movie review sentiment classification.",
    ),
    "sst2": DatasetSpec(
        name="sst2",
        applications=("text_classification",),
        modality="text",
        source="huggingface",
        acquisition="automatic",
        size_tier="small",
        num_classes=2,
        default_split="validation",
        description="Stanford Sentiment Treebank binary sentiment task.",
    ),
    "ag_news": DatasetSpec(
        name="ag_news",
        applications=("text_classification",),
        modality="text",
        source="huggingface",
        acquisition="automatic",
        size_tier="medium",
        num_classes=4,
        default_split="test",
        description="Four class news topic classification.",
    ),
    "yelp_polarity": DatasetSpec(
        name="yelp_polarity",
        applications=("text_classification",),
        modality="text",
        source="huggingface",
        acquisition="automatic",
        size_tier="large",
        num_classes=2,
        default_split="test",
        description="Yelp review polarity classification.",
    ),
    "dbpedia_14": DatasetSpec(
        name="dbpedia_14",
        applications=("text_classification",),
        modality="text",
        source="huggingface",
        acquisition="automatic",
        size_tier="large",
        num_classes=14,
        default_split="test",
        description="DBpedia ontology classification.",
    ),
    "amazon_polarity": DatasetSpec(
        name="amazon_polarity",
        applications=("text_classification",),
        modality="text",
        source="huggingface",
        acquisition="automatic",
        size_tier="very_large",
        num_classes=2,
        default_split="test",
        description="Amazon review polarity classification.",
    ),
    "tweet_eval_sentiment": DatasetSpec(
        name="tweet_eval_sentiment",
        applications=("text_classification",),
        modality="text",
        source="huggingface",
        acquisition="automatic",
        size_tier="small",
        num_classes=3,
        default_split="test",
        description="TweetEval three class sentiment classification.",
    ),
}


SIZE_ORDER = {
    "tiny": 0,
    "small": 1,
    "medium": 2,
    "large": 3,
    "very_large": 4,
}


def dataset_names() -> List[str]:
    return sorted(DATASETS)


def get_dataset_spec(name: str) -> DatasetSpec:
    try:
        return DATASETS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown dataset: {name}") from exc


def datasets_for_application(application: str) -> List[str]:
    return sorted(
        name
        for name, spec in DATASETS.items()
        if application in spec.applications
    )


def automatic_datasets(max_tier: str = "medium") -> List[str]:
    if max_tier not in SIZE_ORDER:
        raise ValueError(f"Unknown size tier: {max_tier}")
    limit = SIZE_ORDER[max_tier]
    return sorted(
        name
        for name, spec in DATASETS.items()
        if spec.acquisition == "automatic"
        and SIZE_ORDER[spec.size_tier] <= limit
    )
