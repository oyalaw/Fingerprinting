from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Set

from .dataset_catalog import DATASETS


@dataclass(frozen=True)
class VariantSpec:
    family: str
    architecture: str
    datasets_by_application: Mapping[str, List[str]]
    native_frameworks: Set[str]


IMAGE_CLASSIFICATION_DATASETS = [
    "synthetic_image",
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
    "imagenet",
]

TEXT_CLASSIFICATION_DATASETS = [
    "synthetic_text",
    "imdb",
    "sst2",
    "ag_news",
    "yelp_polarity",
    "dbpedia_14",
    "amazon_polarity",
    "tweet_eval_sentiment",
]

MASKED_LANGUAGE_MODELING_DATASETS = list(TEXT_CLASSIFICATION_DATASETS)


ACTIVITY_DATASETS = [
    "synthetic_sequence",
    "uci_har",
]

DETECTION_DATASETS = [
    "synthetic_detection",
    "voc2007",
    "voc2012",
    "coco2017",
]

SEGMENTATION_DATASETS = [
    "synthetic_segmentation",
    "voc2012_segmentation",
    "oxford_iiit_pet_segmentation",
]


RECONSTRUCTION_DATASETS = [
    "synthetic_image",
    "mnist",
    "fashion_mnist",
    "cifar10",
]

IMAGE_GENERATION_DATASETS = [
    "synthetic_image",
    "cifar10",
    "stl10",
]

GRAPH_DATASETS = [
    "synthetic_graph",
]


def _spec(
    family: str,
    architecture: str,
    applications: Mapping[str, List[str]],
    native_frameworks: Set[str],
) -> VariantSpec:
    return VariantSpec(
        family=family,
        architecture=architecture,
        datasets_by_application=applications,
        native_frameworks=native_frameworks,
    )


# The taxonomy is deliberately explicit:
#
# family -> architecture -> variant
#
# "variant" identifies the concrete model configuration being fingerprinted.
# Batch size, precision, input size, and runtime remain separate experimental
# attributes and are not model variants.
VARIANTS: Dict[str, VariantSpec] = {
    # CNN: ResNet
    "resnet18": _spec(
        "cnn", "resnet",
        {"image_classification": IMAGE_CLASSIFICATION_DATASETS},
        {"pytorch"},
    ),
    "resnet34": _spec(
        "cnn", "resnet",
        {"image_classification": IMAGE_CLASSIFICATION_DATASETS},
        {"pytorch"},
    ),
    "resnet50": _spec(
        "cnn", "resnet",
        {"image_classification": IMAGE_CLASSIFICATION_DATASETS},
        {"pytorch", "tensorflow"},
    ),
    "resnet101": _spec(
        "cnn", "resnet",
        {"image_classification": IMAGE_CLASSIFICATION_DATASETS},
        {"pytorch", "tensorflow"},
    ),

    # CNN: MobileNet
    "mobilenet_v2": _spec(
        "cnn", "mobilenet",
        {"image_classification": IMAGE_CLASSIFICATION_DATASETS},
        {"pytorch", "tensorflow"},
    ),
    "mobilenet_v3_small": _spec(
        "cnn", "mobilenet",
        {"image_classification": IMAGE_CLASSIFICATION_DATASETS},
        {"pytorch", "tensorflow"},
    ),
    "mobilenet_v3_large": _spec(
        "cnn", "mobilenet",
        {"image_classification": IMAGE_CLASSIFICATION_DATASETS},
        {"pytorch", "tensorflow"},
    ),

    # CNN: EfficientNet
    "efficientnet_b0": _spec(
        "cnn", "efficientnet",
        {"image_classification": IMAGE_CLASSIFICATION_DATASETS},
        {"pytorch", "tensorflow"},
    ),
    "efficientnet_b1": _spec(
        "cnn", "efficientnet",
        {"image_classification": IMAGE_CLASSIFICATION_DATASETS},
        {"pytorch", "tensorflow"},
    ),
    "efficientnet_b2": _spec(
        "cnn", "efficientnet",
        {"image_classification": IMAGE_CLASSIFICATION_DATASETS},
        {"pytorch", "tensorflow"},
    ),

    # CNN: Detection and segmentation. These are artifact-backed in v0.5.
    "yolov5s": _spec(
        "cnn", "yolo",
        {"object_detection": DETECTION_DATASETS},
        set(),
    ),
    "yolov5m": _spec(
        "cnn", "yolo",
        {"object_detection": DETECTION_DATASETS},
        set(),
    ),
    "yolov8n": _spec(
        "cnn", "yolo",
        {"object_detection": DETECTION_DATASETS},
        set(),
    ),
    "yolov8s": _spec(
        "cnn", "yolo",
        {"object_detection": DETECTION_DATASETS},
        set(),
    ),
    "faster_rcnn_resnet50_fpn": _spec(
        "cnn", "faster_rcnn",
        {"object_detection": DETECTION_DATASETS},
        set(),
    ),
    "unet_base": _spec(
        "cnn", "unet",
        {"image_segmentation": SEGMENTATION_DATASETS},
        set(),
    ),
    "deeplabv3_resnet50": _spec(
        "cnn", "deeplabv3",
        {"image_segmentation": SEGMENTATION_DATASETS},
        set(),
    ),

    # RNN: LSTM
    "lstm_1layer": _spec(
        "rnn", "lstm",
        {
            "activity_recognition": ACTIVITY_DATASETS,
            "text_classification": TEXT_CLASSIFICATION_DATASETS,
        },
        {"pytorch", "tensorflow"},
    ),
    "lstm_2layer": _spec(
        "rnn", "lstm",
        {
            "activity_recognition": ACTIVITY_DATASETS,
            "text_classification": TEXT_CLASSIFICATION_DATASETS,
        },
        {"pytorch", "tensorflow"},
    ),
    "bilstm_2layer": _spec(
        "rnn", "lstm",
        {
            "activity_recognition": ACTIVITY_DATASETS,
            "text_classification": TEXT_CLASSIFICATION_DATASETS,
        },
        {"pytorch", "tensorflow"},
    ),

    # RNN: GRU
    "gru_1layer": _spec(
        "rnn", "gru",
        {
            "activity_recognition": ACTIVITY_DATASETS,
            "text_classification": TEXT_CLASSIFICATION_DATASETS,
        },
        {"pytorch", "tensorflow"},
    ),
    "gru_2layer": _spec(
        "rnn", "gru",
        {
            "activity_recognition": ACTIVITY_DATASETS,
            "text_classification": TEXT_CLASSIFICATION_DATASETS,
        },
        {"pytorch", "tensorflow"},
    ),
    "bigru_2layer": _spec(
        "rnn", "gru",
        {
            "activity_recognition": ACTIVITY_DATASETS,
            "text_classification": TEXT_CLASSIFICATION_DATASETS,
        },
        {"pytorch", "tensorflow"},
    ),

    # Transformer. v0.8.7 adds real native branching so architecture and
    # variant fingerprinting are not deterministic single-child decisions.
    "tiny_transformer_2layer": _spec(
        "transformer", "tiny_transformer",
        {
            "text_classification": TEXT_CLASSIFICATION_DATASETS,
            "masked_language_modeling": MASKED_LANGUAGE_MODELING_DATASETS,
        },
        {"pytorch", "tensorflow"},
    ),
    "tiny_transformer_4layer": _spec(
        "transformer", "tiny_transformer",
        {
            "text_classification": TEXT_CLASSIFICATION_DATASETS,
            "masked_language_modeling": MASKED_LANGUAGE_MODELING_DATASETS,
        },
        {"pytorch", "tensorflow"},
    ),
    "tiny_transformer_6layer": _spec(
        "transformer", "tiny_transformer",
        {
            "text_classification": TEXT_CLASSIFICATION_DATASETS,
            "masked_language_modeling": MASKED_LANGUAGE_MODELING_DATASETS,
        },
        {"pytorch", "tensorflow"},
    ),
    # Hugging Face-backed PyTorch BERT/DistilBERT models are instantiated
    # from config with random weights, so no model download is required.
    "bert_tiny": _spec(
        "transformer", "bert",
        {
            "text_classification": TEXT_CLASSIFICATION_DATASETS,
            "masked_language_modeling": MASKED_LANGUAGE_MODELING_DATASETS,
        },
        {"pytorch"},
    ),
    "bert_mini": _spec(
        "transformer", "bert",
        {
            "text_classification": TEXT_CLASSIFICATION_DATASETS,
            "masked_language_modeling": MASKED_LANGUAGE_MODELING_DATASETS,
        },
        {"pytorch"},
    ),
    "bert_small": _spec(
        "transformer", "bert",
        {
            "text_classification": TEXT_CLASSIFICATION_DATASETS,
            "masked_language_modeling": MASKED_LANGUAGE_MODELING_DATASETS,
        },
        {"pytorch"},
    ),
    "bert_base": _spec(
        "transformer", "bert",
        {
            "text_classification": TEXT_CLASSIFICATION_DATASETS,
            "masked_language_modeling": MASKED_LANGUAGE_MODELING_DATASETS,
        },
        {"pytorch"},
    ),
    "bert_large": _spec(
        "transformer", "bert",
        {
            "text_classification": TEXT_CLASSIFICATION_DATASETS,
            "masked_language_modeling": MASKED_LANGUAGE_MODELING_DATASETS,
        },
        {"pytorch"},
    ),
    "distilbert_base": _spec(
        "transformer", "distilbert",
        {
            "text_classification": TEXT_CLASSIFICATION_DATASETS,
            "masked_language_modeling": MASKED_LANGUAGE_MODELING_DATASETS,
        },
        {"pytorch"},
    ),
    # torchvision Vision Transformers provide a same-modality control against
    # CNN image classifiers.
    "vit_b16": _spec(
        "transformer", "vit",
        {"image_classification": IMAGE_CLASSIFICATION_DATASETS},
        {"pytorch"},
    ),
    "vit_b32": _spec(
        "transformer", "vit",
        {"image_classification": IMAGE_CLASSIFICATION_DATASETS},
        {"pytorch"},
    ),
    "vit_l16": _spec(
        "transformer", "vit",
        {"image_classification": IMAGE_CLASSIFICATION_DATASETS},
        {"pytorch"},
    ),
    "detr_resnet50": _spec(
        "transformer", "detr",
        {"object_detection": DETECTION_DATASETS},
        set(),
    ),

    # Autoencoders. All three architectures now have native executable
    # variants; dense and convolutional branches also contain depth variants.
    "dense_autoencoder_2layer": _spec(
        "autoencoder", "dense_autoencoder",
        {
            "reconstruction": RECONSTRUCTION_DATASETS,
            "anomaly_detection": RECONSTRUCTION_DATASETS,
        },
        {"pytorch", "tensorflow"},
    ),
    "dense_autoencoder_3layer": _spec(
        "autoencoder", "dense_autoencoder",
        {
            "reconstruction": RECONSTRUCTION_DATASETS,
            "anomaly_detection": RECONSTRUCTION_DATASETS,
        },
        {"pytorch", "tensorflow"},
    ),
    "dense_autoencoder_5layer": _spec(
        "autoencoder", "dense_autoencoder",
        {
            "reconstruction": RECONSTRUCTION_DATASETS,
            "anomaly_detection": RECONSTRUCTION_DATASETS,
        },
        {"pytorch", "tensorflow"},
    ),
    "convolutional_autoencoder_2layer": _spec(
        "autoencoder", "convolutional_autoencoder",
        {
            "reconstruction": RECONSTRUCTION_DATASETS,
            "anomaly_detection": RECONSTRUCTION_DATASETS,
        },
        {"pytorch", "tensorflow"},
    ),
    "convolutional_autoencoder_4layer": _spec(
        "autoencoder", "convolutional_autoencoder",
        {
            "reconstruction": RECONSTRUCTION_DATASETS,
            "anomaly_detection": RECONSTRUCTION_DATASETS,
        },
        {"pytorch", "tensorflow"},
    ),
    "convolutional_autoencoder_6layer": _spec(
        "autoencoder", "convolutional_autoencoder",
        {
            "reconstruction": RECONSTRUCTION_DATASETS,
            "anomaly_detection": RECONSTRUCTION_DATASETS,
        },
        {"pytorch", "tensorflow"},
    ),
    "vae_fc": _spec(
        "autoencoder", "variational_autoencoder",
        {
            "reconstruction": RECONSTRUCTION_DATASETS,
            "anomaly_detection": RECONSTRUCTION_DATASETS,
        },
        {"pytorch", "tensorflow"},
    ),
    "vae_conv": _spec(
        "autoencoder", "variational_autoencoder",
        {
            "reconstruction": RECONSTRUCTION_DATASETS,
            "anomaly_detection": RECONSTRUCTION_DATASETS,
        },
        {"pytorch", "tensorflow"},
    ),
    "beta_vae": _spec(
        "autoencoder", "variational_autoencoder",
        {
            "reconstruction": RECONSTRUCTION_DATASETS,
            "anomaly_detection": RECONSTRUCTION_DATASETS,
        },
        {"pytorch", "tensorflow"},
    ),

    # Graph neural networks. v0.5 catalogs these for hierarchical
    # fingerprinting and artifact-backed execution.
    "gcn_2layer": _spec(
        "gnn", "gcn",
        {
            "node_classification": GRAPH_DATASETS,
            "graph_classification": GRAPH_DATASETS,
        },
        set(),
    ),
    "gat_8head": _spec(
        "gnn", "gat",
        {
            "node_classification": GRAPH_DATASETS,
            "graph_classification": GRAPH_DATASETS,
        },
        set(),
    ),
    "graphsage_mean": _spec(
        "gnn", "graphsage",
        {
            "node_classification": GRAPH_DATASETS,
            "graph_classification": GRAPH_DATASETS,
        },
        set(),
    ),
    "gin_5layer": _spec(
        "gnn", "gin",
        {"graph_classification": GRAPH_DATASETS},
        set(),
    ),

    # Diffusion models
    "ddpm_unet_small": _spec(
        "diffusion", "pixel_diffusion",
        {
            "image_generation": IMAGE_GENERATION_DATASETS,
            "image_denoising": IMAGE_GENERATION_DATASETS,
        },
        set(),
    ),
    "ddpm_unet_base": _spec(
        "diffusion", "pixel_diffusion",
        {
            "image_generation": IMAGE_GENERATION_DATASETS,
            "image_denoising": IMAGE_GENERATION_DATASETS,
        },
        set(),
    ),
    "ldm_base": _spec(
        "diffusion", "latent_diffusion",
        {"image_generation": IMAGE_GENERATION_DATASETS},
        set(),
    ),
    "stable_diffusion_v1_5": _spec(
        "diffusion", "latent_diffusion",
        {"image_generation": IMAGE_GENERATION_DATASETS},
        set(),
    ),
    "stable_diffusion_v2_1": _spec(
        "diffusion", "latent_diffusion",
        {"image_generation": IMAGE_GENERATION_DATASETS},
        set(),
    ),

    # Generative adversarial networks
    "dcgan_64": _spec(
        "gan", "dcgan",
        {"image_generation": IMAGE_GENERATION_DATASETS},
        set(),
    ),
    "wgan_gp_64": _spec(
        "gan", "wgan",
        {"image_generation": IMAGE_GENERATION_DATASETS},
        set(),
    ),
    "stylegan2_ada": _spec(
        "gan", "stylegan",
        {"image_generation": IMAGE_GENERATION_DATASETS},
        set(),
    ),

    # Feedforward networks
    "mlp_2layer": _spec(
        "mlp", "feedforward_mlp",
        {
            "image_classification": IMAGE_CLASSIFICATION_DATASETS,
            "activity_recognition": ACTIVITY_DATASETS,
        },
        {"pytorch", "tensorflow"},
    ),
    "mlp_4layer": _spec(
        "mlp", "feedforward_mlp",
        {
            "image_classification": IMAGE_CLASSIFICATION_DATASETS,
            "activity_recognition": ACTIVITY_DATASETS,
        },
        {"pytorch", "tensorflow"},
    ),
    "mlp_8layer": _spec(
        "mlp", "feedforward_mlp",
        {
            "image_classification": IMAGE_CLASSIFICATION_DATASETS,
            "activity_recognition": ACTIVITY_DATASETS,
        },
        {"pytorch", "tensorflow"},
    ),

    # State-space sequence models
    "mamba_tiny": _spec(
        "state_space", "mamba",
        {"text_classification": TEXT_CLASSIFICATION_DATASETS},
        set(),
    ),
    "mamba_small": _spec(
        "state_space", "mamba",
        {"text_classification": TEXT_CLASSIFICATION_DATASETS},
        set(),
    ),
}


# Backward compatibility for v0.4 configuration files where the concrete model
# was stored in ai.architecture and ai.variant did not exist.
LEGACY_VARIANT_TO_ARCHITECTURE: Dict[str, str] = {
    name: spec.architecture for name, spec in VARIANTS.items()
}
# Historical shorthand labels from pre-hierarchy configurations.
LEGACY_VARIANT_TO_ARCHITECTURE.update({
    "yolov5": "yolo",
    "yolov8": "yolo",
    "unet": "unet",
    "lstm": "lstm",
    "gru": "gru",
    "tiny_transformer": "tiny_transformer",
    "bert": "bert",
    "distilbert": "distilbert",
    "dense_autoencoder": "dense_autoencoder",
    "convolutional_autoencoder": "convolutional_autoencoder",
    "vae": "variational_autoencoder",
    "mlp": "feedforward_mlp",
})

LEGACY_VARIANT_RENAMES: Dict[str, str] = {
    "yolov5": "yolov5s",
    "yolov8": "yolov8n",
    "unet": "unet_base",
    "lstm": "lstm_2layer",
    "gru": "gru_2layer",
    "tiny_transformer": "tiny_transformer_2layer",
    "bert": "bert_base",
    "distilbert": "distilbert_base",
    "dense_autoencoder": "dense_autoencoder_3layer",
    "convolutional_autoencoder": "convolutional_autoencoder_4layer",
    "vae": "vae_fc",
    "mlp": "mlp_4layer",
}


FRAMEWORKS: Dict[str, Set[str]] = {
    "pytorch": set(VARIANTS),
    "tensorflow": set(VARIANTS),
}

RUNTIMES = ["native", "tflite", "onnxruntime", "tensorrt"]

DEVICES = [
    "jetson_agx_orin",
    "jetson_orin_nano",
    "jetson_xavier",
    "dell_desktop",
    "dell_laptop",
    "generic_linux_desktop",
    "generic_windows_desktop",
    "raspberry_pi_5",
    "android_phone",
    "iphone",
    "ipad",
    "macbook",
    "custom",
]

# Operating-system labels are deliberately separate from hardware labels.
# The interactive workflow presents a numbered OS menu after the device is
# selected so users do not have to free-type routine operating-system values.
OPERATING_SYSTEMS = [
    "ubuntu",
    "debian",
    "fedora",
    "rhel",
    "rocky_linux",
    "almalinux",
    "arch_linux",
    "raspberry_pi_os",
    "windows_10",
    "windows_11",
    "macos",
    "android",
    "ios",
    "ipados",
    "other_linux",
    "custom",
]

OPERATING_SYSTEMS_BY_DEVICE = {
    "jetson_agx_orin": ["ubuntu", "other_linux", "custom"],
    "jetson_orin_nano": ["ubuntu", "other_linux", "custom"],
    "jetson_xavier": ["ubuntu", "other_linux", "custom"],
    "dell_desktop": [
        "ubuntu",
        "windows_11",
        "windows_10",
        "debian",
        "fedora",
        "other_linux",
        "custom",
    ],
    "dell_laptop": [
        "ubuntu",
        "windows_11",
        "windows_10",
        "debian",
        "fedora",
        "other_linux",
        "custom",
    ],
    "generic_linux_desktop": [
        "ubuntu",
        "debian",
        "fedora",
        "rhel",
        "rocky_linux",
        "almalinux",
        "arch_linux",
        "other_linux",
        "custom",
    ],
    "generic_windows_desktop": ["windows_11", "windows_10", "custom"],
    "raspberry_pi_5": [
        "raspberry_pi_os",
        "ubuntu",
        "other_linux",
        "custom",
    ],
    "android_phone": ["android", "custom"],
    "iphone": ["ios", "custom"],
    "ipad": ["ipados", "ios", "custom"],
    "macbook": ["macos", "ubuntu", "other_linux", "custom"],
}

EXECUTION_TASKS = ["inference", "training"]

DEPLOYMENTS_BY_TASK = {
    "inference": ["local", "remote"],
    "training": ["local", "remote", "federated"],
}

TRAINABLE_APPLICATIONS = {
    "image_classification",
    "activity_recognition",
    "text_classification",
    "masked_language_modeling",
    "reconstruction",
    "anomaly_detection",
}


def frameworks() -> List[str]:
    return sorted(FRAMEWORKS)


def families() -> List[str]:
    return sorted({spec.family for spec in VARIANTS.values()})


def operating_systems_for_device(device_label: str) -> List[str]:
    """Return interactive OS choices appropriate for the selected hardware.

    Unknown/custom hardware gets the complete OS catalogue. ``custom`` is
    retained as an escape hatch for research platforms whose OS is not yet in
    the registry, while routine configurations remain fully menu-driven.
    """
    choices = OPERATING_SYSTEMS_BY_DEVICE.get(device_label)
    if choices is None:
        return list(OPERATING_SYSTEMS)
    return list(choices)


def _available_variants(
    framework: str,
    runtime: str,
) -> Set[str]:
    allowed = set(FRAMEWORKS.get(framework, set()))
    if runtime == "native":
        return {
            name
            for name in allowed
            if framework in VARIANTS[name].native_frameworks
        }
    return allowed


def families_for_framework(
    framework: str,
    runtime: str = "native",
) -> List[str]:
    allowed = _available_variants(framework, runtime)
    return sorted({VARIANTS[name].family for name in allowed})


def architectures_for_family(family: str) -> List[str]:
    return sorted(
        {
            spec.architecture
            for spec in VARIANTS.values()
            if spec.family == family
        }
    )


def architectures_for(
    framework: str,
    family: str,
    runtime: str = "native",
) -> List[str]:
    allowed = _available_variants(framework, runtime)
    return sorted(
        {
            VARIANTS[name].architecture
            for name in allowed
            if VARIANTS[name].family == family
        }
    )


def variants_for_architecture(architecture: str) -> List[str]:
    return sorted(
        name
        for name, spec in VARIANTS.items()
        if spec.architecture == architecture
    )


def variants_for(
    framework: str,
    family: str,
    architecture: str,
    runtime: str = "native",
) -> List[str]:
    allowed = _available_variants(framework, runtime)
    return sorted(
        name
        for name in allowed
        if (
            VARIANTS[name].family == family
            and VARIANTS[name].architecture == architecture
        )
    )


def applications_for(
    architecture: str,
    variant: str | None = None,
) -> List[str]:
    if variant is not None:
        spec = VARIANTS[variant]
        if spec.architecture != architecture:
            return []
        return sorted(spec.datasets_by_application)

    applications = set()
    for name in variants_for_architecture(architecture):
        applications.update(VARIANTS[name].datasets_by_application)
    return sorted(applications)


def datasets_for(
    architecture: str,
    application: str,
    variant: str | None = None,
) -> List[str]:
    if variant is not None:
        spec = VARIANTS[variant]
        if spec.architecture != architecture:
            return []
        datasets = list(spec.datasets_by_application.get(application, []))
        return [name for name in datasets if name in DATASETS]

    datasets = set()
    for name in variants_for_architecture(architecture):
        datasets.update(
            VARIANTS[name].datasets_by_application.get(application, [])
        )
    return sorted(name for name in datasets if name in DATASETS)


def native_supported(framework: str, variant: str) -> bool:
    return framework in VARIANTS[variant].native_frameworks


def requires_artifact(
    framework: str,
    runtime: str,
    variant: str,
) -> bool:
    if runtime in {"tflite", "onnxruntime", "tensorrt"}:
        return True
    return not native_supported(framework, variant)


def hierarchy_for_variant(variant: str) -> Dict[str, str]:
    spec = VARIANTS[variant]
    return {
        "family": spec.family,
        "architecture": spec.architecture,
        "variant": variant,
    }


def upgrade_legacy_model_labels(
    architecture: str,
) -> tuple[str, str] | None:
    if architecture not in LEGACY_VARIANT_TO_ARCHITECTURE:
        return None

    variant = LEGACY_VARIANT_RENAMES.get(architecture, architecture)
    return LEGACY_VARIANT_TO_ARCHITECTURE[architecture], variant



OPTIONAL_VARIANT_DEPENDENCIES: Dict[str, List[str]] = {
    "bert_tiny": ["transformers"],
    "bert_mini": ["transformers"],
    "bert_small": ["transformers"],
    "bert_base": ["transformers"],
    "bert_large": ["transformers"],
    "distilbert_base": ["transformers"],
}


def variant_execution_status(
    framework: str,
    variant: str,
) -> Dict[str, object]:
    """Return explicit catalog/native availability for UI and audits."""
    spec = VARIANTS[variant]
    native = framework in spec.native_frameworks
    dependencies = list(OPTIONAL_VARIANT_DEPENDENCIES.get(variant, []))
    return {
        "family": spec.family,
        "architecture": spec.architecture,
        "variant": variant,
        "framework": framework,
        "native": native,
        "artifact_only": not native,
        "optional_dependencies": dependencies,
    }


def model_catalog_rows(framework: str) -> List[Dict[str, object]]:
    return [
        variant_execution_status(framework, name)
        for name in sorted(VARIANTS)
    ]

def deployments_for_task(task: str) -> List[str]:
    return list(DEPLOYMENTS_BY_TASK.get(task, []))
