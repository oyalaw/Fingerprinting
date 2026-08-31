from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .base import Workload


class PyTorchWorkload(Workload):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch backend requested but torch is not installed"
            ) from exc

        self.torch = torch
        self.nn = nn
        self.architecture = config["ai"]["architecture"]
        self.variant = config["ai"]["variant"]
        self.application = config["ai"]["application"]
        self.num_classes = int(config["ai"]["num_classes"])
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = self._build_model().to(self.device).eval()
        self.learning_rate = float(
            config["execution"].get("learning_rate", 1e-3)
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
        )

    def _build_model(self):
        nn = self.nn
        arch = self.architecture
        variant = self.variant
        ai = self.config["ai"]
        input_size = int(ai["input_size"])

        if arch in {"resnet", "mobilenet", "efficientnet"}:
            try:
                from torchvision import models
            except ImportError as exc:
                raise RuntimeError(
                    "CNN architecture requested but torchvision "
                    "is not installed"
                ) from exc

            builders = {
                "resnet18": models.resnet18,
                "resnet34": models.resnet34,
                "resnet50": models.resnet50,
                "resnet101": models.resnet101,
                "mobilenet_v2": models.mobilenet_v2,
                "mobilenet_v3_small": models.mobilenet_v3_small,
                "mobilenet_v3_large": models.mobilenet_v3_large,
                "efficientnet_b0": models.efficientnet_b0,
                "efficientnet_b1": models.efficientnet_b1,
                "efficientnet_b2": models.efficientnet_b2,
            }
            builder = builders.get(variant)
            if builder is None:
                raise ValueError(
                    f"Native PyTorch builder not implemented for "
                    f"architecture={arch!r}, variant={variant!r}"
                )
            return builder(weights=None, num_classes=self.num_classes)

        if arch == "vit":
            try:
                from torchvision import models
            except ImportError as exc:
                raise RuntimeError(
                    "Vision Transformer requested but torchvision "
                    "is not installed"
                ) from exc
            builders = {
                "vit_b16": models.vit_b_16,
                "vit_b32": models.vit_b_32,
                "vit_l16": models.vit_l_16,
            }
            builder = builders.get(variant)
            if builder is None:
                raise ValueError(f"Unsupported ViT variant: {variant!r}")
            patch = 32 if variant == "vit_b32" else 16
            if input_size % patch:
                raise ValueError(
                    f"{variant} requires ai.input_size divisible by {patch}; "
                    f"received {input_size}"
                )
            return builder(
                weights=None,
                image_size=input_size,
                num_classes=self.num_classes,
            )

        if arch in {"lstm", "gru"}:
            input_dim = int(ai["input_dim"])
            hidden = 128
            layers, bidirectional = _rnn_variant_settings(
                architecture=arch,
                variant=variant,
            )

            if self.application == "text_classification":
                vocab = int(ai["vocab_size"])
                return _TorchTextRNN(
                    nn=nn,
                    rnn_kind=arch,
                    vocab_size=vocab,
                    embed_dim=128,
                    hidden_dim=hidden,
                    num_layers=layers,
                    bidirectional=bidirectional,
                    num_classes=self.num_classes,
                )

            return _TorchSequenceRNN(
                nn=nn,
                rnn_kind=arch,
                input_dim=input_dim,
                hidden_dim=hidden,
                num_layers=layers,
                bidirectional=bidirectional,
                num_classes=self.num_classes,
            )

        if arch == "tiny_transformer":
            layer_map = {
                "tiny_transformer_2layer": 2,
                "tiny_transformer_4layer": 4,
                "tiny_transformer_6layer": 6,
            }
            layers = layer_map.get(variant)
            if layers is None:
                raise ValueError(
                    f"Unsupported native tiny transformer variant: {variant!r}"
                )
            return _TorchTinyTransformer(
                nn=nn,
                torch=self.torch,
                vocab_size=int(ai["vocab_size"]),
                max_length=int(ai["max_text_length"]),
                embed_dim=128,
                nhead=4,
                layers=layers,
                num_classes=self.num_classes,
                token_level=self.application == "masked_language_modeling",
            )

        if arch == "bert":
            return self._build_hf_bert()

        if arch == "distilbert":
            return self._build_hf_distilbert()

        if arch == "dense_autoencoder":
            depth_map = {
                "dense_autoencoder_2layer": 2,
                "dense_autoencoder_3layer": 3,
                "dense_autoencoder_5layer": 5,
            }
            depth = depth_map.get(variant)
            if depth is None:
                raise ValueError(f"Unsupported dense autoencoder: {variant!r}")
            return _TorchDenseAutoencoder(
                nn=nn,
                input_size=input_size,
                depth=depth,
            )

        if arch == "convolutional_autoencoder":
            depth_map = {
                "convolutional_autoencoder_2layer": 2,
                "convolutional_autoencoder_4layer": 4,
                "convolutional_autoencoder_6layer": 6,
            }
            depth = depth_map.get(variant)
            if depth is None:
                raise ValueError(
                    f"Unsupported convolutional autoencoder: {variant!r}"
                )
            return _TorchConvAutoencoder(nn=nn, depth=depth)

        if arch == "variational_autoencoder":
            if variant == "vae_fc":
                return _TorchFCVAE(
                    nn=nn,
                    torch=self.torch,
                    input_size=input_size,
                    beta=1.0,
                )
            if variant in {"vae_conv", "beta_vae"}:
                return _TorchConvVAE(
                    nn=nn,
                    torch=self.torch,
                    input_size=input_size,
                    beta=4.0 if variant == "beta_vae" else 1.0,
                )
            raise ValueError(f"Unsupported VAE variant: {variant!r}")

        if arch == "feedforward_mlp":
            depth_map = {
                "mlp_2layer": 2,
                "mlp_4layer": 4,
                "mlp_8layer": 8,
            }
            depth = depth_map.get(variant)
            if depth is None:
                raise ValueError(f"Unsupported MLP variant: {variant!r}")
            if self.application == "image_classification":
                input_dim = 3 * input_size * input_size
            elif self.application == "activity_recognition":
                input_dim = int(ai["sequence_length"]) * int(ai["input_dim"])
            else:
                raise ValueError(
                    "Native MLP supports image_classification or "
                    "activity_recognition"
                )
            return _TorchMLP(
                nn=nn,
                input_dim=input_dim,
                depth=depth,
                num_classes=self.num_classes,
            )

        raise ValueError(
            f"Native PyTorch builder not implemented for "
            f"architecture={arch!r}, variant={variant!r}. "
            "Use a supported native variant or an artifact-backed runtime."
        )

    def _build_hf_bert(self):
        try:
            from transformers import (
                BertConfig,
                BertForMaskedLM,
                BertForSequenceClassification,
            )
        except ImportError as exc:
            raise RuntimeError(
                "BERT native execution requires the optional 'transformers' "
                "package. Install it with `python -m pip install transformers`."
            ) from exc

        settings = {
            "bert_tiny": (128, 2, 2, 512),
            "bert_mini": (256, 4, 4, 1024),
            "bert_small": (512, 4, 8, 2048),
            "bert_base": (768, 12, 12, 3072),
            "bert_large": (1024, 24, 16, 4096),
        }
        values = settings.get(self.variant)
        if values is None:
            raise ValueError(f"Unsupported BERT variant: {self.variant!r}")
        hidden, layers, heads, intermediate = values
        ai = self.config["ai"]
        config = BertConfig(
            vocab_size=int(ai["vocab_size"]),
            hidden_size=hidden,
            num_hidden_layers=layers,
            num_attention_heads=heads,
            intermediate_size=intermediate,
            max_position_embeddings=max(int(ai["max_text_length"]) + 2, 128),
            num_labels=self.num_classes,
        )
        if self.application == "masked_language_modeling":
            return BertForMaskedLM(config)
        return BertForSequenceClassification(config)

    def _build_hf_distilbert(self):
        if self.variant != "distilbert_base":
            raise ValueError(
                f"Unsupported DistilBERT variant: {self.variant!r}"
            )
        try:
            from transformers import (
                DistilBertConfig,
                DistilBertForMaskedLM,
                DistilBertForSequenceClassification,
            )
        except ImportError as exc:
            raise RuntimeError(
                "DistilBERT native execution requires the optional "
                "'transformers' package. Install it with "
                "`python -m pip install transformers`."
            ) from exc
        ai = self.config["ai"]
        config = DistilBertConfig(
            vocab_size=int(ai["vocab_size"]),
            dim=768,
            hidden_dim=3072,
            n_layers=6,
            n_heads=12,
            max_position_embeddings=max(int(ai["max_text_length"]) + 2, 128),
            num_labels=self.num_classes,
        )
        if self.application == "masked_language_modeling":
            return DistilBertForMaskedLM(config)
        return DistilBertForSequenceClassification(config)

    @staticmethod
    def _logits(output):
        if hasattr(output, "logits"):
            return output.logits
        if isinstance(output, (tuple, list)):
            return output[0]
        return output

    def infer(self, array: np.ndarray) -> np.ndarray:
        torch = self.torch
        if self.application in {"text_classification", "masked_language_modeling"}:
            tensor = torch.as_tensor(array, dtype=torch.long, device=self.device)
        else:
            tensor = torch.as_tensor(
                array, dtype=torch.float32, device=self.device
            )

        with torch.no_grad():
            output = self._logits(self.model(tensor))

        return output.detach().cpu().numpy()

    def train_batch(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
    ) -> Dict[str, Any]:
        torch = self.torch
        self.model.train()

        if self.application in {"text_classification", "masked_language_modeling"}:
            x = torch.as_tensor(inputs, dtype=torch.long, device=self.device)
        else:
            x = torch.as_tensor(inputs, dtype=torch.float32, device=self.device)

        self.optimizer.zero_grad(set_to_none=True)
        output = self._logits(self.model(x))

        if self.application == "masked_language_modeling":
            y = torch.as_tensor(targets, dtype=torch.long, device=self.device)
            if output.ndim != 3 or y.shape != output.shape[:2]:
                raise ValueError(
                    f"MLM output/target shape mismatch: {tuple(output.shape)} "
                    f"vs {tuple(y.shape)}"
                )
            loss = torch.nn.functional.cross_entropy(
                output.reshape(-1, output.shape[-1]),
                y.reshape(-1),
                ignore_index=-100,
            )
            predictions_all = output.argmax(dim=-1)
            mask = y != -100
            if not bool(mask.any()):
                raise ValueError("MLM training batch contains no masked targets")
            predictions = predictions_all[mask]
            evaluated_targets = y[mask]
            accuracy = float(
                (predictions == evaluated_targets).float().mean().item()
            )
        elif self.application in {
            "reconstruction",
            "anomaly_detection",
            "image_denoising",
        }:
            y = torch.as_tensor(
                targets, dtype=torch.float32, device=self.device
            )
            reconstruction_loss = torch.nn.functional.mse_loss(output, y)
            mae = torch.mean(torch.abs(output - y))
            kl_loss = getattr(self.model, "last_kl_loss", None)
            beta = float(getattr(self.model, "vae_beta", 1.0))
            if kl_loss is not None:
                loss = reconstruction_loss + beta * kl_loss
            else:
                loss = reconstruction_loss
            accuracy = None
            predictions = None
        else:
            y = torch.as_tensor(
                targets, dtype=torch.long, device=self.device
            ).reshape(-1)
            loss = torch.nn.functional.cross_entropy(output, y)
            predictions = output.argmax(dim=1)
            accuracy = float((predictions == y).float().mean().item())

        loss.backward()
        self.optimizer.step()
        self.model.eval()

        metrics: Dict[str, Any] = {
            "loss": float(loss.detach().cpu().item()),
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            "samples": int(inputs.shape[0]),
        }
        if accuracy is not None:
            metrics["accuracy"] = accuracy
            metric_targets = (
                evaluated_targets
                if self.application == "masked_language_modeling"
                else y
            )
            metrics["_targets"] = (
                metric_targets.detach().cpu().numpy().astype(np.int64, copy=False)
            )
            metrics["_predictions"] = (
                predictions.detach().cpu().numpy().astype(np.int64, copy=False)
            )
            if self.application == "masked_language_modeling":
                metrics["evaluated_tokens"] = int(metric_targets.numel())
        else:
            metrics["reconstruction_loss"] = float(
                reconstruction_loss.detach().cpu().item()
            )
            metrics["mse"] = metrics["reconstruction_loss"]
            metrics["mae"] = float(mae.detach().cpu().item())
            if kl_loss is not None:
                metrics["kl_loss"] = float(kl_loss.detach().cpu().item())
                metrics["vae_beta"] = beta
        return metrics

    def evaluation_extras(self) -> Dict[str, Any]:
        kl_loss = getattr(self.model, "last_kl_loss", None)
        if kl_loss is None:
            return {}
        return {
            "kl_loss": float(kl_loss.detach().cpu().item()),
            "vae_beta": float(getattr(self.model, "vae_beta", 1.0)),
        }

    def get_parameters(self) -> list[np.ndarray]:
        return [
            value.detach().cpu().numpy().copy()
            for value in self.model.state_dict().values()
        ]

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        state = self.model.state_dict()
        if len(parameters) != len(state):
            raise ValueError(
                f"Expected {len(state)} parameter tensors, "
                f"received {len(parameters)}"
            )

        new_state = {}
        for (name, current), value in zip(state.items(), parameters):
            tensor = self.torch.as_tensor(
                value,
                dtype=current.dtype,
                device=current.device,
            )
            if tuple(tensor.shape) != tuple(current.shape):
                raise ValueError(
                    f"Parameter {name!r} shape mismatch: "
                    f"{tuple(tensor.shape)} != {tuple(current.shape)}"
                )
            new_state[name] = tensor

        self.model.load_state_dict(new_state, strict=True)
        # FedAvg local optimization starts from the received global model.
        # Reset Adam state so momentum from an earlier round is not carried
        # into the next round.
        self.optimizer = self.torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
        )


def _rnn_variant_settings(
    architecture: str,
    variant: str,
) -> tuple[int, bool]:
    expected_prefix = "lstm" if architecture == "lstm" else "gru"

    if variant == f"{expected_prefix}_1layer":
        return 1, False
    if variant == f"{expected_prefix}_2layer":
        return 2, False
    if variant == f"bi{expected_prefix}_2layer":
        return 2, True

    raise ValueError(
        f"Variant {variant!r} is not valid for architecture "
        f"{architecture!r}"
    )


class _TorchSequenceRNN:
    def __new__(
        cls,
        nn,
        rnn_kind,
        input_dim,
        hidden_dim,
        num_layers,
        bidirectional,
        num_classes,
    ):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                rnn_cls = nn.LSTM if rnn_kind == "lstm" else nn.GRU
                self.rnn = rnn_cls(
                    input_dim,
                    hidden_dim,
                    num_layers=num_layers,
                    batch_first=True,
                    bidirectional=bidirectional,
                )
                output_dim = hidden_dim * (2 if bidirectional else 1)
                self.fc = nn.Linear(output_dim, num_classes)

            def forward(self, x):
                output, _state = self.rnn(x)
                return self.fc(output[:, -1, :])

        return Model()


class _TorchTextRNN:
    def __new__(
        cls,
        nn,
        rnn_kind,
        vocab_size,
        embed_dim,
        hidden_dim,
        num_layers,
        bidirectional,
        num_classes,
    ):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embed_dim)
                rnn_cls = nn.LSTM if rnn_kind == "lstm" else nn.GRU
                self.rnn = rnn_cls(
                    embed_dim,
                    hidden_dim,
                    num_layers=num_layers,
                    batch_first=True,
                    bidirectional=bidirectional,
                )
                output_dim = hidden_dim * (2 if bidirectional else 1)
                self.fc = nn.Linear(output_dim, num_classes)

            def forward(self, x):
                x = self.embedding(x)
                output, _state = self.rnn(x)
                return self.fc(output[:, -1, :])

        return Model()


class _TorchTinyTransformer:
    def __new__(
        cls,
        nn,
        torch,
        vocab_size,
        max_length,
        embed_dim,
        nhead,
        layers,
        num_classes,
        token_level=False,
    ):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embed_dim)
                self.position = nn.Embedding(max_length, embed_dim)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=nhead,
                    dim_feedforward=embed_dim * 4,
                    batch_first=True,
                    norm_first=True,
                )
                self.encoder = nn.TransformerEncoder(
                    encoder_layer,
                    num_layers=layers,
                )
                output_dim = vocab_size if token_level else num_classes
                self.fc = nn.Linear(embed_dim, output_dim)

            def forward(self, x):
                positions = self.position(
                    torch.arange(x.shape[1], device=x.device)
                ).unsqueeze(0)
                x = self.embedding(x) + positions
                x = self.encoder(x)
                if token_level:
                    return self.fc(x)
                return self.fc(x.mean(dim=1))

        return Model()


class _TorchDenseAutoencoder:
    def __new__(cls, nn, input_size: int, depth: int):
        flat = 3 * input_size * input_size
        if depth == 2:
            dims = [flat, 256, flat]
        elif depth == 3:
            dims = [flat, 512, 128, flat]
        elif depth == 5:
            dims = [flat, 512, 256, 64, 256, flat]
        else:
            raise ValueError(f"Unsupported dense autoencoder depth: {depth}")

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                layers = []
                for index, (left, right) in enumerate(zip(dims[:-1], dims[1:])):
                    layers.append(nn.Linear(left, right))
                    if index < len(dims) - 2:
                        layers.append(nn.ReLU())
                    else:
                        layers.append(nn.Sigmoid())
                self.network = nn.Sequential(*layers)

            def forward(self, x):
                shape = x.shape
                y = self.network(x.flatten(1))
                return y.reshape(shape)

        return Model()


class _TorchConvAutoencoder:
    def __new__(cls, nn, depth: int):
        channels = {
            2: [3, 32, 3],
            4: [3, 32, 64, 32, 3],
            6: [3, 32, 64, 128, 64, 32, 3],
        }.get(depth)
        if channels is None:
            raise ValueError(f"Unsupported conv autoencoder depth: {depth}")

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                layers = []
                for index, (left, right) in enumerate(
                    zip(channels[:-1], channels[1:])
                ):
                    layers.append(nn.Conv2d(left, right, kernel_size=3, padding=1))
                    if index < len(channels) - 2:
                        layers.append(nn.ReLU())
                    else:
                        layers.append(nn.Sigmoid())
                self.network = nn.Sequential(*layers)

            def forward(self, x):
                return self.network(x)

        return Model()


class _TorchFCVAE:
    def __new__(cls, nn, torch, input_size: int, beta: float):
        flat = 3 * input_size * input_size
        hidden = 256
        latent = 64

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc = nn.Linear(flat, hidden)
                self.mu = nn.Linear(hidden, latent)
                self.logvar = nn.Linear(hidden, latent)
                self.dec1 = nn.Linear(latent, hidden)
                self.dec2 = nn.Linear(hidden, flat)
                self.last_kl_loss = None
                self.vae_beta = beta

            def forward(self, x):
                shape = x.shape
                h = torch.relu(self.enc(x.flatten(1)))
                mu = self.mu(h)
                logvar = self.logvar(h)
                if self.training:
                    std = torch.exp(0.5 * logvar)
                    z = mu + torch.randn_like(std) * std
                else:
                    z = mu
                self.last_kl_loss = -0.5 * torch.mean(
                    1.0 + logvar - mu.pow(2) - logvar.exp()
                )
                y = torch.relu(self.dec1(z))
                y = torch.sigmoid(self.dec2(y))
                return y.reshape(shape)

        return Model()


class _TorchConvVAE:
    def __new__(cls, nn, torch, input_size: int, beta: float):
        if input_size % 4:
            raise ValueError(
                "Convolutional VAE requires ai.input_size divisible by 4"
            )
        spatial = input_size // 4
        encoded = 64 * spatial * spatial
        latent = 64

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Conv2d(3, 32, 4, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, 4, stride=2, padding=1),
                    nn.ReLU(),
                )
                self.mu = nn.Linear(encoded, latent)
                self.logvar = nn.Linear(encoded, latent)
                self.decode_fc = nn.Linear(latent, encoded)
                self.decoder = nn.Sequential(
                    nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
                    nn.ReLU(),
                    nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
                    nn.Sigmoid(),
                )
                self.last_kl_loss = None
                self.vae_beta = beta

            def forward(self, x):
                h = self.encoder(x).flatten(1)
                mu = self.mu(h)
                logvar = self.logvar(h)
                if self.training:
                    std = torch.exp(0.5 * logvar)
                    z = mu + torch.randn_like(std) * std
                else:
                    z = mu
                self.last_kl_loss = -0.5 * torch.mean(
                    1.0 + logvar - mu.pow(2) - logvar.exp()
                )
                y = self.decode_fc(z).reshape(-1, 64, spatial, spatial)
                return self.decoder(y)

        return Model()


class _TorchMLP:
    def __new__(cls, nn, input_dim: int, depth: int, num_classes: int):
        hidden = 256
        # depth counts Linear layers including the classifier.
        dims = [input_dim] + [hidden] * (depth - 1) + [num_classes]

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                layers = []
                for index, (left, right) in enumerate(zip(dims[:-1], dims[1:])):
                    layers.append(nn.Linear(left, right))
                    if index < len(dims) - 2:
                        layers.append(nn.ReLU())
                self.network = nn.Sequential(*layers)

            def forward(self, x):
                return self.network(x.flatten(1))

        return Model()
