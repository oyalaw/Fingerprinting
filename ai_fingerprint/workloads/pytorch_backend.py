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
        torch = self.torch
        nn = self.nn
        arch = self.architecture
        variant = self.variant

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
            return builder(
                weights=None,
                num_classes=self.num_classes,
            )

        if arch in {"lstm", "gru"}:
            input_dim = int(self.config["ai"]["input_dim"])
            hidden = 128
            layers, bidirectional = _rnn_variant_settings(
                architecture=arch,
                variant=variant,
            )

            if self.application == "text_classification":
                vocab = int(self.config["ai"]["vocab_size"])
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
            if variant != "tiny_transformer_2layer":
                raise ValueError(
                    f"Unsupported native tiny transformer variant: "
                    f"{variant!r}"
                )
            vocab = int(self.config["ai"]["vocab_size"])
            return _TorchTinyTransformer(
                nn=nn,
                vocab_size=vocab,
                embed_dim=128,
                nhead=4,
                layers=2,
                num_classes=self.num_classes,
            )

        if arch == "convolutional_autoencoder":
            if variant != "convolutional_autoencoder_4layer":
                raise ValueError(
                    f"Unsupported native autoencoder variant: "
                    f"{variant!r}"
                )
            return _TorchConvAutoencoder(nn=nn)

        raise ValueError(
            f"Native PyTorch builder not implemented for "
            f"architecture={arch!r}, variant={variant!r}. "
            "Use a supported native variant or an artifact backed runtime."
        )

    def infer(self, array: np.ndarray) -> np.ndarray:
        torch = self.torch
        if self.application == "text_classification":
            tensor = torch.as_tensor(
                array,
                dtype=torch.long,
                device=self.device,
            )
        else:
            tensor = torch.as_tensor(
                array,
                dtype=torch.float32,
                device=self.device,
            )

        with torch.no_grad():
            output = self.model(tensor)

        return output.detach().cpu().numpy()

    def train_batch(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
    ) -> Dict[str, float]:
        torch = self.torch
        self.model.train()

        if self.application == "text_classification":
            x = torch.as_tensor(
                inputs,
                dtype=torch.long,
                device=self.device,
            )
        else:
            x = torch.as_tensor(
                inputs,
                dtype=torch.float32,
                device=self.device,
            )

        self.optimizer.zero_grad(set_to_none=True)
        output = self.model(x)

        if self.application in {
            "reconstruction",
            "anomaly_detection",
            "image_denoising",
        }:
            y = torch.as_tensor(
                targets,
                dtype=torch.float32,
                device=self.device,
            )
            loss = torch.nn.functional.mse_loss(output, y)
            accuracy = None
        else:
            y = torch.as_tensor(
                targets,
                dtype=torch.long,
                device=self.device,
            ).reshape(-1)
            loss = torch.nn.functional.cross_entropy(output, y)
            predictions = output.argmax(dim=1)
            accuracy = float(
                (predictions == y).float().mean().item()
            )

        loss.backward()
        self.optimizer.step()
        self.model.eval()

        metrics: Dict[str, float] = {
            "loss": float(loss.detach().cpu().item()),
            "learning_rate": float(
                self.optimizer.param_groups[0]["lr"]
            ),
        }
        if accuracy is not None:
            metrics["accuracy"] = accuracy
        return metrics

    def get_parameters(self) -> list[np.ndarray]:
        return [
            value.detach().cpu().numpy().copy()
            for value in self.model.state_dict().values()
        ]

    def set_parameters(
        self,
        parameters: list[np.ndarray],
    ) -> None:
        state = self.model.state_dict()
        if len(parameters) != len(state):
            raise ValueError(
                f"Expected {len(state)} parameter tensors, "
                f"received {len(parameters)}"
            )

        new_state = {}
        for (name, current), value in zip(
            state.items(),
            parameters,
        ):
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
                output, state = self.rnn(x)
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
                self.embedding = nn.Embedding(
                    vocab_size,
                    embed_dim,
                )
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
                output, state = self.rnn(x)
                return self.fc(output[:, -1, :])

        return Model()


class _TorchTinyTransformer:
    def __new__(
        cls,
        nn,
        vocab_size,
        embed_dim,
        nhead,
        layers,
        num_classes,
    ):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(
                    vocab_size,
                    embed_dim,
                )
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=nhead,
                    batch_first=True,
                )
                self.encoder = nn.TransformerEncoder(
                    encoder_layer,
                    num_layers=layers,
                )
                self.fc = nn.Linear(embed_dim, num_classes)

            def forward(self, x):
                x = self.embedding(x)
                x = self.encoder(x)
                x = x.mean(dim=1)
                return self.fc(x)

        return Model()


class _TorchConvAutoencoder:
    def __new__(cls, nn):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(64, 32, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 3, kernel_size=3, padding=1),
                    nn.Sigmoid(),
                )

            def forward(self, x):
                return self.network(x)

        return Model()
