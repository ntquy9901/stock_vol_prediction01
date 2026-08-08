"""Shared sequence models for pooled P1-P3 experiments."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class PooledPriceLSTM(nn.Module):
    """P1: one price encoder and prediction head shared by pooled samples."""

    def __init__(self, price_dim: int, hidden_dim: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        self.price_lstm = nn.LSTM(price_dim, hidden_dim, num_layers=2, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x_price: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.price_lstm(x_price)
        return self.head(hidden[-1]).squeeze(-1)


class PooledPriceNewsLSTM(nn.Module):
    """P2/P3: shared price/news encoders with an optional ticker-indexed news gate."""

    def __init__(
        self,
        price_dim: int,
        news_dim: int,
        num_tickers: int,
        use_gate: bool,
        hidden_dim: int = 64,
        news_hidden_dim: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.use_gate = use_gate
        self.price_lstm = nn.LSTM(price_dim, hidden_dim, num_layers=2, batch_first=True, dropout=dropout)
        self.news_lstm = nn.LSTM(news_dim, news_hidden_dim, num_layers=2, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + news_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        if use_gate:
            self.gate_logits = nn.Parameter(torch.zeros(num_tickers))
        else:
            self.register_parameter("gate_logits", None)

    def _encode_news(self, x_news: torch.Tensor, news_mask: torch.Tensor) -> torch.Tensor:
        if news_mask.ndim != 2 or news_mask.shape != x_news.shape[:2]:
            raise ValueError("news_mask must have shape [batch, sequence]")
        if news_mask.device != x_news.device:
            raise ValueError("news_mask must be on the x_news device")
        news_mask = news_mask.to(dtype=torch.bool)
        representations = x_news.new_zeros((x_news.shape[0], self.news_lstm.hidden_size))
        valid_indices = news_mask.any(dim=1).nonzero(as_tuple=False).squeeze(-1)
        if valid_indices.numel() == 0:
            return representations

        sequences = [x_news[index, news_mask[index]] for index in valid_indices]
        lengths = torch.tensor([sequence.shape[0] for sequence in sequences], device=x_news.device)
        padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        packed = pack_padded_sequence(padded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.news_lstm(packed)
        representations[valid_indices] = hidden[-1]
        return representations

    def _validated_ticker_ids(self, ticker_ids: torch.Tensor, batch_size: int) -> torch.Tensor:
        integer_dtypes = {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint16,
            torch.uint32,
            torch.uint64,
        }
        if ticker_ids.ndim != 1 or ticker_ids.shape[0] != batch_size:
            raise ValueError("ticker_ids must be a 1-D tensor matching the batch size")
        if ticker_ids.dtype not in integer_dtypes:
            raise ValueError("ticker_ids must use a non-boolean integer dtype")
        if ticker_ids.device != self.gate_logits.device:
            raise ValueError("ticker_ids must be on the gate parameter device")
        ticker_ids = ticker_ids.to(dtype=torch.long)
        if (ticker_ids < 0).any() or (ticker_ids >= self.gate_logits.numel()).any():
            raise ValueError("ticker_ids must be within the configured ticker range")
        return ticker_ids

    def forward(
        self,
        x_price: torch.Tensor,
        x_news: torch.Tensor,
        news_mask: torch.Tensor,
        ticker_ids: torch.Tensor,
    ) -> torch.Tensor:
        _, (price_hidden, _) = self.price_lstm(x_price)
        news_hidden = self._encode_news(x_news, news_mask)
        if self.use_gate:
            ticker_ids = self._validated_ticker_ids(ticker_ids, x_price.shape[0])
            news_hidden = torch.sigmoid(self.gate_logits[ticker_ids])[:, None] * news_hidden
        return self.head(torch.cat((price_hidden[-1], news_hidden), dim=1)).squeeze(-1)


class _ResidualMessagePassing(nn.Module):
    """Small native-PyTorch GAT-style aggregation used by the research-only ablation."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        if adjacency.ndim == 2:
            adjacency = adjacency.unsqueeze(0).expand(node_features.shape[0], -1, -1)
        if adjacency.ndim != 3 or adjacency.shape[:2] != node_features.shape[:2]:
            raise ValueError("adjacency must be [nodes, nodes] or [batch, nodes, nodes]")
        topology = adjacency != 0
        if not topology.any(dim=-1).all():
            raise ValueError("each graph node requires a self-loop or neighbor")
        weights = torch.softmax(
            adjacency.to(node_features.dtype).masked_fill(~topology, float("-inf")), dim=-1,
        )
        return self.projection(torch.bmm(weights, node_features))


class GraphAblationModel(nn.Module):
    """G0/G1 with a graph-safe P3 initialization and frozen pretrained encoders."""

    def __init__(self, p3: PooledPriceNewsLSTM, use_gnn: bool) -> None:
        super().__init__()
        self.use_gnn = use_gnn
        self.price_encoder = p3.price_lstm
        self.news_encoder = p3.news_lstm
        self.gate_logits = p3.gate_logits
        self.head = p3.head
        self._news_encoder = p3
        self.message_passing = _ResidualMessagePassing(p3.head[0].in_features) if use_gnn else None
        for component in (self.price_encoder, self.news_encoder):
            component.eval()
            for parameter in component.parameters():
                parameter.requires_grad_(False)
        self.gate_logits.requires_grad_(False)

    @classmethod
    def from_p3_checkpoint(
        cls, path: str, use_gnn: bool, graph_train_end_date: str | None = None,
        graph_manifest_hash: str | None = None,
    ) -> "GraphAblationModel":
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if not checkpoint.get("graph_safe") or not checkpoint.get("training_sample_hash"):
            raise ValueError("P3 checkpoint is not graph-safe")
        checkpoint_boundary = checkpoint.get("graph_train_end_date")
        max_training_date = checkpoint.get("max_training_target_date")
        if not checkpoint_boundary or not max_training_date or max_training_date > checkpoint_boundary:
            raise ValueError("graph-safe P3 checkpoint has invalid training provenance")
        if graph_train_end_date is not None and checkpoint_boundary != graph_train_end_date:
            raise ValueError("graph-safe P3 checkpoint graph train boundary differs")
        if graph_manifest_hash is not None and checkpoint.get("graph_manifest_hash") != graph_manifest_hash:
            raise ValueError("graph-safe P3 checkpoint graph manifest hash differs")
        state = checkpoint.get("model_state")
        if not isinstance(state, dict):
            raise ValueError("P3 checkpoint has no model_state")
        try:
            price_dim = int(state["price_lstm.weight_ih_l0"].shape[1])
            hidden_dim = int(state["price_lstm.weight_hh_l0"].shape[1])
            news_dim = int(state["news_lstm.weight_ih_l0"].shape[1])
            news_hidden_dim = int(state["news_lstm.weight_hh_l0"].shape[1])
            num_tickers = int(state["gate_logits"].numel())
        except (KeyError, IndexError, AttributeError) as error:
            raise ValueError("P3 checkpoint does not contain a compatible gated model") from error
        p3 = PooledPriceNewsLSTM(price_dim, news_dim, num_tickers, use_gate=True,
                                 hidden_dim=hidden_dim, news_hidden_dim=news_hidden_dim, dropout=0.0)
        p3.load_state_dict(state, strict=True)
        model = cls(p3, use_gnn)
        model.graph_train_end_date = checkpoint_boundary
        model.graph_manifest_hash = checkpoint.get("graph_manifest_hash")
        return model

    def train(self, mode: bool = True) -> "GraphAblationModel":
        super().train(mode)
        self.price_encoder.eval()
        self.news_encoder.eval()
        return self

    def forward(
        self, x_price: torch.Tensor, x_news: torch.Tensor, news_mask: torch.Tensor,
        ticker_ids: torch.Tensor, adjacency: torch.Tensor,
    ) -> torch.Tensor:
        batched = x_price.ndim == 4
        if batched:
            if x_news.ndim != 4 or news_mask.ndim != 3 or ticker_ids.ndim != 2:
                raise ValueError("batched graph inputs must be [batch, nodes, time, features]")
            batch_size, node_count = x_price.shape[:2]
            x_price = x_price.reshape(batch_size * node_count, *x_price.shape[2:])
            x_news = x_news.reshape(batch_size * node_count, *x_news.shape[2:])
            news_mask = news_mask.reshape(batch_size * node_count, news_mask.shape[-1])
            ticker_ids = ticker_ids.reshape(batch_size * node_count)
            if adjacency.ndim == 2:
                adjacency = adjacency.unsqueeze(0).expand(batch_size, -1, -1)
            if adjacency.ndim != 3 or adjacency.shape != (batch_size, node_count, node_count):
                raise ValueError("batched adjacency must be [batch, nodes, nodes]")
        with torch.no_grad():
            _, (price_hidden, _) = self.price_encoder(x_price)
            news_hidden = self._news_encoder._encode_news(x_news, news_mask)
            ticker_ids = self._news_encoder._validated_ticker_ids(ticker_ids, x_price.shape[0])
            gated_news = torch.sigmoid(self.gate_logits[ticker_ids])[:, None] * news_hidden
            base = torch.cat((price_hidden[-1], gated_news), dim=1)
        if self.message_passing is not None:
            if batched:
                base = base.reshape(batch_size, node_count, -1)
                base = base + self.message_passing(base, adjacency)
            else:
                base = base + self.message_passing(base.unsqueeze(0), adjacency).squeeze(0)
        output = self.head(base).squeeze(-1)
        return output.reshape(batch_size, node_count) if batched else output
