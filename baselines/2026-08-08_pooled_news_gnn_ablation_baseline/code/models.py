"""Shared sequence models for pooled P1-P3 experiments."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn.functional import softplus
from torch.nn.utils.rnn import pack_padded_sequence


# Volatility is strictly positive; the standardized linear head does not structurally
# guarantee it.  The graph message-passing widens the normalized-prediction variance
# enough that its lower tail crosses each ticker's ``raw = z*std + mean = 0`` boundary,
# denormalizing to nonpositive volatility.  A denormalized-scale floor at this epsilon
# (three orders of magnitude below the ~1e-3 typical Parkinson scale) restores positivity
# without reshaping the bulk of the distribution.
POSITIVITY_EPSILON = 1e-6


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

    def forward(
        self, node_features: torch.Tensor, adjacency: torch.Tensor,
        presence_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if adjacency.ndim == 2:
            adjacency = adjacency.unsqueeze(0).expand(node_features.shape[0], -1, -1)
        if adjacency.ndim != 3 or adjacency.shape[:2] != node_features.shape[:2]:
            raise ValueError("adjacency must be [nodes, nodes] or [batch, nodes, nodes]")
        present: torch.Tensor | None = None
        if presence_mask is not None:
            if presence_mask.shape != node_features.shape[:2]:
                raise ValueError("presence_mask must be [batch, nodes]")
            present = presence_mask.to(dtype=torch.bool)
            # Absent nodes contribute nothing and are never attended to (zero their
            # features and any incoming edge), so present outputs cannot depend on them.
            node_features = node_features * present.unsqueeze(-1)
            adjacency = adjacency * present.unsqueeze(1).to(adjacency.dtype)
        topology = adjacency != 0
        needs_neighbor = topology.any(dim=-1)
        if present is not None:
            needs_neighbor = needs_neighbor | ~present
        if not needs_neighbor.all():
            raise ValueError("each present graph node requires a self-loop or neighbor")
        weights = torch.softmax(
            adjacency.to(node_features.dtype).masked_fill(~topology, float("-inf")), dim=-1,
        )
        if present is not None:
            # Absent rows have an all -inf logit row (NaN after softmax); zero them so
            # absent nodes emit nothing and the NaN never reaches the aggregation.
            weights = weights.masked_fill(~present.unsqueeze(-1), 0.0)
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
        num_tickers = self.gate_logits.numel()
        self.register_buffer("target_mean", torch.zeros(num_tickers))
        self.register_buffer("target_std", torch.ones(num_tickers))
        self._positivity_configured = False
        self._positivity_epsilon = POSITIVITY_EPSILON

    def configure_positivity(self, store: Any, epsilon: float = POSITIVITY_EPSILON) -> "GraphAblationModel":
        """Install the per-ticker target scaling used to floor denormalized predictions.

        Enforcing positivity requires each ticker's train-fitted target mean/std (the same
        values ``PreprocessorStore.inverse_targets`` uses), so no new statistic is derived
        and the leakage/scaler contract is unchanged.
        """

        if epsilon <= 0:
            raise ValueError("positivity epsilon must be positive")
        mean = self.target_mean.clone()
        std = self.target_std.clone()
        for ticker_id in range(mean.numel()):
            preprocessor = store.preprocessors.get(ticker_id)
            if preprocessor is None:
                raise ValueError(f"positivity config missing preprocessor for ticker_id {ticker_id}")
            scaler = preprocessor.target_scaler
            mean[ticker_id] = float(scaler.mean[0])
            std[ticker_id] = float(scaler.std[0])
        self.target_mean.copy_(mean)
        self.target_std.copy_(std)
        self._positivity_epsilon = float(epsilon)
        self._positivity_configured = True
        return self

    def _apply_positivity(self, output: torch.Tensor, ticker_ids: torch.Tensor) -> torch.Tensor:
        """Map normalized predictions through a denormalized-scale positive floor.

        ``raw = z*std + mean`` is passed through ``epsilon*softplus(raw/epsilon) + epsilon``:
        an identity for ``raw >> epsilon`` (the bulk, so the spread is not collapsed) that
        smoothly floors the sub-epsilon tail to a strictly positive value, then renormalized
        so the model still emits normalized predictions for the unchanged evaluation path.
        """

        mean = self.target_mean[ticker_ids].reshape(output.shape)
        std = self.target_std[ticker_ids].reshape(output.shape)
        raw = output * std + mean
        epsilon = self._positivity_epsilon
        raw_positive = epsilon * softplus(raw / epsilon) + epsilon
        return (raw_positive - mean) / std

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

    def encode_base(
        self, x_price: torch.Tensor, x_news: torch.Tensor, news_mask: torch.Tensor,
        ticker_ids: torch.Tensor, presence_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Frozen-encoder node embeddings ``cat(price_hidden, gated_news)`` — the cacheable
        part of ``forward``.

        The encoders are frozen (``requires_grad_(False)``) and dropout-free, so this output is a
        deterministic function of the inputs alone: identical every epoch and identical between
        G0 and G1.  It can therefore be computed once per seed and reused, instead of recomputed
        on every forward pass.  With ``presence_mask`` only PRESENT nodes are run through the
        LSTMs (each node's sequence is encoded independently, so present rows are bit-identical to
        the full-batch encode); absent nodes get a zero embedding and never influence present
        outputs (message passing zeroes their features and incoming edges).

        Returns ``[batch, nodes, hidden]`` when ``x_price`` is 4-D, else ``[nodes, hidden]``.
        """

        batched = x_price.ndim == 4
        if batched:
            if x_news.ndim != 4 or news_mask.ndim != 3 or ticker_ids.ndim != 2:
                raise ValueError("batched graph inputs must be [batch, nodes, time, features]")
            batch_size, node_count = x_price.shape[:2]
            if presence_mask is not None and presence_mask.shape != (batch_size, node_count):
                raise ValueError("batched presence_mask must be [batch, nodes]")
            flat_price = x_price.reshape(batch_size * node_count, *x_price.shape[2:])
            flat_news = x_news.reshape(batch_size * node_count, *x_news.shape[2:])
            flat_mask = news_mask.reshape(batch_size * node_count, news_mask.shape[-1])
            flat_ticker = ticker_ids.reshape(batch_size * node_count)
            flat_presence = None if presence_mask is None else presence_mask.reshape(batch_size * node_count)
        else:
            if presence_mask is not None and (presence_mask.ndim != 1
                                              or presence_mask.shape[0] != x_price.shape[0]):
                raise ValueError("presence_mask must be a 1-D vector over nodes")
            flat_price, flat_news, flat_mask = x_price, x_news, news_mask
            flat_ticker, flat_presence = ticker_ids, presence_mask
        flat_ticker = self._news_encoder._validated_ticker_ids(flat_ticker, flat_price.shape[0])
        with torch.no_grad():
            base = self._encode_nodes(flat_price, flat_news, flat_mask, flat_ticker, flat_presence)
        if batched:
            base = base.reshape(batch_size, node_count, -1)
        return base

    def _encode_nodes(
        self, price: torch.Tensor, news: torch.Tensor, mask: torch.Tensor,
        ticker_ids: torch.Tensor, presence: torch.Tensor | None,
    ) -> torch.Tensor:
        if presence is None:
            return self._encode_present_rows(price, news, mask, ticker_ids)
        present = presence.to(dtype=torch.bool)
        base = price.new_zeros((price.shape[0], self.head[0].in_features))
        index = present.nonzero(as_tuple=False).squeeze(-1)
        if index.numel() == 0:
            return base
        base[index] = self._encode_present_rows(price[index], news[index], mask[index], ticker_ids[index])
        return base

    def _encode_present_rows(
        self, price: torch.Tensor, news: torch.Tensor, mask: torch.Tensor, ticker_ids: torch.Tensor,
    ) -> torch.Tensor:
        _, (price_hidden, _) = self.price_encoder(price)
        news_hidden = self._news_encoder._encode_news(news, mask)
        gated_news = torch.sigmoid(self.gate_logits[ticker_ids])[:, None] * news_hidden
        return torch.cat((price_hidden[-1], gated_news), dim=1)

    def apply_graph_head(
        self, base: torch.Tensor, adjacency: torch.Tensor, ticker_ids: torch.Tensor,
        presence_mask: torch.Tensor | None = None, apply_message_passing: bool = True,
    ) -> torch.Tensor:
        """Trainable message-passing (G1 only) + head + positivity, given a precomputed ``base``.

        ``base`` is ``[batch, nodes, hidden]`` (batched) or ``[nodes, hidden]`` (single) as
        returned by :meth:`encode_base`.  This is the only part that carries gradients and reads
        the trainable message-passing / head parameters.

        ``apply_message_passing=False`` reads out the same trained model with the graph residual
        removed -- the pure backbone+head+positivity (P3) pathway.  Because ``head`` and the frozen
        encoders are shared, this is bit-identical to a ``use_gnn=False`` model built from the same
        weights, so 'G1 minus the GAT = P3' holds exactly (see the nesting test).
        """

        batched = base.ndim == 3
        if self.message_passing is not None and apply_message_passing:
            if batched:
                base = base + self.message_passing(base, adjacency, presence_mask)
            else:
                batch_presence = None if presence_mask is None else presence_mask.unsqueeze(0)
                base = base + self.message_passing(base.unsqueeze(0), adjacency, batch_presence).squeeze(0)
        output = self.head(base).squeeze(-1)
        if self._positivity_configured:
            flat_ticker = ticker_ids.reshape(-1).to(torch.long) if batched else ticker_ids.to(torch.long)
            output = self._apply_positivity(output, flat_ticker)
        return output

    def forward(
        self, x_price: torch.Tensor, x_news: torch.Tensor, news_mask: torch.Tensor,
        ticker_ids: torch.Tensor, adjacency: torch.Tensor,
        presence_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        base = self.encode_base(x_price, x_news, news_mask, ticker_ids, presence_mask)
        return self.apply_graph_head(base, adjacency, ticker_ids, presence_mask)
