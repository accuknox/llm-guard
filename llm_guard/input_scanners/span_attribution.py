from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from llm_guard.util import get_logger, lazy_load_dep

LOGGER = get_logger()


@dataclass
class Span:
    """A character span in the original prompt that drives the target class."""

    text: str
    score: float
    start: int
    end: int


class SpanDetector:
    """Gradient-based (Integrated Gradients) span attribution for a HuggingFace
    sequence-classification model.

    Given a prompt, it locates the character spans that most drive the target
    (violation) class. It chunks long prompts with the model's own tokenizer so
    inputs beyond the 512-token limit are handled, classifies each chunk, and
    runs Integrated Gradients only on the chunk(s) that trigger.

    Requires a PyTorch model (needs gradients). ONNX models are not supported and
    produce an empty result with a warning.
    """

    def __init__(
        self,
        model,
        tokenizer,
        target_class: int,
        *,
        device=None,
        chunk_size: int = 510,
        chunk_overlap: int = 50,
        attribution_threshold: float = 0.3,
        classify_threshold: float = 0.5,
        n_steps: int = 50,
        embedding_layer=None,
    ):
        self._torch = lazy_load_dep("torch")
        self._model = model
        self._tokenizer = tokenizer
        self._target_class = target_class
        self._device = device
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._attribution_threshold = attribution_threshold
        self._classify_threshold = classify_threshold
        self._n_steps = n_steps

        self._embedding_layer = embedding_layer
        # captum import + Integrated Gradients setup are deferred to the first
        # detect() call, so an ONNX model can be skipped gracefully (see detect)
        # rather than crashing at construction.
        self._lig = None

    # ── internals ──

    def _ensure_ready(self) -> None:
        if self._lig is not None:
            return
        if self._device is None:
            self._device = next(self._model.parameters()).device
        captum_attr = lazy_load_dep("captum.attr", "captum")
        layer = self._embedding_layer or self._find_embedding_layer()
        self._lig = captum_attr.LayerIntegratedGradients(self._forward, layer)

    def _find_embedding_layer(self):
        base = getattr(self._model, "base_model", self._model)
        embeddings = getattr(base, "embeddings", None)
        if embeddings is not None and hasattr(embeddings, "word_embeddings"):
            return embeddings.word_embeddings
        raise AttributeError(
            f"Cannot auto-detect the embedding layer for {type(self._model).__name__}. "
            "Pass it explicitly via the embedding_layer parameter."
        )

    def _forward(self, input_ids, attention_mask):
        return self._model(input_ids=input_ids, attention_mask=attention_mask).logits

    def _wrap(self, chunk_ids: list[int]) -> list[int]:
        """Add the model's [CLS]/[SEP] (or <s>/</s>) special tokens."""
        return [self._tokenizer.cls_token_id, *chunk_ids, self._tokenizer.sep_token_id]

    def _chunk(self, token_ids: list[int], offsets: list[tuple[int, int]]):
        stride = self._chunk_size - self._chunk_overlap
        n = len(token_ids)
        for start in range(0, n, stride):
            end = min(start + self._chunk_size, n)
            yield token_ids[start:end], offsets[start:end]
            if end >= n:
                break

    def _classify(self, chunk_ids: list[int]) -> float:
        torch = self._torch
        input_ids = torch.tensor([self._wrap(chunk_ids)], device=self._device)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            logits = self._forward(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1)
        return probs[0, self._target_class].item()

    def _attribute(
        self, chunk_ids: list[int], chunk_offsets: list[tuple[int, int]], prompt: str
    ) -> list[Span]:
        torch = self._torch
        input_ids = torch.tensor([self._wrap(chunk_ids)], device=self._device)
        attention_mask = torch.ones_like(input_ids)

        pad_id = self._tokenizer.pad_token_id or 0
        baseline = torch.full_like(input_ids, pad_id)
        baseline[0, 0] = self._tokenizer.cls_token_id
        baseline[0, -1] = self._tokenizer.sep_token_id

        attributions = self._lig.attribute(
            input_ids,
            baselines=baseline,
            target=self._target_class,
            additional_forward_args=(attention_mask,),
            n_steps=self._n_steps,
        )

        # Drop [CLS]/[SEP], L2-norm across the embedding dim -> one score per token.
        scores = attributions.squeeze(0)[1:-1].norm(dim=-1)
        max_val = scores.max()
        if max_val > 0:
            scores = scores / max_val

        return self._extract_spans(scores.cpu().tolist(), chunk_offsets, prompt)

    def _extract_spans(
        self, scores: list[float], offsets: list[tuple[int, int]], prompt: str
    ) -> list[Span]:
        high = [i for i, s in enumerate(scores) if s >= self._attribution_threshold]
        if not high:
            return []

        # Group contiguous high-attribution tokens (merge gaps of <= 1 token).
        groups: list[list[int]] = [[high[0]]]
        for idx in high[1:]:
            if idx - groups[-1][-1] <= 2:
                groups[-1].append(idx)
            else:
                groups.append([idx])

        spans = []
        for group in groups:
            start = offsets[group[0]][0]
            end = offsets[group[-1]][1]
            avg = sum(scores[i] for i in group) / len(group)
            spans.append(Span(text=prompt[start:end], score=round(avg, 4), start=start, end=end))
        return spans

    def _merge_spans(self, spans: list[Span], prompt: str) -> list[Span]:
        """Merge spans that overlap/touch in character space (dedupes the chunk
        overlap region), keeping the max score."""
        if not spans:
            return []
        spans = sorted(spans, key=lambda s: s.start)
        merged = [spans[0]]
        for span in spans[1:]:
            last = merged[-1]
            if span.start <= last.end:
                end = max(last.end, span.end)
                merged[-1] = Span(
                    text=prompt[last.start : end],
                    score=max(last.score, span.score),
                    start=last.start,
                    end=end,
                )
            else:
                merged.append(span)
        return merged

    # ── public api ──

    def detect(self, prompt: str) -> list[Span]:
        torch = self._torch
        if not isinstance(self._model, torch.nn.Module):
            LOGGER.warning(
                "Span attribution requires a PyTorch model (needs gradients); "
                "got a non-torch model such as ONNX. Skipping span detection."
            )
            return []

        self._ensure_ready()

        encoded = self._tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
        token_ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]
        if not token_ids:
            return []

        chunks = list(self._chunk(token_ids, offsets))
        chunk_scores = [self._classify(ids) for ids, _ in chunks]

        # Attribute chunks over the threshold; if none, fall back to the single
        # highest-scoring chunk so a flagged prompt always yields a span.
        flagged = [i for i, s in enumerate(chunk_scores) if s >= self._classify_threshold]
        if not flagged:
            flagged = [max(range(len(chunk_scores)), key=lambda i: chunk_scores[i])]

        spans: list[Span] = []
        for i in flagged:
            chunk_ids, chunk_offsets = chunks[i]
            spans.extend(self._attribute(chunk_ids, chunk_offsets, prompt))

        return self._merge_spans(spans, prompt)

    def detect_as_dicts(self, prompt: str) -> list[dict]:
        """Convenience wrapper returning JSON-serializable dicts for API use."""
        return [dataclasses.asdict(span) for span in self.detect(prompt)]
