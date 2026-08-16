"""Sentence embeddings via ONNX Runtime.

This is a drop-in replacement for the previous sentence-transformers/torch
implementation. It runs the exact same model (all-MiniLM-L6-v2), int8-quantised
to ONNX, which keeps vectors within ~0.992 cosine of the fp32 originals while
dropping the dependency tree from ~1.5 GB to ~120 MB — the difference between
fitting and not fitting in a serverless function.

The pooling below is not incidental: all-MiniLM-L6-v2 is a mean-pooled,
L2-normalised model. The raw transformer output is per-token, so reproducing
sentence-transformers' vectors means replicating both steps by hand.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

MODEL_DIR = Path(__file__).parent / "models" / "all-MiniLM-L6-v2"

# From the model's own sentence_bert_config.json. Inputs longer than this are
# truncated, matching what sentence-transformers would have done.
MAX_SEQ_LENGTH = 256


@lru_cache(maxsize=1)
def _load() -> tuple[ort.InferenceSession, Tokenizer]:
    options = ort.SessionOptions()

    # Serverless instances are single-vCPU. Letting ONNX spin up its default
    # thread pool costs more in contention than it wins in parallelism.
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1

    session = ort.InferenceSession(
        str(MODEL_DIR / "model.onnx"),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )

    tokenizer = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    tokenizer.enable_truncation(max_length=MAX_SEQ_LENGTH)

    return session, tokenizer


@lru_cache(maxsize=512)
def embed_text(text: str) -> np.ndarray:
    """Embed a single string into a 384-dim L2-normalised vector.

    Cached because callers re-embed the same strings constantly — every request
    embeds all four static table descriptions in filter_relevant_tables, and
    those never change within an instance's lifetime.

    The returned array is shared across cache hits. Callers must treat it as
    read-only; it is flagged non-writeable to make that a loud failure rather
    than a silent corruption of the cache.
    """
    session, tokenizer = _load()
    encoding = tokenizer.encode(text)

    attention_mask = np.array([encoding.attention_mask], dtype=np.int64)
    inputs = {
        "input_ids": np.array([encoding.ids], dtype=np.int64),
        "attention_mask": attention_mask,
        "token_type_ids": np.array([encoding.type_ids], dtype=np.int64),
    }

    token_embeddings = session.run(None, inputs)[0]  # (1, seq_len, 384)

    # Mean pooling over real tokens only — padding must not drag the mean.
    mask = attention_mask[..., None].astype(np.float32)
    summed = (token_embeddings * mask).sum(axis=1)
    pooled = summed / np.clip(mask.sum(axis=1), 1e-9, None)

    normalized = pooled / np.linalg.norm(pooled, axis=1, keepdims=True)

    vector = normalized[0]
    vector.flags.writeable = False
    return vector


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
