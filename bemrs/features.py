"""Configured feature projection without experiment monkey patches."""
import numpy as np

class FeatureSubset:
    def __init__(self, encoder, names, group):
        self.encoder = encoder
        self.indices = [i for i, name in enumerate(names) if name.endswith('_mean')]
        if group != 'mean' or not self.indices:
            raise ValueError(f'Unsupported feature group: {group}')
        self.output_dim = len(self.indices)
        self.extractor_version = f'bemrs_{group}{self.output_dim}'
        self.extractor_description = 'Mean behavior descriptors across fixed probes and instances'

    def __getattr__(self, name):
        return getattr(self.encoder, name)

    def encode_single(self, code):
        return np.asarray(self.encoder.encode_single(code), dtype=np.float32)[self.indices]

    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            return self.encode_single(texts)
        return np.asarray([self.encode_single(t) for t in texts], dtype=np.float32)

    def _fallback_feature(self):
        return np.full(self.output_dim, -1.0, dtype=np.float32)
