"""Steps 3-4: preprocessing fitted on train only, applied causally everywhere.

Columns: 0 = X, 1 = Y, 2.. = optional reference channels (shared-noise proxies).
"""
import numpy as np
from scipy.signal import butter, sosfilt


class CausalPreprocessor:
    def __init__(self, band=None, fs=1.0, order=4, ref_cols=(), filter_warmup=100):
        self.band = band
        self.fs = fs
        self.order = order
        self.ref_cols = list(ref_cols)
        self.filter_warmup = filter_warmup if band is not None else 0
        self.sos = None
        if band is not None:
            lo, hi = band
            if lo and hi:
                self.sos = butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
            elif hi:
                self.sos = butter(order, hi, btype="lowpass", fs=fs, output="sos")
            else:
                self.sos = butter(order, lo, btype="highpass", fs=fs, output="sos")
        self._fitted = False

    def _filter(self, block):
        if self.sos is None:
            return block
        # sosfilt is forward-only: output at t depends on inputs <= t.
        # Each block starts from zero state, so the transient is dropped.
        return sosfilt(self.sos, block, axis=0)[self.filter_warmup :]

    def _split_cols(self, block):
        main = block[:, :2]
        ref = block[:, self.ref_cols] if self.ref_cols else None
        return main, ref

    def fit(self, train_blocks):
        filtered = np.concatenate([self._filter(b) for b in train_blocks])
        main, ref = self._split_cols(filtered)
        self.W = None
        if ref is not None:
            # Common-mode regression weights, estimated on train only.
            design = np.column_stack([ref, np.ones(len(ref))])
            self.W, *_ = np.linalg.lstsq(design, main, rcond=None)
            main = main - design @ self.W
        self.mean = main.mean(axis=0)
        self.std = main.std(axis=0)
        self._fitted = True
        return self

    def transform(self, blocks):
        if not self._fitted:
            raise RuntimeError("fit() on train blocks before transform()")
        out = []
        for b in blocks:
            main, ref = self._split_cols(self._filter(b))
            if self.W is not None:
                main = main - np.column_stack([ref, np.ones(len(ref))]) @ self.W
            out.append((main - self.mean) / self.std)
        return out
