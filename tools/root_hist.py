"""A small uniform-binning 1D histogram, with ROOT-like read semantics.

Converted ROOT macros think in TH1 terms — which bin a value falls in,
inclusive `Integral(i, j)` sums of bin *contents* (not densities), merging
groups of bins — so those operations are provided here directly instead of
being re-expressed with numpy histogram helpers at each call site.

Contents are stored ROOT-style in an array of nbins + 2 slots: index 0 is
underflow, 1..nbins the real bins, nbins + 1 overflow, so that values pushed
outside the axis (by a convolution, say) are not silently folded back in.
"""

from __future__ import annotations

import numpy as np


class Hist1D:
    """Fixed-width 1D histogram with ROOT TH1 lookup and integration rules."""

    def __init__(self, nbins: int, xmin: float, xmax: float,
                 name: str = "", title: str = "") -> None:
        self.nbins = int(nbins)
        self.xmin = float(xmin)
        self.xmax = float(xmax)
        self.name = name
        self.title = title
        self.contents = np.zeros(self.nbins + 2, dtype=np.float64)
        self.entries = 0.0

    # --- construction --------------------------------------------------------

    @classmethod
    def from_uproot(cls, obj, name: str = "") -> "Hist1D":
        """Build from an uproot TH1. Keeps GetEntries() and under/overflow."""
        axis = obj.axis()
        edges = axis.edges()
        nbins = len(edges) - 1
        hist = cls(nbins, float(edges[0]), float(edges[-1]),
                   name=name or obj.name, title=obj.title)
        # uproot's values(flow=True) is [underflow, bins..., overflow].
        hist.contents = np.asarray(obj.values(flow=True), dtype=np.float64).copy()
        hist.entries = float(obj.member("fEntries"))
        return hist

    def clone(self, name: str = "") -> "Hist1D":
        out = Hist1D(self.nbins, self.xmin, self.xmax, name or self.name, self.title)
        out.contents = self.contents.copy()
        out.entries = self.entries
        return out

    def reset(self) -> "Hist1D":
        self.contents[:] = 0.0
        self.entries = 0.0
        return self

    # --- axis ----------------------------------------------------------------

    @property
    def bin_width(self) -> float:
        return (self.xmax - self.xmin) / self.nbins

    def bin_center(self, ibin: int | np.ndarray):
        return self.xmin + (np.asarray(ibin, dtype=np.float64) - 0.5) * self.bin_width

    def bin_low_edge(self, ibin: int) -> float:
        return self.xmin + (ibin - 1) * self.bin_width

    def bin_up_edge(self, ibin: int) -> float:
        return self.xmin + ibin * self.bin_width

    def centers(self) -> np.ndarray:
        """Centers of the real bins 1..nbins."""
        return self.bin_center(np.arange(1, self.nbins + 1))

    def find_bin(self, x):
        """Bin index holding x: 0 for underflow, nbins+1 for overflow."""
        x = np.asarray(x, dtype=np.float64)
        span = self.xmax - self.xmin
        raw = 1 + np.floor(self.nbins * (x - self.xmin) / span).astype(np.int64)
        out = np.where(x < self.xmin, 0, np.where(x >= self.xmax, self.nbins + 1, raw))
        return int(out) if out.ndim == 0 else out

    # --- contents ------------------------------------------------------------

    def content(self, ibin: int) -> float:
        return float(self.contents[ibin])

    def set_content(self, ibin: int, value: float) -> None:
        self.contents[ibin] = value

    def values(self) -> np.ndarray:
        """Real bins 1..nbins (no flow), as a view."""
        return self.contents[1:self.nbins + 1]

    def fill(self, x, weight=1.0) -> None:
        """Add weight(s) at x, vectorized. Out-of-range lands in flow slots."""
        idx = np.atleast_1d(self.find_bin(x))
        w = np.broadcast_to(np.asarray(weight, dtype=np.float64), idx.shape)
        np.add.at(self.contents, idx, w)
        self.entries += idx.size

    def scale(self, factor: float) -> "Hist1D":
        self.contents *= factor
        return self

    def integral(self, binx1: int | None = None, binx2: int | None = None) -> float:
        """Inclusive sum of bin *contents* (not densities), flow excluded by
        default — i.e. ROOT's Integral, not an integral over x."""
        if binx1 is None:
            binx1, binx2 = 1, self.nbins
        binx1 = max(0, int(binx1))
        binx2 = min(self.nbins + 1, int(binx2))
        if binx2 < binx1:
            return 0.0
        return float(self.contents[binx1:binx2 + 1].sum())

    # --- shape ---------------------------------------------------------------

    def get_maximum_bin(self) -> int:
        """Index of the tallest real bin (flow excluded)."""
        return int(np.argmax(self.values())) + 1

    def find_first_bin_above(self, threshold: float) -> int:
        above = np.flatnonzero(self.values() > threshold)
        return int(above[0]) + 1 if above.size else -1

    def find_last_bin_above(self, threshold: float) -> int:
        above = np.flatnonzero(self.values() > threshold)
        return int(above[-1]) + 1 if above.size else -1

    def rebin(self, ngroup: int) -> "Hist1D":
        """Merge groups of ngroup bins, in place.

        Any bins left over when nbins is not a multiple of ngroup go to
        overflow and the upper edge shrinks, as ROOT's Rebin does.
        """
        ngroup = int(ngroup)
        if ngroup <= 1:
            return self
        new_nbins = self.nbins // ngroup
        merged = np.zeros(new_nbins + 2, dtype=np.float64)
        merged[0] = self.contents[0]
        body = self.values()
        usable = new_nbins * ngroup
        merged[1:new_nbins + 1] = body[:usable].reshape(new_nbins, ngroup).sum(axis=1)
        # leftovers + the old overflow become the new overflow
        merged[new_nbins + 1] = body[usable:].sum() + self.contents[self.nbins + 1]
        self.xmax = self.xmin + usable * self.bin_width
        self.nbins = new_nbins
        self.contents = merged
        return self
