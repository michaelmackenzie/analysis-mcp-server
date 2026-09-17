"""A spectrum: bin contents on a uniform grid, and the few things done to one.

Every histogram in play here -- the ones EdepAna writes, the DIO table, the
tracker resolution -- is uniform-binned, so a spectrum needs nothing more than
an array of bin contents plus the axis' lower edge and bin width. That makes
each operation one numpy call: merging bins is a reshape and a sum, moving one
binning onto another is a digitize and an add, and smearing by a response is
`np.convolve`. There is no bin lookup, no under/overflow slot, and no ROOT
`Integral` convention to remember -- a total is `values.sum()`.

A `Kernel` is a response expressed as probability mass per offset bin on the
target's own grid, which is what makes the convolution a single call.
Convolutions drop whatever they push off the axis instead of piling it up at
the edges, as smearing physically should.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import NamedTuple

import numpy as np


@dataclass(frozen=True)
class Spectrum:
    """Bin contents on a uniform axis. Operations return a new Spectrum."""

    values: np.ndarray       # contents of each bin, no under/overflow
    xmin: float
    width: float
    name: str = ""
    title: str = ""
    entries: float = 0.0     # fill count, kept for normalizing by it later

    @classmethod
    def from_uproot(cls, obj, name: str = "") -> "Spectrum":
        """Build from an uproot TH1, keeping its GetEntries()."""
        edges = obj.axis().edges()
        return cls(
            values=np.asarray(obj.values(flow=False), dtype=np.float64),
            xmin=float(edges[0]),
            width=float((edges[-1] - edges[0]) / (len(edges) - 1)),
            name=name or obj.name,
            title=obj.title,
            entries=float(obj.member("fEntries")),
        )

    @property
    def nbins(self) -> int:
        return int(self.values.size)

    @property
    def xmax(self) -> float:
        return self.xmin + self.nbins * self.width

    def centers(self) -> np.ndarray:
        return self.xmin + (np.arange(self.nbins) + 0.5) * self.width

    def edges(self) -> np.ndarray:
        return self.xmin + np.arange(self.nbins + 1) * self.width

    def scaled(self, factor: float) -> "Spectrum":
        return replace(self, values=self.values * factor)

    def rebin(self, ngroup: int) -> "Spectrum":
        """Merge groups of ngroup bins. Bins left over at the top are dropped."""
        ngroup = int(ngroup)
        if ngroup <= 1:
            return self
        n = self.nbins // ngroup
        merged = self.values[:n * ngroup].reshape(n, ngroup).sum(axis=1)
        return replace(self, values=merged, width=self.width * ngroup)

    def regrid(self, like: "Spectrum") -> "Spectrum":
        """Sum these contents into `like`'s bins; content off its axis is dropped.

        For going coarser (a finely binned DIO spectrum onto the signal's
        binning), so that spectra to be compared bin by bin share one axis.
        """
        index = np.floor((self.centers() - like.xmin) / like.width).astype(np.int64)
        inside = (index >= 0) & (index < like.nbins)
        values = np.bincount(index[inside], weights=self.values[inside],
                             minlength=like.nbins)
        return replace(like, values=values, name=self.name, title=self.title,
                       entries=self.entries)

    def smear(self, kernel: "Kernel") -> "Spectrum":
        """Redistribute each bin's content over the kernel's offsets."""
        full = np.convolve(self.values, kernel.mass)
        values = np.zeros(self.nbins)
        # full[k] is the content landing in bin k + kernel.first_bin.
        lo = max(0, kernel.first_bin)
        hi = min(self.nbins, kernel.first_bin + full.size)
        if hi > lo:
            values[lo:hi] = full[lo - kernel.first_bin:hi - kernel.first_bin]
        return replace(self, values=values)


class Kernel(NamedTuple):
    """A response as probability mass per offset bin of some target grid.

    `mass[k]` is the fraction of a bin's content that moves `first_bin + k`
    bins along, so a one-sided response (energy loss, never a gain) costs no
    zero padding on the other side.
    """

    mass: np.ndarray
    first_bin: int

    @classmethod
    def gaussian(cls, width: float, sigma: float, n_sigma: float = 5.0) -> "Kernel":
        """A Gaussian of width `sigma`, integrated over each offset bin."""
        half = max(1, int(math.ceil(n_sigma * sigma / width)))
        edges = (np.arange(-half, half + 2) - 0.5) * width
        cdf = np.array([0.5 * math.erf(e / (sigma * math.sqrt(2.0))) for e in edges])
        mass = np.diff(cdf)
        return cls(mass / mass.sum(), -half)   # renormalize the truncated tails

    @classmethod
    def from_density(cls, response: "Spectrum", width: float) -> "Kernel":
        """Resample a measured response (a density in offset) onto `width`.

        Interpolated onto the target's bins and rescaled to carry the same
        total as the measured response -- which is deliberately not 1 when the
        caller has folded an efficiency into it; that factor rides along into
        whatever is smeared. Leading and trailing empty bins are trimmed, so a
        response measured over a wide axis costs only its filled part.
        """
        lo = int(math.floor(response.xmin / width))
        hi = int(math.ceil(response.xmax / width))
        offsets = np.arange(lo, hi + 1) * width
        mass = np.interp(offsets, response.centers(), response.values,
                         left=0.0, right=0.0) * width
        total = response.values.sum() * response.width
        if mass.sum() > 0.0:
            mass = mass * (total / mass.sum())
        filled = np.flatnonzero(mass)
        if filled.size:
            lo += int(filled[0])
            mass = mass[filled[0]:filled[-1] + 1]
        return cls(mass, lo)

    def as_spectrum(self, width: float, name: str = "") -> "Spectrum":
        """The kernel as a spectrum on that grid, for plotting."""
        return Spectrum(self.mass, (self.first_bin - 0.5) * width, width, name=name)
