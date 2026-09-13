"""Small, readable NCHW image layers. Loops prioritize clarity over speed."""

import importlib

import numpy as np

from ..tensor import Tensor
from .modules import Module, Parameter, _positive_int

_core = importlib.import_module("tensorforge.tensor")


class Conv2d(Module):
    """2D cross-correlation with square kernels and integer stride/padding.

    Input: (batch, in_channels, height, width).
    Output: (batch, out_channels, out_height, out_width).
    No dilation or channel groups in this release.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        for name, value in (
            ("in_channels", in_channels),
            ("out_channels", out_channels),
            ("kernel_size", kernel_size),
            ("stride", stride),
        ):
            _positive_int(value, name)
        if type(padding) is not int or padding < 0:
            raise ValueError("padding must be a nonnegative integer.")
        self.kernel_size, self.stride, self.padding = kernel_size, stride, padding
        bound = 1 / np.sqrt(in_channels * kernel_size**2)
        self.weight = Parameter(
            _core._rng.uniform(-bound, bound, (out_channels, in_channels, kernel_size, kernel_size))
        )
        self.bias = Parameter(np.zeros(out_channels)) if bias else None

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.weight.shape[1]:
            raise ValueError("Conv2d expects NCHW input with matching channels.")
        k, s, p = self.kernel_size, self.stride, self.padding
        n, _, height, width = x.shape
        oh, ow = (height + 2 * p - k) // s + 1, (width + 2 * p - k) // s + 1
        if oh <= 0 or ow <= 0:
            raise ValueError("Kernel is larger than the padded input.")
        padded = np.pad(x._data, ((0, 0), (0, 0), (p, p), (p, p)))
        weight = self.weight._data
        result = np.zeros((n, weight.shape[0], oh, ow), dtype=np.result_type(x.dtype, weight.dtype))
        for row in range(oh):
            for col in range(ow):
                patch = padded[:, :, row * s : row * s + k, col * s : col * s + k]
                # Sum input channels and kernel axes, retaining batch/output axes.
                result[:, :, row, col] = np.einsum("nchw,ochw->no", patch, weight)

        def vjp(g):
            gx = np.zeros(padded.shape, dtype=np.result_type(g.dtype, weight.dtype))
            gw = np.zeros_like(weight)
            for row in range(oh):
                for col in range(ow):
                    patch = padded[:, :, row * s : row * s + k, col * s : col * s + k]
                    incoming = g[:, :, row, col]
                    gx[:, :, row * s : row * s + k, col * s : col * s + k] += np.einsum(
                        "no,ochw->nchw", incoming, weight
                    )
                    gw += np.einsum("no,nchw->ochw", incoming, patch)
            # Slice with p:p+size so padding=0 also works.
            return gx[:, :, p : p + height, p : p + width], gw

        out = Tensor._op(result, (x, self.weight), vjp)
        return out if self.bias is None else out + self.bias.reshape(1, -1, 1, 1)


class MaxPool2d(Module):
    """Square max-pooling without padding; ties select the first maximum."""

    def __init__(self, kernel_size=2, stride=None):
        super().__init__()
        _positive_int(kernel_size, "kernel_size")
        stride = kernel_size if stride is None else stride
        _positive_int(stride, "stride")
        self.kernel_size, self.stride = kernel_size, stride

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError("MaxPool2d expects NCHW input.")
        n, c, height, width = x.shape
        k, s = self.kernel_size, self.stride
        oh, ow = (height - k) // s + 1, (width - k) // s + 1
        if oh <= 0 or ow <= 0:
            raise ValueError("Pooling window is larger than the input.")
        result = np.empty((n, c, oh, ow), dtype=x.dtype)
        winners = np.empty((n, c, oh, ow), dtype=np.int64)
        for row in range(oh):
            for col in range(ow):
                patch = x._data[:, :, row * s : row * s + k, col * s : col * s + k].reshape(
                    n, c, -1
                )
                winners[:, :, row, col] = patch.argmax(axis=-1)
                result[:, :, row, col] = patch.max(axis=-1)

        def vjp(g):
            gx = np.zeros_like(x._data)
            batch, channel = np.arange(n)[:, None], np.arange(c)[None, :]
            for row in range(oh):
                for col in range(ow):
                    winner = winners[:, :, row, col]
                    # Overlapping windows may select the same input: accumulate.
                    np.add.at(
                        gx,
                        (batch, channel, row * s + winner // k, col * s + winner % k),
                        g[:, :, row, col],
                    )
            return (gx,)

        return Tensor._op(result, (x,), vjp)


class AvgPool2d(Module):
    """Square average pooling without padding, including overlapping windows."""

    def __init__(self, kernel_size=2, stride=None):
        super().__init__()
        _positive_int(kernel_size, "kernel_size")
        stride = kernel_size if stride is None else stride
        _positive_int(stride, "stride")
        self.kernel_size, self.stride = kernel_size, stride

    def forward(self, x):
        from ..tensor import stack

        if x.ndim != 4:
            raise ValueError("AvgPool2d expects NCHW input.")
        k, s = self.kernel_size, self.stride
        oh, ow = (x.shape[-2] - k) // s + 1, (x.shape[-1] - k) // s + 1
        if oh <= 0 or ow <= 0:
            raise ValueError("Pooling window is larger than the input.")
        rows = []
        for row in range(oh):
            columns = [
                x[:, :, row * s : row * s + k, col * s : col * s + k].mean(axis=(-2, -1))
                for col in range(ow)
            ]
            rows.append(stack(columns, axis=-1))
        return stack(rows, axis=-2)
