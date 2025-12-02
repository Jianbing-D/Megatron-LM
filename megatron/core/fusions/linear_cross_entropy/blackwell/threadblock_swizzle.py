# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

import typing

import cutlass
import cutlass.cute as cute

class ThreadBlockSwizzle:
    """
    CTA Raster
    """
    def __init__(self,
                 N: int = 1,
                 use_2cta_instrs: bool = False):
        self.N = N
        self._log_tile: cutlass.Constexpr[cutlass.Int32] = 0
        self.use_2cta_instrs = use_2cta_instrs

    def get_log_tile(
        self,
        tiled_shape: typing.Tuple[int, int, int],
    ) -> int:
        n = tiled_shape[1]
        if self.N >= 8 and n >= 6:
            return 3
        elif self.N >= 4 and n >= 3:
            return 2
        elif self.N >= 2 and n >= 2:
            return 1
        else:
            return 0

    def get_grid_shape(
        self,
        tiled_shape: typing.Tuple[int, int, int],
    ) -> typing.Tuple[int, int, int]:
        log_tile: int = self.get_log_tile(tiled_shape)
        tile: int = 1 << log_tile
        scale: int = 2 if self.use_2cta_instrs else 1
        self._log_tile = cutlass.const_expr(log_tile)
        return (
            ((tiled_shape[0] // scale) * tile) * scale,
            (tiled_shape[1] + tile - 1) // tile,
            tiled_shape[2]
        )

    @cute.jit
    def get_tile_offset(
        self,
        bidx: cutlass.Int32,
        bidy: cutlass.Int32,
        bidz: cutlass.Int32,
    ) -> typing.Tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32]:
        return (
            bidx >> self._log_tile,
            (bidy << self._log_tile) + ((bidx) & ((1 << self._log_tile) - 1)),
            bidz,
        )


class HorizontalThreadBlockSwizzle:

    def get_grid_shape(
        self,
        tiled_shape: typing.Tuple[int, int, int],
    ) -> typing.Tuple[int, int, int]:
        return (
            tiled_shape[1],
            tiled_shape[0],
            tiled_shape[2],
        )

    @cute.jit
    def get_tile_offset(
        self,
        bidx: cutlass.Int32,
        bidy: cutlass.Int32,
        bidz: cutlass.Int32,
    ) -> typing.Tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32]:
        return (
            bidy,
            bidx,
            bidz,
        )
    