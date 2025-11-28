# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

from typing import Optional, Tuple, Type

import cuda.bindings.driver as cuda  # type: ignore
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline  # type: ignore
import cutlass.utils as utils  # type: ignore
import cutlass.utils.blackwell_helpers as sm100_utils  # type: ignore
from cutlass.cute.nvgpu import cpasync, tcgen05

from ..utils import EntropyReductionEnum

import math

SM100_TMEM_CAPACITY_COLUMNS: int = 512

def make_thread_cooperative_group(
    size: int,
    alignment: Optional[int] = None
):
    """
    Create a thread cooperative group.
    """
    return pipeline.CooperativeGroup(
        pipeline.Agent.Thread, size,
        alignment=alignment if alignment is not None else size
    )

def next_power_of_two(n: int) -> int:
    """
    Calculate the next power of two for a given number.
    """
    if n <= 0:
        return 1
    return 2 ** math.ceil(math.log2(n))

class BwdTwoKernelsGradHidden:
    """
    This class includes two separate kernels for dHidden.
    """
    def __init__(
        self,
        reduction: int,
        acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
        use_2cta_instrs: bool = False,
        mma_tiler_mn: Tuple[int, int] = (128, 128),
        vocab_per_split: int = 512
    ):
        self.REDUCTION: cutlass.Constexpr[cutlass.Int32] = cutlass.const_expr(reduction)
        self.acc_dtype = acc_dtype
        self.use_2cta_instrs = use_2cta_instrs
        self.vocab_per_split = vocab_per_split

        # as the a/b dtype is BF16/FP16, so that 64 elements can be loaded with 128B swizzle
        self.mma_tiler = (*mma_tiler_mn, 64)
        self.k1st_mma_tiler = self.mma_tiler
        self.k2nd_mma_tiler = (
            self.mma_tiler[0],
            self.mma_tiler[2],
            self.mma_tiler[1]
        )

        self.cta_group = tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        self.cluster_shape_mn = (2, 1) if self.use_2cta_instrs else (1, 1)

        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")

        self.threads_per_warp: int = 32

        self.load_ab_warp_ids = 0
        self.mma_ab_warp_ids = 1
        self.load_trans_b_warp_ids = 2
        self.mma_trans_b_warp_ids = 3
        self.scale_warp_ids = (4, 5, 6, 7)
        self.epi_warp_ids = (8, 9, 10, 11)

        self.warps_per_cta: int = len(
            (self.load_ab_warp_ids,
             self.mma_ab_warp_ids,
             self.load_trans_b_warp_ids,
             self.mma_trans_b_warp_ids,
             *self.scale_warp_ids,
             *self.epi_warp_ids)
        )
        self.threads_per_cta: int = self.threads_per_warp * self.warps_per_cta
        
        self.warpgroups_per_cta = (self.warps_per_cta + 4 - 1) // 4
        self.threads_per_warpgroup: int = 128
        assert self.threads_per_warpgroup * self.warpgroups_per_cta == self.threads_per_cta


        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=self.threads_per_cta
        )
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=2, num_threads=self.threads_per_warp
        )

        self.buffer_align_bytes: int = 1024
        self.num_regs_other: int = 32
        self.num_regs_epi: int = 192

    def _compute_grid(
        self,
        problem_mnk: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        cta_tiler: Tuple[int, int, int],
        num_epi_acc_stage: int,
    ) -> Tuple[int, int, int]:
        cluster_shape_mnk = (*cluster_shape_mn, 1)

        # the output shape is (M, K)
        grid = cute.round_up(
            (
                cute.ceil_div(problem_mnk[0], cta_tiler[0]),
                cute.ceil_div(problem_mnk[2], cta_tiler[1] * num_epi_acc_stage),
                1, # TODO: perhaps split problem_mnk[1] to multiple chunks
            ),
            cluster_shape_mnk,
        )
        return grid

    def _compute_stages(
        self,
        k1st_tiled_mma: cute.TiledMma,
        k1st_mma_tiler: Tuple[int, int, int],
        k2nd_mma_tiler: Tuple[int, int, int],
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
    ):
        # TODO: maybe we can decide these parameters
        # based on the input hyper-parameters
        num_acc_stage = 2
        num_ab_stage = 4
        num_trans_b_stage = 3
        num_epi_acc_stage = 3
        # split 1st MMA TMEM into multiple tiles
        num_scale_stage_per_tile = 2
        # split 2nd MMA TMEM into multiple tiles
        num_epi_stage_per_tile = 2

        return (num_acc_stage,
                num_ab_stage,
                num_trans_b_stage,
                num_epi_acc_stage,
                num_scale_stage_per_tile,
                num_epi_stage_per_tile)

    def _setup_attributes(
        self,
        k1st_tiled_mma: cute.TiledMma,
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
    ):
        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (k1st_tiled_mma.thr_id.shape,)
        )
        
        (self.num_acc_stage, 
         self.num_ab_stage, 
         self.num_trans_b_stage, 
         self.num_epi_acc_stage,
         self.num_scale_stage_per_tile, 
         self.num_epi_stage_per_tile) =\
             self._compute_stages(
                k1st_tiled_mma,
                self.k1st_mma_tiler,
                self.k2nd_mma_tiler,
                a_dtype,
                b_dtype,
             )

        self.tmem_1st_acc_cols = self.k1st_mma_tiler[1] * self.num_acc_stage
        self.tmem_2nd_acc_cols = self.k2nd_mma_tiler[1] * self.num_epi_acc_stage
        self.tmem_1st_acc_convert_cols = self.k1st_mma_tiler[1] * (a_dtype.width // 8) // (self.acc_dtype.width // 8)

        tmem_cols = [self.tmem_1st_acc_cols, self.tmem_1st_acc_convert_cols, self.tmem_2nd_acc_cols]

        self.tmem_1st_acc_offset = 0
        self.tmem_1st_acc_convert_offset = self.tmem_1st_acc_offset + self.tmem_1st_acc_cols
        self.tmem_2nd_acc_offset = self.tmem_1st_acc_convert_offset + self.tmem_1st_acc_convert_cols

        self.tmem_alloc_cols = next_power_of_two(sum(tmem_cols))
        assert self.tmem_alloc_cols <= SM100_TMEM_CAPACITY_COLUMNS

    @cute.kernel
    def kernel(
        self,
        k1st_tiled_mma: cute.TiledMma,
        k2nd_tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB: cute.Tensor,
        tma_atom_trans_b: cute.CopyAtom,
        mTransB: cute.Tensor,
        mLabels: cute.Tensor,
        mDlogprobs: cute.Tensor,
        mMaximum: cute.Tensor,
        mAccu: cute.Tensor,
        scalarNumValidTokens: cute.Pointer,
        mDHidden: cute.Tensor,
        ignore_index: cutlass.Int64,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        convtert_a_tmem_layout_staged: cute.ComposedLayout,
        trans_b_smem_layout_staged: cute.ComposedLayout,
        cluster_layout_vmnk: cute.Layout,
        problem_mnk: Tuple[int, int, int],
        rank: cutlass.Int32,
    ) -> None:
        """
        The backward kernel for dHidden.
        """
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, tidy, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()

        # FIXME: block swizzling applied here
        pidm, pidn = bidx, bidy            

        cta_rank_in_cluster = 0
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)

        # prefetch tma descriptors
        if warp_idx == self.load_ab_warp_ids:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_trans_b)

        smem = utils.SmemAllocator()
        smem_storage = smem.allocate(self.shared_storage)

        ab_pipeline = pipeline.PipelineTmaUmma.create(
            num_stages=self.num_ab_stage,
            producer_group=make_thread_cooperative_group(len([self.load_ab_warp_ids])),
            consumer_group=make_thread_cooperative_group(len([self.mma_ab_warp_ids])),
            tx_count=self.tma_copy_ab_bytes,
            barrier_storage=smem_storage.load_ab_mbar_ptr.data_ptr(),
        )
        ab_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_ab_stage
        )
        ab_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_ab_stage
        )

        k1st_mma_pipeline = pipeline.PipelineUmmaAsync.create(
            num_stages=self.num_acc_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_ab_warp_ids])),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.scale_warp_ids)
            ),
            barrier_storage=smem_storage.k1st_mma_mbar_ptr.data_ptr(),
        )
        k1st_mma_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_acc_stage
        )
        k1st_mma_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_acc_stage
        )

        convertAcc_pipeline = pipeline.PipelineAsyncUmma.create(
            num_stages=1,
            producer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.scale_warp_ids)
            ),
            consumer_group=make_thread_cooperative_group(len([self.mma_trans_b_warp_ids])),
            barrier_storage=smem_storage.convert_acc_mbar_ptr.data_ptr()
        )
        convertAcc_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, 1
        )
        convertAcc_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, 1
        )

        transB_pipeline = pipeline.PipelineTmaUmma.create(
            num_stages=self.num_trans_b_stage,
            producer_group=make_thread_cooperative_group(len([self.load_trans_b_warp_ids])),
            consumer_group=make_thread_cooperative_group(len([self.mma_trans_b_warp_ids])),
            tx_count=self.tma_copy_trans_b_bytes,
            barrier_storage=smem_storage.load_trans_b_mbar_ptr.data_ptr()
        )
        transB_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_trans_b_stage
        )
        transB_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_trans_b_stage
        )

        k2nd_mma_pipeline = pipeline.PipelineUmmaAsync.create(
            num_stages=self.num_epi_acc_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_trans_b_warp_ids])),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.epi_warp_ids)
            ),
            barrier_storage=smem_storage.k2nd_mma_mbar_ptr.data_ptr()
        )
        k2nd_mma_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_epi_acc_stage
        )
        k2nd_mma_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_epi_acc_stage
        )

        tmem_dealloc_mbar_ptr = smem_storage.tmem_dealloc_mbar_ptr.data_ptr()
        if warp_idx == self.load_ab_warp_ids:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(
                    tmem_dealloc_mbar_ptr,
                    self.threads_per_warp * len(self.epi_warp_ids)
                )
                cute.arch.mbarrier_init_fence()

        # -------- matrix partition ------------ #
        # swizzle o [(tileM, tileK), loopM, loopK, (stage)]
        sA = smem_storage.sA.get_tensor(
            a_smem_layout_staged.outer,
            swizzle=a_smem_layout_staged.inner
        )
        # swizzle o [(tileN, tileK), loopN, loopK, (stage)]
        sB = smem_storage.sB.get_tensor(
            b_smem_layout_staged.outer,
            swizzle=b_smem_layout_staged.inner
        )
        # swizzle o [(tileN, tileK), loopN, loopK, (stage)]
        sTransB = smem_storage.sTransB.get_tensor(
            trans_b_smem_layout_staged.outer,
            swizzle=trans_b_smem_layout_staged.inner
        )
        
        # slice relates to CTA-id
        k1st_thr_mma = k1st_tiled_mma.get_slice(0)
        # these are only SMEM descriptors, so one item for each MMA
        # [MMA, loopM, loopK, (stage)]
        tCsA = k1st_thr_mma.make_fragment_A(sA)
        # [MMA, loopN, loopK, (stage)]
        tCsB = k1st_thr_mma.make_fragment_B(sB)

        # [tileM, tileK, loopK]
        gA = cute.local_tile(
            mA, (self.k1st_mma_tiler[0], self.k1st_mma_tiler[2]), (pidm, None)
        )
        # [tileN, tileK, loopN, loopK]
        gB = cute.local_tile(
            mB, (self.k1st_mma_tiler[1], self.k1st_mma_tiler[2]), (None, None)
        )

        mTransBChunk = cute.local_tile(
            mTransB,
            (self.k2nd_mma_tiler[1] * self.num_epi_acc_stage, self.k2nd_mma_tiler[2]),
            (pidn, None)
        )
        # [tileN, tileK, num_epi_acc_stage, 1, loopK]
        gTransB = cute.flat_divide(
            mTransBChunk,
            (self.k2nd_mma_tiler[1], self.k2nd_mma_tiler[2])
        )
        k2nd_left_idx: cutlass.Int64 = pidn * self.num_epi_acc_stage * self.k2nd_mma_tiler[1]
        k2nd_right_idx: cutlass.Int64 = min(
            (pidn + 1) * self.num_epi_acc_stage * self.k2nd_mma_tiler[1],
            cute.size(mTransB, mode=[0])
        )
        k2nd_valid_acc_stage: cutlass.Int64 = cute.ceil_div(
            (k2nd_right_idx - k2nd_left_idx),
            self.k2nd_mma_tiler[1]
        )

        mDHiddenChunk = cute.local_tile(
            mDHidden,
            (self.epi_tile[0], 
             self.epi_tile[1] * self.num_epi_acc_stage),
            (pidm, pidn)
        )
        # [tileM, tileN, loopM, loopN]
        gDHidden = cute.flat_divide(
            mDHiddenChunk,
            (self.epi_tile[0], self.epi_tile[1])
        )
        
        # make sure SMEM and GMEM tensor has the same size in the first rank
        # [MMA, tileCntM, tileCntK, loopK]
        tCgA = k1st_thr_mma.partition_A(gA)
        # [MMA, tileCntN, tileCntK, loopN, loopK]
        tCgB = k1st_thr_mma.partition_B(gB)

        # [1]
        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        # [CPY, stage] & [CPY, loopK]
        tTMAsA, tTMAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2], # cta_coord,
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3)
        )

        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        tTMAsB, tTMAgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1], # cta_coord
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3)
        )

        k2nd_thr_mma = k2nd_tiled_mma.get_slice(0)
        # [MMA, tileCntN, tileCntK, num_epi_acc_stage, 1, loopK]
        tOgTransB = k2nd_thr_mma.partition_B(gTransB)

        tTMAsTransB, tTMAgTransB = cpasync.tma_partition(
            tma_atom_trans_b,
            block_in_cluster_coord_vmnk[1], # cta_coord
            b_cta_layout,
            cute.group_modes(sTransB, 0, 3),
            cute.group_modes(tOgTransB, 0, 3)
        )

        tOsB = k2nd_thr_mma.make_fragment_B(sTransB)

        # ----- Allocate TMEM ----- #
        tmem_holding_buf = smem_storage.tmem_holding_buf
        if warp_idx == self.load_ab_warp_ids:
            cute.arch.alloc_tmem(
                self.tmem_alloc_cols,
                tmem_holding_buf,
                is_two_cta=self.use_2cta_instrs
            )
        self.cta_sync_barrier.arrive_and_wait()
        tmem_ptr = cute.arch.retrieve_tmem_ptr(
            self.acc_dtype, alignment=1,
            ptr_to_buffer_holding_addr=tmem_holding_buf
        )

        # ----- Reshape TMEM ----- #
        k1st_acc_tmem_shape = (self.k1st_mma_tiler[0], self.tmem_1st_acc_cols)
        k1st_acc_shape = k1st_thr_mma.partition_shape_C(k1st_acc_tmem_shape)
        tCtC_fake = k1st_thr_mma.make_fragment_C(k1st_acc_shape)
        # [(tileM, tileN), stageM, stageN]
        tCtC = cute.make_tensor(tmem_ptr, tCtC_fake.layout)

        k1st_acc_tmem_half_shape = (self.k1st_mma_tiler[0], self.k1st_mma_tiler[1])
        k1st_acc_shape_half = k1st_thr_mma.partition_shape_C(k1st_acc_tmem_half_shape)
        tCtC_half_fake_in_fp32 = cute.flat_divide(
            k1st_thr_mma.make_fragment_C(k1st_acc_shape_half)[(None, None), 0, 0],
            (self.scale_tile[0],
             self.scale_tile[1] // self.num_scale_stage_per_tile)
        )[(None, None, 0, 0)]
        # [tileM, tileN * BF16 / FP32]
        tCtC_half_in_fp32 = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + self.tmem_1st_acc_convert_offset, 
                            dtype=self.acc_dtype),
            tCtC_half_fake_in_fp32.layout
        )

        type_scale: cutlass.Constexpr[cutlass.Int32] = self.acc_dtype.width // tma_atom_b.value_type.width

        k2nd_acc_tmem_shape = (self.k2nd_mma_tiler[0], self.tmem_2nd_acc_cols)
        k2nd_acc_shape = k2nd_thr_mma.partition_shape_C(k2nd_acc_tmem_shape)
        tOtC_fake = k2nd_thr_mma.make_fragment_C(k2nd_acc_shape)
        # [(tileM, tileN), stageM, stageN]
        tOtC = cute.make_tensor(tmem_ptr + self.tmem_2nd_acc_offset, tOtC_fake.layout)

        tConvertA = cute.make_tensor(
            tCtC_half_in_fp32.iterator,
            convtert_a_tmem_layout_staged.outer
        )
        tOtA_fake = k2nd_thr_mma.make_fragment_A(tConvertA)[None, None, None, 0]
        tOtA = cute.make_tensor(
            cute.recast_ptr(tConvertA.iterator, dtype=tma_atom_b.value_type),
            tOtA_fake.layout
        )

        # ------ load AB -------- #
        if warp_idx == self.load_ab_warp_ids:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)

            for n in cutlass.range(cute.size(gB, mode=[2])):
                for k in cutlass.range(cute.size(gA, mode=[2])):
                    ab_pipeline.producer_acquire(ab_producer_state)

                    cute.copy(
                        tma_atom_a,
                        tTMAgA[(None, k)],
                        tTMAsA[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                    )
                    cute.copy(
                        tma_atom_b,
                        tTMAgB[(None, n, k)],
                        tTMAsB[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                    )

                    ab_pipeline.producer_commit(ab_producer_state)
                    ab_producer_state.advance()

        # ------ 1st MMA -------- #
        if warp_idx == self.mma_ab_warp_ids:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)

            for n in cutlass.range(cute.size(gB, mode=[2])):
                k1st_mma_pipeline.producer_acquire(k1st_mma_producer_state)
                k1st_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                for k in cutlass.range(cute.size(gA, mode=[2])):
                    ab_pipeline.consumer_wait(ab_consumer_state)

                    for kblock_idx in cutlass.range(cute.size(tCsA, mode=[2]), unroll_full=True):
                        cute.gemm(
                            k1st_tiled_mma,
                            cute.append_ones(tCtC[(None, None, k1st_mma_producer_state.index)]),
                            tCsA[(None, None, kblock_idx, ab_consumer_state.index)],
                            tCsB[(None, None, kblock_idx, ab_consumer_state.index)],
                            cute.append_ones(tCtC[(None, None, k1st_mma_producer_state.index)]),
                        )
                        k1st_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                    ab_pipeline.consumer_release(ab_consumer_state)
                    ab_consumer_state.advance()

                k1st_mma_pipeline.producer_commit(k1st_mma_producer_state)
                k1st_mma_producer_state.advance()
            

        # ------ scale Acc -------- #
        if warp_idx in self.scale_warp_ids:
            cute.arch.warpgroup_reg_alloc(self.num_regs_epi)

            copy_atom_t2r = sm100_utils.get_tmem_load_op(
                self.k1st_mma_tiler,
                utils.LayoutEnum.ROW_MAJOR,
                self.acc_dtype,
                self.acc_dtype,
                (self.scale_tile[0], self.scale_tile[1] // self.num_scale_stage_per_tile),
                self.use_2cta_instrs,
            )
            # [tileM, subTileN, stageM, CntSubTileN, stageN]
            tAccScale = cute.flat_divide(
                tCtC[(None, None), 0, None],
                (self.scale_tile[0],
                 self.scale_tile[1] // self.num_scale_stage_per_tile),
            )
            tiled_copy_t2r = tcgen05.make_tmem_copy(
                copy_atom_t2r,
                tAccScale[(None, None, 0, 0, 0)],
            )
            thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)

            tTMEM_load_tAccScale = thr_copy_t2r.partition_S(tAccScale)
            tTMEM_load_tAccScale = cute.group_modes(
                tTMEM_load_tAccScale,
                3, cute.rank(tTMEM_load_tAccScale) - 1
            )
            
            # predicates
            cAccScale = cute.make_identity_tensor(self.k1st_mma_tiler[:2])
            _tCcAccScale = k1st_thr_mma.partition_C(cAccScale)
            # [tileM, subTileN, stageM, CntSubTileN, stageN]
            tCcAccScale = cute.flat_divide(
                _tCcAccScale[((None, None), 0, None)],
                (self.scale_tile[0],
                 self.scale_tile[1] // self.num_scale_stage_per_tile),
            )
            tTMEM_load_cAccScale = thr_copy_t2r.partition_D(tCcAccScale)
            tTMEM_load_cAccScale_shape = cute.select(
                tTMEM_load_cAccScale.shape,
                mode=[0, 1, 2]
            )
            # [subTileN, 1, 1]
            tTMEM_load_rAccScale = cute.make_fragment(
                tTMEM_load_cAccScale_shape,
                self.acc_dtype
            )
            
            copy_atom_g2r_int64 = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), mLabels.element_type
            )
            copy_atom_g2r_fp32 = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), mDlogprobs.element_type
            )
            scale_thread_layout = cute.make_layout((128, 1), stride=(1, 1))
            tiled_copy_g2r_int64 = cute.make_tiled_copy_tv(
                copy_atom_g2r_int64,
                scale_thread_layout,
                cute.make_layout((1, 1))
            )
            tiled_copy_g2r_fp32 = cute.make_tiled_copy_tv(
                copy_atom_g2r_fp32,
                scale_thread_layout,
                cute.make_layout((1, 1))
            )
            thr_copy_g2r_int64 = tiled_copy_g2r_int64.get_slice(tidx)
            thr_copy_g2r_fp32 = tiled_copy_g2r_fp32.get_slice(tidx)

            # [tileM]
            gLabels = cute.local_tile(mLabels, (self.scale_tile[0],), (pidm,))
            gMaximum = cute.local_tile(mMaximum, (self.scale_tile[0],), (pidm,))
            gAccu = cute.local_tile(mAccu, (self.scale_tile[0],), (pidm,))
            
            # slice along M direction
            tMCAcc = thr_copy_g2r_int64.partition_S(cAccScale)[(None, None, 0)]
            tMCAcc_mask = cute.make_fragment(tMCAcc.shape, cutlass.Boolean)
            # [(1, 1), 1, 1]
            tMCAcc_mask = cute.append_ones(tMCAcc_mask)
            tMCAcc_mask[0] = cute.elem_less(pidm * self.scale_tile[0] + tidx, cute.size(mA, mode=[0]))

            # [(1, 1), 1, 1]
            tMgLabels = thr_copy_g2r_int64.partition_S(cute.append_ones(gLabels))
            tMrLabels = cute.make_fragment(tMgLabels.shape, tMgLabels.element_type)
            cute.copy(tiled_copy_g2r_int64, tMgLabels, tMrLabels, pred=tMCAcc_mask)
            tMgMaximum = thr_copy_g2r_fp32.partition_S(cute.append_ones(gMaximum))
            tMrMaximum = cute.make_fragment(tMgMaximum.layout, tMgMaximum.element_type)
            cute.copy(tiled_copy_g2r_fp32, tMgMaximum, tMrMaximum, pred=tMCAcc_mask)
            tMgAccu = thr_copy_g2r_fp32.partition_S(cute.append_ones(gAccu))
            tMrAccu = cute.make_fragment(tMgAccu.layout, tMgAccu.element_type)
            cute.copy(tiled_copy_g2r_fp32, tMgAccu, tMrAccu, pred=tMCAcc_mask)

            tMrDlogprobs = cute.make_fragment(tMrAccu.layout, mDlogprobs.element_type)
            if cutlass.const_expr(self.REDUCTION == EntropyReductionEnum.kMean):
                # mean reduction
                num_valid_tokens = cute.make_tensor(scalarNumValidTokens, layout=(1,))
                tMrDlogprobs[0] = mDlogprobs[0] / num_valid_tokens[0].to(cutlass.Float32)
            elif cutlass.const_expr(self.REDUCTION == EntropyReductionEnum.kSum):
                # sum reduction
                tMrDlogprobs[0] = mDlogprobs[0]
            elif cutlass.const_expr(self.REDUCTION == EntropyReductionEnum.kNone):
                # no reduction
                gDlogprobs = cute.local_tile(mDlogprobs, (self.scale_tile[0],), (pidm,))
                tMgDlogprobs = thr_copy_g2r_fp32.partition_S(cute.append_ones(gDlogprobs))
                cute.copy(tiled_copy_g2r_fp32, tMgDlogprobs, tMrDlogprobs, pred=tMCAcc_mask)

            repeat_num = ((self.scale_tile[1] // self.num_scale_stage_per_tile)
                * (tma_atom_b.value_type.width // 8) 
                // (self.acc_dtype.width // 8))
            copy_atom_r2t = cute.make_copy_atom(
                tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(repeat_num)),
                self.acc_dtype,
            )
            tAccScale_half_in_fp32 = cute.flat_divide(
                tCtC_half_in_fp32,
                (self.scale_tile[0],
                 (self.scale_tile[1] // self.num_scale_stage_per_tile) // type_scale)
            )

            tiled_copy_r2t = tcgen05.make_tmem_copy(
                copy_atom_r2t,
                tAccScale_half_in_fp32[(None, None, 0, 0)]
            )
            thr_copy_r2t = tiled_copy_r2t.get_slice(tidx)

            tTMEM_store_tDLogits = thr_copy_r2t.partition_D(tAccScale_half_in_fp32)
            tTMEM_store_tDLogits = cute.group_modes(
                tTMEM_store_tDLogits,
                2, cute.rank(tTMEM_store_tDLogits) - 1
            )
            tCcAccScale_half_in_fp32 = cute.flat_divide(
                _tCcAccScale[((None, None), 0, None)],
                (self.scale_tile[0],
                 (self.scale_tile[1] // self.num_scale_stage_per_tile) // type_scale)
            )
            tTMEM_store_cAccScale_half_in_fp32 = thr_copy_r2t.partition_S(tCcAccScale_half_in_fp32)
            tTMEM_store_rDLogits = cute.make_fragment(
                cute.select(tTMEM_store_cAccScale_half_in_fp32.shape, mode=[0, 1, 2]),
                self.acc_dtype
            )

            tTMEM_store_rDLogits_in_half = cute.recast_tensor(tTMEM_store_rDLogits, tma_atom_b.value_type)

            # do scaling
            tMrAccu[0] = cute.arch.rcp_approx(tMrAccu[0])
            tMrDlogprobs[0] *= (tMrLabels[0] != ignore_index)
            tMr_d_acc_exp_logits = tMrDlogprobs[0] * tMrAccu[0]

            for n in cutlass.range(cute.size(gB, mode=[2])):
                k1st_mma_pipeline.consumer_wait(k1st_mma_consumer_state)

                left_idx: cutlass.Int64 = n * self.scale_tile[1]
                right_idx: cutlass.Int64 = min((n + 1) * self.scale_tile[1], problem_mnk[1])
                num_n_subtiles: cutlass.Int64 = cute.ceil_div(
                    (right_idx - left_idx), cute.size(tTMEM_load_rAccScale, mode=[0])
                )
                for n_subtile in cutlass.range(num_n_subtiles):
                    cute.copy(
                        tiled_copy_t2r,
                        tTMEM_load_tAccScale[(None, None, None, n_subtile, k1st_mma_consumer_state.index)],
                        tTMEM_load_rAccScale,
                    )

                    for idx in cutlass.range(cute.size(tTMEM_load_rAccScale, mode=[0]), unroll_full=True):
                        # exp_logits
                        tTMEM_load_rAccScale[idx] = cute.exp(tTMEM_load_rAccScale[idx] - tMrMaximum[0])

                        position: cutlass.Int64 = (
                            rank * problem_mnk[1]
                            + n * self.scale_tile[1]
                            + n_subtile * cute.size(tTMEM_load_rAccScale, mode=[0])
                            + idx
                        )
                        label_mask: cutlass.Boolean = (
                            position == tMrLabels[0] and tMrLabels[0] != ignore_index
                        )
                        # d_logits
                        tTMEM_load_rAccScale[idx] *= tMr_d_acc_exp_logits
                        tTMEM_load_rAccScale[idx] += label_mask * -tMrDlogprobs[0]

                        # apply predicate
                        valid: cutlass.Boolean = cute.elem_less(
                            pidm * self.scale_tile[0] + tTMEM_load_cAccScale[idx, 0, 0, 0, n_subtile, 0][0],
                            problem_mnk[0]
                        ) and cute.elem_less(
                            n * self.scale_tile[1]
                            + tTMEM_load_cAccScale[idx, 0, 0, 0, n_subtile, 0][1],
                            problem_mnk[1]
                        )
                        tTMEM_load_rAccScale[idx] *= valid

                    # type conversion
                    tTMEM_store_rDLogits_in_half.store(
                        tTMEM_load_rAccScale.load().to(tTMEM_store_rDLogits_in_half.element_type)
                    )

                    # store back to TMEM
                    if n_subtile == 0:
                        convertAcc_pipeline.producer_acquire(convertAcc_producer_state)

                    cute.copy(
                        tiled_copy_r2t,
                        tTMEM_store_rDLogits,
                        tTMEM_store_tDLogits[(None, None, None, n_subtile)]
                    )

                convertAcc_pipeline.producer_commit(convertAcc_producer_state)
                convertAcc_producer_state.advance()

                k1st_mma_pipeline.consumer_release(k1st_mma_consumer_state)
                k1st_mma_consumer_state.advance()

        # ------ load trans B -------- #
        if warp_idx == self.load_trans_b_warp_ids:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)

            for k in cutlass.range(cute.size(gB, mode=[2])):
                for n in cutlass.range(k2nd_valid_acc_stage):
                    transB_pipeline.producer_acquire(transB_producer_state)

                    cute.copy(
                        tma_atom_trans_b,
                        tTMAgTransB[(None, n, None, k)],
                        cute.append_ones(tTMAsTransB[(None, transB_producer_state.index)]),
                        tma_bar_ptr=transB_pipeline.producer_get_barrier(transB_producer_state)
                    )

                    transB_pipeline.producer_commit(transB_producer_state)
                    transB_producer_state.advance()

        # ------- 2nd MMA -------- #
        if warp_idx == self.mma_trans_b_warp_ids:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)

            for k in cutlass.range(cute.size(gB, mode=[2])):
                convertAcc_pipeline.consumer_wait(convertAcc_consumer_state)
                
                for n in cutlass.range(k2nd_valid_acc_stage):
                    transB_pipeline.consumer_wait(transB_consumer_state)

                    if k == 0:
                        k2nd_mma_pipeline.producer_acquire(k2nd_mma_producer_state)

                    k2nd_tiled_mma.set(tcgen05.Field.ACCUMULATE, k != 0)
                    for kblock_idx in cutlass.range(cute.size(tOsB, mode=[2]), unroll_full=True):
                        cute.gemm(
                            k2nd_tiled_mma,
                            cute.append_ones(tOtC[(None, None, n)]),
                            tOtA[(None, None, kblock_idx)],
                            tOsB[(None, None, kblock_idx, transB_consumer_state.index)],
                            cute.append_ones(tOtC[(None, None, n)]),
                        )
                        k2nd_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                    transB_pipeline.consumer_release(transB_consumer_state)
                    transB_consumer_state.advance()

                    if k == cute.size(gB, mode=[2]) - 1:
                        k2nd_mma_pipeline.producer_commit(k2nd_mma_producer_state)
                        k2nd_mma_producer_state.advance()

                convertAcc_pipeline.consumer_release(convertAcc_consumer_state)
                convertAcc_consumer_state.advance()

        # ------ epilogue -------- #
        if warp_idx in self.epi_warp_ids:
            cute.arch.warpgroup_reg_alloc(self.num_regs_epi)

            copy_atom_t2r = sm100_utils.get_tmem_load_op(
                self.k2nd_mma_tiler,
                utils.LayoutEnum.ROW_MAJOR,
                self.acc_dtype,
                self.acc_dtype,
                (self.epi_tile[0], 
                 self.epi_tile[1] // self.num_epi_stage_per_tile),
                self.use_2cta_instrs
            )
            # [tileM, subTileN, stageM, CntSubTileN, stageN]
            tO = cute.flat_divide(
                tOtC[((None, None), 0, None)],
                (self.epi_tile[0], 
                 self.epi_tile[1] // self.num_epi_stage_per_tile)
            )
            tiled_copy_t2r = tcgen05.make_tmem_copy(
                copy_atom_t2r,
                tO[(None, None, 0, 0, 0)]
            )
            thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)

            tTMEM_load_tO = thr_copy_t2r.partition_S(tO)
            tTMEM_load_tO = cute.group_modes(
                tTMEM_load_tO,
                3, cute.rank(tTMEM_load_tO) - 1
            )
            
            cO = cute.make_identity_tensor((self.epi_tile[0], self.epi_tile[1] * self.num_epi_acc_stage))
            tOcO = k2nd_thr_mma.partition_C(cO)
            tOcOSingle = cute.flat_divide(
                tOcO[((None, None), 0, None)],
                (self.epi_tile[0],
                 self.epi_tile[1] // self.num_epi_stage_per_tile)
            )
            tTMEM_load_cO = thr_copy_t2r.partition_D(tOcOSingle)
            tTMEM_load_cO_shape = cute.select(
                tTMEM_load_cO.shape,
                mode=[0, 1, 2]
            )
            # [subTileN, 1, 1]
            tTMEM_load_rO = cute.make_fragment(
                tTMEM_load_cO_shape,
                self.acc_dtype
            )
            tTMEM_load_rO_half = cute.make_fragment(
                tTMEM_load_cO_shape,
                mDHidden.element_type
            )

            epi_thread_layout = cute.make_layout((128, 1), stride=(1, 1))
            # blackwell supports STG.256
            copy_atom_r2g = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                tTMEM_load_rO_half.element_type,
                num_bits_per_copy=256
            )
            tiled_copy_r2g = cute.make_tiled_copy_tv(
                copy_atom_r2g,
                epi_thread_layout,
                copy_atom_r2g.layout_dst_tv
            )
            thr_copy_r2g = tiled_copy_r2g.get_slice(tidx)

            # [CPY, loopM, loopN]
            tR2GgDHidden = thr_copy_r2g.partition_D(gDHidden)

            tR2GrDHidden = cute.tiled_divide(
                tTMEM_load_rO_half,
                tR2GgDHidden.layout.shape[0]
            )

            # predicates
            tR2GcO = thr_copy_r2g.partition_S(cO)
            tR2GcO_pred = cute.make_fragment(tR2GcO.shape, cutlass.Boolean)
            for chunk in cutlass.range(cute.size(tR2GcO, mode=[2]), unroll_full=True):
                for elem in cutlass.range(cute.size(tR2GcO, mode=[0]), unroll_full=True):
                    tR2GcO_pred[elem, 0, chunk] = cute.elem_less(
                        pidm * self.epi_tile[0] + tR2GcO[elem, 0, chunk][0],
                        cute.size(mDHidden, mode=[0])
                    ) and cute.elem_less(
                        pidn * self.epi_tile[1] * self.num_epi_acc_stage
                        + tR2GcO[elem, 0, chunk][1],
                        cute.size(mDHidden, mode=[1])
                    )

            for n in cutlass.range(k2nd_valid_acc_stage):
                k2nd_mma_pipeline.consumer_wait(k2nd_mma_consumer_state)

                left_idx: cutlass.Int64 = n * self.epi_tile[1] + k2nd_left_idx
                right_idx: cutlass.Int64 = min(
                    (n + 1) * self.epi_tile[1] + k2nd_left_idx,
                    cute.size(mTransB, mode=[0])
                )
                valid_subtiles: cutlass.Int64 = cute.ceil_div(
                    (right_idx - left_idx),
                    cute.size(tTMEM_load_rO, mode=[0])
                )
                for n_subtile in cutlass.range(valid_subtiles):
                    cute.copy(
                        tiled_copy_t2r,
                        tTMEM_load_tO[(None, None, None, n_subtile, k2nd_mma_consumer_state.index)],
                        tTMEM_load_rO
                    )
                    tTMEM_load_rO_half.store(tTMEM_load_rO.load().to(tTMEM_load_rO_half.element_type))

                    for chunk in cutlass.range(cute.size(tR2GrDHidden, mode=[1]), unroll_full=True):
                        copy_id = n_subtile * cute.size(tR2GrDHidden, mode=[1]) + chunk
                        pred_id = n * self.num_epi_stage_per_tile * cute.size(tR2GrDHidden, mode=[1]) + copy_id
                        cute.copy(
                            tiled_copy_r2g,
                            tR2GrDHidden[(None, chunk, None, None)],
                            tR2GgDHidden[(None, None, copy_id, None, k2nd_mma_consumer_state.index)],
                            pred=cute.append_ones(tR2GcO_pred[((0, None), None, pred_id)])
                        )

                k2nd_mma_pipeline.consumer_release(k2nd_mma_consumer_state)
                k2nd_mma_consumer_state.advance()

        # ----- Deallocate TMEM ----- #
        self.cta_sync_barrier.arrive_and_wait()
        if warp_idx == self.load_ab_warp_ids:
            cute.arch.relinquish_tmem_alloc_permit()
            cute.arch.dealloc_tmem(
                tmem_ptr,
                self.tmem_alloc_cols,
                is_two_cta=self.use_2cta_instrs
            )
        


    @cute.jit
    def __call__(
        self,
        hidden: cute.Tensor,
        weight: cute.Tensor,
        labels: cute.Tensor,
        dlogprobs: cute.Tensor,
        maximum: cute.Tensor,
        accu: cute.Tensor,
        scalarNumValidTokens: cute.Pointer,
        dHidden: cute.Tensor,
        ignore_index: cutlass.Int64,
        rank: cutlass.Int32,
        stream: cuda.CUstream,
    ) -> None:
        a_dtype: Type[cutlass.Numeric] = hidden.element_type
        b_dtype: Type[cutlass.Numeric] = weight.element_type
        if cutlass.const_expr(hidden.element_type != weight.element_type):
            raise RuntimeError(f"data type don't match: {a_dtype} v.s. {b_dtype}")
        if cutlass.const_expr(a_dtype not in [cutlass.Float16, cutlass.BFloat16]):
            raise RuntimeError("hidden can only be FP16 or BF16")
        if cutlass.const_expr(hidden.layout.shape[1] != weight.layout.shape[1]):
            raise RuntimeError("K dimension doesn't match")
        if cutlass.const_expr(cute.rank(hidden) != 2):
            raise RuntimeError("hidden must be a 2D tensor with shape (batchsize * seqlen, dim)")
        if cutlass.const_expr(cute.rank(weight) != 2):
            raise RuntimeError("weight must be a 2D tensor with shape (vocabsize, dim)")
        if cutlass.const_expr(cute.rank(labels) != 1):
            raise RuntimeError("labels must be a 1D tensor with shape (batchsize * seqlen,)")
        if cutlass.const_expr(cute.rank(dlogprobs) != 1):
            raise RuntimeError("dlogprobs must be a 1D tensor with shape (batchsize * seqlen,) or (1,)")
        if cutlass.const_expr(cute.rank(maximum) != 1):
            raise RuntimeError("maximum must be a 1D tensor with shape (batchsize * seqlen,)")
        if cutlass.const_expr(cute.rank(accu) != 1):
            raise RuntimeError("accu must be a 1D tensor with shape (batchsize * seqlen,)")
        if cutlass.const_expr(cute.rank(dHidden) != 2):
            raise RuntimeError("dHidden must be a 2D tensor with shape (batchsize * seqlen, dim)")

        problem_mnk = (hidden.layout.shape[0],
                       weight.layout.shape[0],
                       hidden.layout.shape[1])
        if cutlass.const_expr((problem_mnk[2] * a_dtype.width // 8) % 16 != 0):
            raise RuntimeError(f"K dimension is not 16B aligned: {problem_mnk[2]}")
        if cutlass.const_expr((problem_mnk[2] * b_dtype.width // 8) % 128 != 0):
            raise RuntimeError(f"K dimension is not 128B aligned: {problem_mnk[2]}")

        a_major_mode = utils.LayoutEnum.from_tensor(hidden).mma_major_mode()
        b_major_mode = utils.LayoutEnum.from_tensor(weight).mma_major_mode()

        k1st_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            a_dtype,
            a_major_mode,
            b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.k1st_mma_tiler[:2]
        )

        self._setup_attributes(
            k1st_tiled_mma,
            a_dtype,
            b_dtype,
        )

        k2nd_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            b_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.MN,
            self.acc_dtype,
            self.cta_group,
            self.k2nd_mma_tiler[:2],
            tcgen05.OperandSource.TMEM
        )

        self.scale_tile = self.k1st_mma_tiler[:2]
        self.epi_tile = self.k2nd_mma_tiler[:2]
        
        # Swizzle o [(tileM, tileK), loopM, loopK, (stage)]
        a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            k1st_tiled_mma,
            self.k1st_mma_tiler,
            a_dtype,
            self.num_ab_stage
        )
        # Swizzle o [(tileN, tileK), loopN, loopK, (stage)]
        b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            k1st_tiled_mma,
            self.k1st_mma_tiler,
            b_dtype,
            self.num_ab_stage
        )

        # Swizzle o [(tileN, tileK), loopN, loopK, (stage)]
        convtert_a_tmem_layout_staged = sm100_utils.make_smem_layout_a(
            k2nd_tiled_mma,
            self.k2nd_mma_tiler,
            b_dtype,
            1
        )

        # Swizzle o [(tileN, tileK), loopN, loopK, (stage)]
        trans_b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            k2nd_tiled_mma,
            self.k2nd_mma_tiler,
            b_dtype,
            self.num_trans_b_stage
        )
        
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(self.cta_group)
        tma_store_op = cpasync.CopyBulkTensorTileS2GOp()

        # Swizzle o [(tileM, tileK), loopM, loopK]
        a_smem_layout = cute.select(
            a_smem_layout_staged,
            mode=[0, 1, 2]
        )
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            hidden,
            a_smem_layout,
            self.k1st_mma_tiler,
            k1st_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        # Swizzle o [(tileN, tileK), loopN, loopK]
        b_smem_layout = cute.select(
            b_smem_layout_staged,
            mode=[0, 1, 2]
        )
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            weight,
            b_smem_layout,
            self.k1st_mma_tiler,
            k1st_tiled_mma,
            self.cluster_layout_vmnk.shape
        )

        # Swizzle o [(tileN, tileK), loopN, loopK]
        trans_b_smem_layout = cute.select(
            trans_b_smem_layout_staged,
            mode=[0, 1, 2]
        )
        tma_atom_trans_b, tma_tensor_trans_b = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            cute.make_tensor(
                weight.iterator, 
                cute.make_layout((weight.layout.shape[1], weight.layout.shape[0]))),
            trans_b_smem_layout,
            self.k2nd_mma_tiler,
            k2nd_tiled_mma,
            self.cluster_layout_vmnk.shape
        )


        
        a_copy_size = cute.size_in_bytes(a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(b_dtype, b_smem_layout)
        self.tma_copy_ab_bytes = a_copy_size + b_copy_size

        trans_b_copy_size = cute.size_in_bytes(b_dtype, trans_b_smem_layout)
        self.tma_copy_trans_b_bytes = trans_b_copy_size

        @cute.struct
        class SharedStorage:
            """
            The shared storage for the dHidden backward kernel.
            """
            load_ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            k1st_mma_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]

            convert_acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 1 * 2]
            load_trans_b_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_trans_b_stage * 2]
            k2nd_mma_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_epi_acc_stage * 2]

            tmem_dealloc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 1]
            tmem_holding_buf: cutlass.Int32

            sA: cute.struct.Align[
                cute.struct.MemRange[a_dtype, cute.cosize(a_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[b_dtype, cute.cosize(b_smem_layout_staged)],
                self.buffer_align_bytes
            ]
            sTransB: cute.struct.Align[
                cute.struct.MemRange[b_dtype, cute.cosize(trans_b_smem_layout_staged)],
                self.buffer_align_bytes
            ]
        self.shared_storage = SharedStorage

        grid = self._compute_grid(
            problem_mnk = problem_mnk,
            cluster_shape_mn = self.cluster_shape_mn,
            cta_tiler = self.k2nd_mma_tiler,
            num_epi_acc_stage = self.num_epi_acc_stage,
        )

        self.kernel(
            k1st_tiled_mma,
            k2nd_tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_trans_b,
            tma_tensor_trans_b,
            labels,
            dlogprobs,
            maximum,
            accu,
            scalarNumValidTokens,
            dHidden,
            ignore_index,
            a_smem_layout_staged,
            b_smem_layout_staged,
            convtert_a_tmem_layout_staged,
            trans_b_smem_layout_staged,
            self.cluster_layout_vmnk,
            problem_mnk,
            rank,
        ).launch(
            grid=grid,
            block=[self.threads_per_warpgroup, self.warpgroups_per_cta, 1],
            cluster=self.cluster_shape_mnk,
            stream=stream
        )


if __name__ == "__main__":
    # python -m megatron.core.fusions.linear_cross_entropy.blackwell.bwd_two_kernels

    import torch
    from cutlass.cute.runtime import from_dlpack
    from ..utils import str_to_reduction_enum

    torch.manual_seed(1111)

    # batchsize = 1
    # seqlen = 7
    # vocabsize = 256
    # dim = 64

    batchsize = 4
    seqlen = 2035
    vocabsize = 152063
    dim = 4096
    dtype = torch.bfloat16
    reduction = "none"
    ignore_index = -100
    rank = 0

    hidden = (
        torch.empty((batchsize, seqlen, dim), dtype=dtype, device="cuda")
        .uniform_(-0.1, 0.1)
        .requires_grad_(True)
    )
    weight = (
        torch.empty((vocabsize, dim), dtype=dtype, device="cuda")
        .uniform_(-0.1, 0.1)
        .requires_grad_(True)
    )
    # hidden = torch.ones((batchsize, seqlen, dim), dtype=dtype, device="cuda")
    # weight = torch.ones((vocabsize, dim), dtype=dtype, device="cuda")
    labels = torch.randint(0, vocabsize, (batchsize, seqlen), dtype=torch.long, device="cuda")

    num_valid_tokens = torch.sum(labels != ignore_index)

    dlogprobs = None
    if reduction == "none":
        dlogprobs = torch.randn((batchsize, seqlen), dtype=torch.float32, device="cuda")
    elif reduction == "sum":
        dlogprobs = torch.randn(1, dtype=torch.float32, device="cuda")
    elif reduction == "mean":
        dlogprobs = torch.randn(1, dtype=torch.float32, device="cuda")
    

    def obtain_accumulate_and_maximum(
        hidden: torch.Tensor,
        weight: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = hidden.to(torch.float32) @ weight.to(torch.float32).T
        maximum, _ = torch.max(logits, dim=-1, keepdim=False)
        accumulate = torch.sum(torch.exp(logits - maximum.unsqueeze(-1)), dim=-1, keepdim=False)
        return maximum.view(-1), accumulate.view(-1)

    maximum, accumulate = obtain_accumulate_and_maximum(hidden, weight)

    dHidden = torch.empty_like(hidden)

    hidden_packed = from_dlpack(
        hidden.view(-1, dim).detach(),
        assumed_align=128
    ).mark_compact_shape_dynamic(mode=0)
    weight_packed = from_dlpack(
        weight.detach(),
        assumed_align=128
    )
    labels_packed = from_dlpack(
        labels.view(-1).detach(),
        assumed_align=8
    ).mark_compact_shape_dynamic(mode=0)

    maximum_packed = from_dlpack(
        maximum.detach(),
        assumed_align=4
    ).mark_compact_shape_dynamic(mode=0)
    accumulate_packed = from_dlpack(
        accumulate.detach(),
        assumed_align=4
    ).mark_compact_shape_dynamic(mode=0)
    scalarNumValidTokens_packed = cute.runtime.make_ptr(
        cutlass.Int64,
        num_valid_tokens.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=8
    )

    dlogprobs_packed = from_dlpack(
        dlogprobs.view(-1).detach(),
        assumed_align=4
    ).mark_compact_shape_dynamic(mode=0)

    dHidden_packed = from_dlpack(
        dHidden.view(-1, dim).detach(),
        assumed_align=128
    ).mark_compact_shape_dynamic(mode=0)

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    dHidden_kernel = BwdTwoKernelsGradHidden(
        reduction=str_to_reduction_enum(reduction),
    )

    dHidden_kernel_compiled = cute.compile(
        dHidden_kernel,
        hidden_packed,
        weight_packed,
        labels_packed,
        dlogprobs_packed,
        maximum_packed,
        accumulate_packed,
        scalarNumValidTokens_packed,
        dHidden_packed,
        ignore_index,
        rank,
        stream,
        # options="--generate-line-info"
    )

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)

    start.record(stream=torch.cuda.current_stream())
    dHidden_kernel_compiled(
        hidden_packed,
        weight_packed,
        labels_packed,
        dlogprobs_packed,
        maximum_packed,
        accumulate_packed,
        scalarNumValidTokens_packed,
        dHidden_packed,
        ignore_index,
        rank,
        stream
    )
    stop.record(stream=torch.cuda.current_stream())
    torch.cuda.synchronize()
    elapsed_time = start.elapsed_time(stop)
    print(f"[INFO]: Kernel elapsed time: {elapsed_time:.4f} ms")

    def torch_backward(
        hidden: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor,
        dlogprobs: torch.Tensor,
        reduction: str,
        num_valid_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = hidden.to(torch.float32) @ weight.to(torch.float32).T
        logits_view = logits.view(-1, weight.shape[0])
        one_hot = torch.zeros_like(logits_view)
        one_hot.scatter_(1, labels.view(-1).unsqueeze(-1), 1)
        pd = torch.nn.functional.softmax(logits_view, dim=-1)
        d_logits = (pd - one_hot)
        if reduction in ["none", "sum"]:
            d_logits *= dlogprobs.view(-1).unsqueeze(-1)
        elif reduction == "mean":
            d_logits *= (dlogprobs.view(-1).unsqueeze(-1) / num_valid_tokens.to(d_logits.dtype))
        d_logits = d_logits.to(hidden.dtype)

        d_hidden = d_logits @ weight
        d_weight = d_logits.T @ hidden.view(-1, dim)
        return d_hidden.view(hidden.shape), d_weight.view(weight.shape)

    start.record(stream=torch.cuda.current_stream())
    torch_d_hidden, torch_d_weight = torch_backward(hidden, weight, labels, dlogprobs, reduction, num_valid_tokens)
    stop.record(stream=torch.cuda.current_stream())
    torch.cuda.synchronize()
    elapsed_time = start.elapsed_time(stop)
    print(f"[INFO]: Torch backward elapsed time: {elapsed_time:.4f} ms")
    # print("torch_d_hidden:\n", torch_d_hidden)
    # print("kernel_d_hidden:\n", dHidden)
    torch.testing.assert_close(dHidden, torch_d_hidden)
    print("[PASSED] dHidden is close to that of PyTorch Operation")

    def torch_native_backward(
        hidden: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor,
        dlogprobs: torch.Tensor,
        reduction: str,
        start: torch.cuda.Event,
        stop: torch.cuda.Event,
    ):
        start.record(stream=torch.cuda.current_stream())
        logits = hidden.to(torch.float32) @ weight.to(torch.float32).T
        logits_view = logits.view(-1, weight.shape[0])
        ce = torch.nn.functional.cross_entropy(logits_view, labels.view(-1), reduction=reduction)
        stop.record(stream=torch.cuda.current_stream())
        torch.cuda.synchronize()

        elapsed_time = start.elapsed_time(stop)
        print(f"[INFO]: Torch native forward elapsed time: {elapsed_time:.4f} ms")

        start.record(stream=torch.cuda.current_stream())
        d_hidden, d_weight = torch.autograd.grad((ce,), (hidden, weight), (dlogprobs.view(ce.shape),), retain_graph=False)
        stop.record(stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(stop)
        print(f"[INFO]: Torch native backward elapsed time: {elapsed_time:.4f} ms")

        return d_hidden.view(hidden.shape), d_weight.view(weight.shape)

    d_hidden_native, d_weight_native = torch_native_backward(hidden, weight, labels, dlogprobs, reduction, start, stop)
    torch.testing.assert_close(dHidden, d_hidden_native)
    print("[PASSED] dHidden is close to that of PyTorch Native Operation")