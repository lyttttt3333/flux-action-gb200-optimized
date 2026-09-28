# Arithmetic extracted from pinned Inductor Norm/RoPE kernels; explicit launch configs.
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

@triton.jit
def base_early(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 65280
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 24)
    x1 = xindex // 24
    _tmp14 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x5 = xindex
    _tmp27 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = 3072 + r0_2 + 128*x0
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1, 1], 6144, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + (6144*x1 + (3072 + r0_2 + 128*x0)), r0_mask & tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp6 = tmp0 >= tmp3
        tmp7 = tl.full([1, 1], 27648, tl.int64)
        tmp8 = tmp0 < tmp7
        tmp9 = tl.load(in_ptr1 + (21504*x1 + ((-3072) + r0_2 + 128*x0)), r0_mask & tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tl.where(tmp4, tmp5, tmp9)
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tmp11 * tmp11
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(r0_mask & xmask, tmp15, _tmp14)
        tmp16 = r0_2 + 128*x0
        tmp17 = tmp16 >= tmp1
        tmp18 = tmp16 < tmp3
        tmp19 = tl.load(in_ptr0 + (6144*x1 + (r0_2 + 128*x0)), r0_mask & tmp18 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp20 = tmp16 >= tmp3
        tmp21 = tmp16 < tmp7
        tmp22 = tl.load(in_ptr1 + (21504*x1 + ((-6144) + r0_2 + 128*x0)), r0_mask & tmp20 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp23 = tl.where(tmp18, tmp19, tmp22)
        tmp24 = tmp23.to(tl.float32)
        tmp25 = tmp24 * tmp24
        tmp26 = tl.broadcast_to(tmp25, [XBLOCK, R0_BLOCK])
        tmp28 = _tmp27 + tmp26
        _tmp27 = tl.where(r0_mask & xmask, tmp28, _tmp27)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tmp27 = tl.sum(_tmp27, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        r0_4 = r0_index // 2
        tmp29 = tl.load(in_ptr2 + (20480 + 2*r0_2 + 256*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp49 = tl.load(in_ptr3 + (2*r0_4), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp53 = tl.load(in_ptr2 + (20481 + 2*r0_2 + 256*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp65 = tl.load(in_ptr3 + (1 + 2*r0_4), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp84 = tl.load(in_ptr4 + (2*r0_4), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp99 = tl.load(in_ptr4 + (1 + 2*r0_4), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp30 = 2*r0_4 + 128*x0
        tmp31 = tl.full([1, 1], 0, tl.int64)
        tmp32 = tmp30 >= tmp31
        tmp33 = tl.full([1, 1], 6144, tl.int64)
        tmp34 = tmp30 < tmp33
        tmp35 = tl.load(in_ptr0 + (6144*x1 + (2*r0_4 + 128*x0)), r0_mask & tmp34 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp36 = tmp30 >= tmp33
        tmp37 = tl.full([1, 1], 27648, tl.int64)
        tmp38 = tmp30 < tmp37
        tmp39 = tl.load(in_ptr1 + (21504*x1 + ((-6144) + 2*r0_4 + 128*x0)), r0_mask & tmp36 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp40 = tl.where(tmp34, tmp35, tmp39)
        tmp41 = tmp40.to(tl.float32)
        tmp42 = tl.full([1, 1], 128.0, tl.float32)
        tmp43 = (tmp27 / tmp42)
        tmp44 = tl.full([1, 1], 1e-06, tl.float32)
        tmp45 = tmp43 + tmp44
        tmp46 = libdevice.rsqrt(tmp45)
        tmp47 = tmp41 * tmp46
        tmp48 = tmp47.to(tl.float32)
        tmp50 = tmp48 * tmp49
        tmp51 = tmp50.to(tl.float32)
        tmp52 = tmp29 * tmp51
        tmp54 = 1 + 2*r0_4 + 128*x0
        tmp55 = tmp54 >= tmp31
        tmp56 = tmp54 < tmp33
        tmp57 = tl.load(in_ptr0 + (6144*x1 + (1 + 2*r0_4 + 128*x0)), r0_mask & tmp56 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp58 = tmp54 >= tmp33
        tmp59 = tmp54 < tmp37
        tmp60 = tl.load(in_ptr1 + (21504*x1 + ((-6143) + 2*r0_4 + 128*x0)), r0_mask & tmp58 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp61 = tl.where(tmp56, tmp57, tmp60)
        tmp62 = tmp61.to(tl.float32)
        tmp63 = tmp62 * tmp46
        tmp64 = tmp63.to(tl.float32)
        tmp66 = tmp64 * tmp65
        tmp67 = tmp66.to(tl.float32)
        tmp68 = tmp53 * tmp67
        tmp69 = tmp52 + tmp68
        tmp70 = 3072 + 2*r0_4 + 128*x0
        tmp71 = tmp70 >= tmp31
        tmp72 = tmp70 < tmp33
        tmp73 = tl.load(in_ptr0 + (6144*x1 + (3072 + 2*r0_4 + 128*x0)), r0_mask & tmp72 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp74 = tmp70 >= tmp33
        tmp75 = tmp70 < tmp37
        tmp76 = tl.load(in_ptr1 + (21504*x1 + ((-3072) + 2*r0_4 + 128*x0)), r0_mask & tmp74 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp77 = tl.where(tmp72, tmp73, tmp76)
        tmp78 = tmp77.to(tl.float32)
        tmp79 = (tmp14 / tmp42)
        tmp80 = tmp79 + tmp44
        tmp81 = libdevice.rsqrt(tmp80)
        tmp82 = tmp78 * tmp81
        tmp83 = tmp82.to(tl.float32)
        tmp85 = tmp83 * tmp84
        tmp86 = tmp85.to(tl.float32)
        tmp87 = tmp29 * tmp86
        tmp88 = 3073 + 2*r0_4 + 128*x0
        tmp89 = tmp88 >= tmp31
        tmp90 = tmp88 < tmp33
        tmp91 = tl.load(in_ptr0 + (6144*x1 + (3073 + 2*r0_4 + 128*x0)), r0_mask & tmp90 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp92 = tmp88 >= tmp33
        tmp93 = tmp88 < tmp37
        tmp94 = tl.load(in_ptr1 + (21504*x1 + ((-3071) + 2*r0_4 + 128*x0)), r0_mask & tmp92 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp95 = tl.where(tmp90, tmp91, tmp94)
        tmp96 = tmp95.to(tl.float32)
        tmp97 = tmp96 * tmp81
        tmp98 = tmp97.to(tl.float32)
        tmp100 = tmp98 * tmp99
        tmp101 = tmp100.to(tl.float32)
        tmp102 = tmp53 * tmp101
        tmp103 = tmp87 + tmp102
        tl.store(out_ptr2 + (r0_2 + 128*x1 + 348160*x0), tmp69, r0_mask & xmask)
        tl.store(out_ptr3 + (r0_2 + 128*x1 + 348160*x0), tmp103, r0_mask & xmask)


@triton.jit
def base_joint(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 152304
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 24)
    x1 = xindex // 24
    _tmp14 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x7 = xindex
    _tmp27 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = 3072 + r0_2 + 128*x0
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1, 1], 6144, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + (6144*x1 + (3072 + r0_2 + 128*x0)), r0_mask & tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp6 = tmp0 >= tmp3
        tmp7 = tl.full([1, 1], 27648, tl.int64)
        tmp8 = tmp0 < tmp7
        tmp9 = tl.load(in_ptr1 + (21504*x1 + ((-3072) + r0_2 + 128*x0)), r0_mask & tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tl.where(tmp4, tmp5, tmp9)
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tmp11 * tmp11
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(r0_mask & xmask, tmp15, _tmp14)
        tmp16 = r0_2 + 128*x0
        tmp17 = tmp16 >= tmp1
        tmp18 = tmp16 < tmp3
        tmp19 = tl.load(in_ptr0 + (6144*x1 + (r0_2 + 128*x0)), r0_mask & tmp18 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp20 = tmp16 >= tmp3
        tmp21 = tmp16 < tmp7
        tmp22 = tl.load(in_ptr1 + (21504*x1 + ((-6144) + r0_2 + 128*x0)), r0_mask & tmp20 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp23 = tl.where(tmp18, tmp19, tmp22)
        tmp24 = tmp23.to(tl.float32)
        tmp25 = tmp24 * tmp24
        tmp26 = tl.broadcast_to(tmp25, [XBLOCK, R0_BLOCK])
        tmp28 = _tmp27 + tmp26
        _tmp27 = tl.where(r0_mask & xmask, tmp28, _tmp27)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tmp27 = tl.sum(_tmp27, 1)[:, None]
    x3 = ((xindex // 24) % 3173)
    x4 = xindex // 76152
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        r0_6 = r0_index // 2
        tmp29 = tl.load(in_ptr2 + (2*r0_2 + 256*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp49 = tl.load(in_ptr3 + (2*r0_6), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp53 = tl.load(in_ptr2 + (1 + 2*r0_2 + 256*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp65 = tl.load(in_ptr3 + (1 + 2*r0_6), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp84 = tl.load(in_ptr4 + (2*r0_6), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp99 = tl.load(in_ptr4 + (1 + 2*r0_6), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp30 = 2*r0_6 + 128*x0
        tmp31 = tl.full([1, 1], 0, tl.int64)
        tmp32 = tmp30 >= tmp31
        tmp33 = tl.full([1, 1], 6144, tl.int64)
        tmp34 = tmp30 < tmp33
        tmp35 = tl.load(in_ptr0 + (6144*x1 + (2*r0_6 + 128*x0)), r0_mask & tmp34 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp36 = tmp30 >= tmp33
        tmp37 = tl.full([1, 1], 27648, tl.int64)
        tmp38 = tmp30 < tmp37
        tmp39 = tl.load(in_ptr1 + (21504*x1 + ((-6144) + 2*r0_6 + 128*x0)), r0_mask & tmp36 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp40 = tl.where(tmp34, tmp35, tmp39)
        tmp41 = tmp40.to(tl.float32)
        tmp42 = tl.full([1, 1], 128.0, tl.float32)
        tmp43 = (tmp27 / tmp42)
        tmp44 = tl.full([1, 1], 1e-06, tl.float32)
        tmp45 = tmp43 + tmp44
        tmp46 = libdevice.rsqrt(tmp45)
        tmp47 = tmp41 * tmp46
        tmp48 = tmp47.to(tl.float32)
        tmp50 = tmp48 * tmp49
        tmp51 = tmp50.to(tl.float32)
        tmp52 = tmp29 * tmp51
        tmp54 = 1 + 2*r0_6 + 128*x0
        tmp55 = tmp54 >= tmp31
        tmp56 = tmp54 < tmp33
        tmp57 = tl.load(in_ptr0 + (6144*x1 + (1 + 2*r0_6 + 128*x0)), r0_mask & tmp56 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp58 = tmp54 >= tmp33
        tmp59 = tmp54 < tmp37
        tmp60 = tl.load(in_ptr1 + (21504*x1 + ((-6143) + 2*r0_6 + 128*x0)), r0_mask & tmp58 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp61 = tl.where(tmp56, tmp57, tmp60)
        tmp62 = tmp61.to(tl.float32)
        tmp63 = tmp62 * tmp46
        tmp64 = tmp63.to(tl.float32)
        tmp66 = tmp64 * tmp65
        tmp67 = tmp66.to(tl.float32)
        tmp68 = tmp53 * tmp67
        tmp69 = tmp52 + tmp68
        tmp70 = 3072 + 2*r0_6 + 128*x0
        tmp71 = tmp70 >= tmp31
        tmp72 = tmp70 < tmp33
        tmp73 = tl.load(in_ptr0 + (6144*x1 + (3072 + 2*r0_6 + 128*x0)), r0_mask & tmp72 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp74 = tmp70 >= tmp33
        tmp75 = tmp70 < tmp37
        tmp76 = tl.load(in_ptr1 + (21504*x1 + ((-3072) + 2*r0_6 + 128*x0)), r0_mask & tmp74 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp77 = tl.where(tmp72, tmp73, tmp76)
        tmp78 = tmp77.to(tl.float32)
        tmp79 = (tmp14 / tmp42)
        tmp80 = tmp79 + tmp44
        tmp81 = libdevice.rsqrt(tmp80)
        tmp82 = tmp78 * tmp81
        tmp83 = tmp82.to(tl.float32)
        tmp85 = tmp83 * tmp84
        tmp86 = tmp85.to(tl.float32)
        tmp87 = tmp29 * tmp86
        tmp88 = 3073 + 2*r0_6 + 128*x0
        tmp89 = tmp88 >= tmp31
        tmp90 = tmp88 < tmp33
        tmp91 = tl.load(in_ptr0 + (6144*x1 + (3073 + 2*r0_6 + 128*x0)), r0_mask & tmp90 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp92 = tmp88 >= tmp33
        tmp93 = tmp88 < tmp37
        tmp94 = tl.load(in_ptr1 + (21504*x1 + ((-3071) + 2*r0_6 + 128*x0)), r0_mask & tmp92 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp95 = tl.where(tmp90, tmp91, tmp94)
        tmp96 = tmp95.to(tl.float32)
        tmp97 = tmp96 * tmp81
        tmp98 = tmp97.to(tl.float32)
        tmp100 = tmp98 * tmp99
        tmp101 = tmp100.to(tl.float32)
        tmp102 = tmp53 * tmp101
        tmp103 = tmp87 + tmp102
        tl.store(out_ptr2 + (r0_2 + 128*x3 + 406144*x0 + 9747456*x4), tmp69, r0_mask & xmask)
        tl.store(out_ptr3 + (r0_2 + 128*x3 + 406144*x0 + 9747456*x4), tmp103, r0_mask & xmask)


@triton.jit
def opt_early(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 65280
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 24)
    x1 = xindex // 24
    _tmp14 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x5 = xindex
    _tmp27 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = 3072 + r0_2 + 128*x0
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1, 1], 6144, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + (6144*x1 + (3072 + r0_2 + 128*x0)), r0_mask & tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp6 = tmp0 >= tmp3
        tmp7 = tl.full([1, 1], 27648, tl.int64)
        tmp8 = tmp0 < tmp7
        tmp9 = tl.load(in_ptr1 + (21504*x1 + ((-3072) + r0_2 + 128*x0)), r0_mask & tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tl.where(tmp4, tmp5, tmp9)
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tmp11 * tmp11
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(r0_mask & xmask, tmp15, _tmp14)
        tmp16 = r0_2 + 128*x0
        tmp17 = tmp16 >= tmp1
        tmp18 = tmp16 < tmp3
        tmp19 = tl.load(in_ptr0 + (6144*x1 + (r0_2 + 128*x0)), r0_mask & tmp18 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp20 = tmp16 >= tmp3
        tmp21 = tmp16 < tmp7
        tmp22 = tl.load(in_ptr1 + (21504*x1 + ((-6144) + r0_2 + 128*x0)), r0_mask & tmp20 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp23 = tl.where(tmp18, tmp19, tmp22)
        tmp24 = tmp23.to(tl.float32)
        tmp25 = tmp24 * tmp24
        tmp26 = tl.broadcast_to(tmp25, [XBLOCK, R0_BLOCK])
        tmp28 = _tmp27 + tmp26
        _tmp27 = tl.where(r0_mask & xmask, tmp28, _tmp27)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tmp27 = tl.sum(_tmp27, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        r0_4 = r0_index // 2
        tmp29 = tl.load(in_ptr2 + (20480 + 2*r0_2 + 256*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp49 = tl.load(in_ptr3 + (2*r0_4), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp53 = tl.load(in_ptr2 + (20481 + 2*r0_2 + 256*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp65 = tl.load(in_ptr3 + (1 + 2*r0_4), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp84 = tl.load(in_ptr4 + (2*r0_4), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp99 = tl.load(in_ptr4 + (1 + 2*r0_4), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp30 = 2*r0_4 + 128*x0
        tmp31 = tl.full([1, 1], 0, tl.int64)
        tmp32 = tmp30 >= tmp31
        tmp33 = tl.full([1, 1], 6144, tl.int64)
        tmp34 = tmp30 < tmp33
        tmp35 = tl.load(in_ptr0 + (6144*x1 + (2*r0_4 + 128*x0)), r0_mask & tmp34 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp36 = tmp30 >= tmp33
        tmp37 = tl.full([1, 1], 27648, tl.int64)
        tmp38 = tmp30 < tmp37
        tmp39 = tl.load(in_ptr1 + (21504*x1 + ((-6144) + 2*r0_4 + 128*x0)), r0_mask & tmp36 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp40 = tl.where(tmp34, tmp35, tmp39)
        tmp41 = tmp40.to(tl.float32)
        tmp42 = tl.full([1, 1], 128.0, tl.float32)
        tmp43 = (tmp27 / tmp42)
        tmp44 = tl.full([1, 1], 1e-06, tl.float32)
        tmp45 = tmp43 + tmp44
        tmp46 = libdevice.rsqrt(tmp45)
        tmp47 = tmp41 * tmp46
        tmp48 = tmp47.to(tl.float32)
        tmp50 = tmp48 * tmp49
        tmp51 = tmp50.to(tl.float32)
        tmp52 = tmp29 * tmp51
        tmp54 = 1 + 2*r0_4 + 128*x0
        tmp55 = tmp54 >= tmp31
        tmp56 = tmp54 < tmp33
        tmp57 = tl.load(in_ptr0 + (6144*x1 + (1 + 2*r0_4 + 128*x0)), r0_mask & tmp56 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp58 = tmp54 >= tmp33
        tmp59 = tmp54 < tmp37
        tmp60 = tl.load(in_ptr1 + (21504*x1 + ((-6143) + 2*r0_4 + 128*x0)), r0_mask & tmp58 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp61 = tl.where(tmp56, tmp57, tmp60)
        tmp62 = tmp61.to(tl.float32)
        tmp63 = tmp62 * tmp46
        tmp64 = tmp63.to(tl.float32)
        tmp66 = tmp64 * tmp65
        tmp67 = tmp66.to(tl.float32)
        tmp68 = tmp53 * tmp67
        tmp69 = tmp52 + tmp68
        tmp70 = 3072 + 2*r0_4 + 128*x0
        tmp71 = tmp70 >= tmp31
        tmp72 = tmp70 < tmp33
        tmp73 = tl.load(in_ptr0 + (6144*x1 + (3072 + 2*r0_4 + 128*x0)), r0_mask & tmp72 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp74 = tmp70 >= tmp33
        tmp75 = tmp70 < tmp37
        tmp76 = tl.load(in_ptr1 + (21504*x1 + ((-3072) + 2*r0_4 + 128*x0)), r0_mask & tmp74 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp77 = tl.where(tmp72, tmp73, tmp76)
        tmp78 = tmp77.to(tl.float32)
        tmp79 = (tmp14 / tmp42)
        tmp80 = tmp79 + tmp44
        tmp81 = libdevice.rsqrt(tmp80)
        tmp82 = tmp78 * tmp81
        tmp83 = tmp82.to(tl.float32)
        tmp85 = tmp83 * tmp84
        tmp86 = tmp85.to(tl.float32)
        tmp87 = tmp29 * tmp86
        tmp88 = 3073 + 2*r0_4 + 128*x0
        tmp89 = tmp88 >= tmp31
        tmp90 = tmp88 < tmp33
        tmp91 = tl.load(in_ptr0 + (6144*x1 + (3073 + 2*r0_4 + 128*x0)), r0_mask & tmp90 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp92 = tmp88 >= tmp33
        tmp93 = tmp88 < tmp37
        tmp94 = tl.load(in_ptr1 + (21504*x1 + ((-3071) + 2*r0_4 + 128*x0)), r0_mask & tmp92 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp95 = tl.where(tmp90, tmp91, tmp94)
        tmp96 = tmp95.to(tl.float32)
        tmp97 = tmp96 * tmp81
        tmp98 = tmp97.to(tl.float32)
        tmp100 = tmp98 * tmp99
        tmp101 = tmp100.to(tl.float32)
        tmp102 = tmp53 * tmp101
        tmp103 = tmp87 + tmp102
        tl.store(out_ptr2 + (r0_2 + 128*x0 + 3072*x1), tmp69, r0_mask & xmask)
        tl.store(out_ptr3 + (r0_2 + 128*x0 + 3072*x1), tmp103, r0_mask & xmask)


@triton.jit
def opt_joint(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 152304
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 24)
    x1 = xindex // 24
    _tmp14 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x7 = xindex
    _tmp27 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = 3072 + r0_2 + 128*x0
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1, 1], 6144, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + (6144*x1 + (3072 + r0_2 + 128*x0)), r0_mask & tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp6 = tmp0 >= tmp3
        tmp7 = tl.full([1, 1], 27648, tl.int64)
        tmp8 = tmp0 < tmp7
        tmp9 = tl.load(in_ptr1 + (21504*x1 + ((-3072) + r0_2 + 128*x0)), r0_mask & tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tl.where(tmp4, tmp5, tmp9)
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tmp11 * tmp11
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(r0_mask & xmask, tmp15, _tmp14)
        tmp16 = r0_2 + 128*x0
        tmp17 = tmp16 >= tmp1
        tmp18 = tmp16 < tmp3
        tmp19 = tl.load(in_ptr0 + (6144*x1 + (r0_2 + 128*x0)), r0_mask & tmp18 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp20 = tmp16 >= tmp3
        tmp21 = tmp16 < tmp7
        tmp22 = tl.load(in_ptr1 + (21504*x1 + ((-6144) + r0_2 + 128*x0)), r0_mask & tmp20 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp23 = tl.where(tmp18, tmp19, tmp22)
        tmp24 = tmp23.to(tl.float32)
        tmp25 = tmp24 * tmp24
        tmp26 = tl.broadcast_to(tmp25, [XBLOCK, R0_BLOCK])
        tmp28 = _tmp27 + tmp26
        _tmp27 = tl.where(r0_mask & xmask, tmp28, _tmp27)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tmp27 = tl.sum(_tmp27, 1)[:, None]
    x3 = ((xindex // 24) % 3173)
    x4 = xindex // 76152
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        r0_6 = r0_index // 2
        tmp29 = tl.load(in_ptr2 + (2*r0_2 + 256*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp49 = tl.load(in_ptr3 + (2*r0_6), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp53 = tl.load(in_ptr2 + (1 + 2*r0_2 + 256*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp65 = tl.load(in_ptr3 + (1 + 2*r0_6), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp84 = tl.load(in_ptr4 + (2*r0_6), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp99 = tl.load(in_ptr4 + (1 + 2*r0_6), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp30 = 2*r0_6 + 128*x0
        tmp31 = tl.full([1, 1], 0, tl.int64)
        tmp32 = tmp30 >= tmp31
        tmp33 = tl.full([1, 1], 6144, tl.int64)
        tmp34 = tmp30 < tmp33
        tmp35 = tl.load(in_ptr0 + (6144*x1 + (2*r0_6 + 128*x0)), r0_mask & tmp34 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp36 = tmp30 >= tmp33
        tmp37 = tl.full([1, 1], 27648, tl.int64)
        tmp38 = tmp30 < tmp37
        tmp39 = tl.load(in_ptr1 + (21504*x1 + ((-6144) + 2*r0_6 + 128*x0)), r0_mask & tmp36 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp40 = tl.where(tmp34, tmp35, tmp39)
        tmp41 = tmp40.to(tl.float32)
        tmp42 = tl.full([1, 1], 128.0, tl.float32)
        tmp43 = (tmp27 / tmp42)
        tmp44 = tl.full([1, 1], 1e-06, tl.float32)
        tmp45 = tmp43 + tmp44
        tmp46 = libdevice.rsqrt(tmp45)
        tmp47 = tmp41 * tmp46
        tmp48 = tmp47.to(tl.float32)
        tmp50 = tmp48 * tmp49
        tmp51 = tmp50.to(tl.float32)
        tmp52 = tmp29 * tmp51
        tmp54 = 1 + 2*r0_6 + 128*x0
        tmp55 = tmp54 >= tmp31
        tmp56 = tmp54 < tmp33
        tmp57 = tl.load(in_ptr0 + (6144*x1 + (1 + 2*r0_6 + 128*x0)), r0_mask & tmp56 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp58 = tmp54 >= tmp33
        tmp59 = tmp54 < tmp37
        tmp60 = tl.load(in_ptr1 + (21504*x1 + ((-6143) + 2*r0_6 + 128*x0)), r0_mask & tmp58 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp61 = tl.where(tmp56, tmp57, tmp60)
        tmp62 = tmp61.to(tl.float32)
        tmp63 = tmp62 * tmp46
        tmp64 = tmp63.to(tl.float32)
        tmp66 = tmp64 * tmp65
        tmp67 = tmp66.to(tl.float32)
        tmp68 = tmp53 * tmp67
        tmp69 = tmp52 + tmp68
        tmp70 = 3072 + 2*r0_6 + 128*x0
        tmp71 = tmp70 >= tmp31
        tmp72 = tmp70 < tmp33
        tmp73 = tl.load(in_ptr0 + (6144*x1 + (3072 + 2*r0_6 + 128*x0)), r0_mask & tmp72 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp74 = tmp70 >= tmp33
        tmp75 = tmp70 < tmp37
        tmp76 = tl.load(in_ptr1 + (21504*x1 + ((-3072) + 2*r0_6 + 128*x0)), r0_mask & tmp74 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp77 = tl.where(tmp72, tmp73, tmp76)
        tmp78 = tmp77.to(tl.float32)
        tmp79 = (tmp14 / tmp42)
        tmp80 = tmp79 + tmp44
        tmp81 = libdevice.rsqrt(tmp80)
        tmp82 = tmp78 * tmp81
        tmp83 = tmp82.to(tl.float32)
        tmp85 = tmp83 * tmp84
        tmp86 = tmp85.to(tl.float32)
        tmp87 = tmp29 * tmp86
        tmp88 = 3073 + 2*r0_6 + 128*x0
        tmp89 = tmp88 >= tmp31
        tmp90 = tmp88 < tmp33
        tmp91 = tl.load(in_ptr0 + (6144*x1 + (3073 + 2*r0_6 + 128*x0)), r0_mask & tmp90 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp92 = tmp88 >= tmp33
        tmp93 = tmp88 < tmp37
        tmp94 = tl.load(in_ptr1 + (21504*x1 + ((-3071) + 2*r0_6 + 128*x0)), r0_mask & tmp92 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp95 = tl.where(tmp90, tmp91, tmp94)
        tmp96 = tmp95.to(tl.float32)
        tmp97 = tmp96 * tmp81
        tmp98 = tmp97.to(tl.float32)
        tmp100 = tmp98 * tmp99
        tmp101 = tmp100.to(tl.float32)
        tmp102 = tmp53 * tmp101
        tmp103 = tmp87 + tmp102
        tl.store(out_ptr2 + (r0_2 + 128*x0 + 3072*x1), tmp69, r0_mask & xmask)
        tl.store(out_ptr3 + (r0_2 + 128*x0 + 3072*x1), tmp103, r0_mask & xmask)


