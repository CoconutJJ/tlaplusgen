/**
  This code is based on:
  https://github.com/pyxis-roc/gpucode-analyzer/blob/sass2c/gpucodeanalyzer/isa/sass/sass_insns.h
*/

#pragma once
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

static inline float TRUNCf (float x)
{
        float i, f;
        f = modff (x, &i);
        return i;
}

static inline float NTZf (float x)
{
        if (signbit (x))
                return 0.0;

        return x;
}

static inline float FTZf (float x)
{
        if (fpclassify (x) == FP_SUBNORMAL)
                return copysignf (0.0, x);

        return x;
}

static float AS_FLOAT (uint32_t r)
{
        union {
                float f;
                uint32_t i;
        } x = {.i = r};

        return x.f;
}

static uint32_t FROM_FLOAT (float f)
{
        union {
                float f;
                uint32_t i;
        } x = {.f = f};

        return x.i;
}

#define S2R(dst, src)  dst = src
#define S2UR(dst, src) dst = src

#define IABS(dst, src) dst = abs (src)

// TODO: RP
#define I2F_RP(dst, src) dst = FROM_FLOAT ((float)src)

// TODO
#define MUFU_RCP(dst, src) dst = FROM_FLOAT ((1.0 / AS_FLOAT (src)))

#define IMAD_U32(dst, src1, src2, src3) dst = src1 * src2 + src3
#define IMAD(dst, src1, src2, src3)     dst = src1 * src2 + src3
#define UIMAD(dst, src1, src2, src3)    IMAD (dst, src1, src2, src3)

#define IMAD_MOV_U32(dst, src1, src2, src3)  IMAD (dst, src1, src2, src3)
#define IMAD_IADD_U32(dst, src1, src2, src3) IMAD (dst, src1, src2, src3)
#define IMAD_SHL_U32(dst, src1, src2, src3)  IMAD (dst, src1, src2, src3)
#define IMAD_MOV(dst, src1, src2, src3)      IMAD (dst, src1, src2, src3)
#define IMAD_IADD(dst, src1, src2, src3)     IMAD (dst, src1, src2, src3)

// TODO: nvbit doesn't seem to capture the right value
// see also Incomprehensible IMAD post on the dev forums
#define IMAD_HI_U32(dst, src1, src2, src3) dst = (((uint64_t)src1 * src2) + ((uint64_t)dst << 32 | src3)) >> 32

// TODO: FTZ, TRUNC, and NTZ
#define F2I_FTZ_U32_TRUNC_NTZ(dst, src) dst = (uint32_t)NTZf (TRUNCf (FTZf (AS_FLOAT (src))))

#define IADD3(dst, src1, src2, src3) dst = src1 + src2 + src3

#define EXIT()     goto label_exit
#define BRA(label) goto label
#define BRA_PRED(pred, label)                                                                                          \
        if (pred)                                                                                                      \
        goto label
#define CALL_REL_NOINC(label) goto label

#define ISETP_GE_OR_D0(dst0, dst1, src1, src2, src3) dst0 = ((int32_t)src1 >= (int32_t)src2) || src3

#define ISETP_GT_AND_D0(dst0, dst1, src1, src2, src3) dst0 = ((int32_t)src1 > (int32_t)src2) && src3
#define ISETP_GT_AND_D1(dst0, dst1, src1, src2, src3) dst1 = !dst0

#define ISETP_LT_AND_D0(dst0, dst1, src1, src2, src3) dst0 = ((int32_t)src1 < (int32_t)src2) && src3
#define ISETP_LT_AND_D1(dst0, dst1, src1, src2, src3) dst1 = !dst0

#define ISETP_GE_AND_D0(dst0, dst1, src1, src2, src3) dst0 = ((int32_t)src1 >= (int32_t)src2) && src3
#define ISETP_GE_AND_D1(dst0, dst1, src1, src2, src3) dst1 = !dst0

#define ISETP_GE_U32_AND_D0(dst0, dst1, src1, src2, src3) dst0 = (src1 >= src2) && src3
#define ISETP_GE_U32_AND_D1(dst0, dst1, src1, src2, src3) dst1 = !dst0

#define ISETP_GT_U32_AND_D0(dst0, dst1, src1, src2, src3) dst0 = (src1 > src2) && src3
#define ISETP_GT_U32_AND_D1(dst0, dst1, src1, src2, src3) dst1 = !dst0

#define ISETP_NE_U32_AND_D0(dst0, dst1, src1, src2, src3) dst0 = (src1 != src2) && src3
#define ISETP_NE_U32_AND_D1(dst0, dst1, src1, src2, src3) dst1 = !dst0

#define ISETP_LE_AND_D0(dst0, dst1, src1, src2, src3) dst0 = ((int32_t)src1 <= (int32_t)src2) && src3
#define ISETP_LE_AND_D1(dst0, dst1, src1, src2, src3) dst1 = !dst0

#define ISETP_NE_AND_D0(dst0, dst1, src1, src2, src3) dst0 = ((int32_t)src1 != (int32_t)src2) && src3
#define ISETP_NE_AND_D1(dst0, dst1, src1, src2, src3) dst1 = !dst0

#define ISETP_EQ_U32_AND_D0(dst0, dst1, src1, src2, src3) dst0 = (src1 == src2) && src3
#define ISETP_EQ_U32_AND_D1(dst0, dst1, src1, src2, src3) dst1 = !dst0

#define UISETP_GE_U32_AND_D0(dst0, dst1, src1, src2, src3) ISETP_GE_U32_AND_D0 (dst0, dst1, src1, src2, src3)
#define UISETP_GE_U32_AND_D1(dst0, dst1, src1, src2, src3) ISETP_GE_U32_AND_D1 (dst0, dst1, src1, src2, src3)

#define UISETP_GT_U32_AND_D0(dst0, dst1, src1, src2, src3) ISETP_GT_U32_AND_D0 (dst0, dst1, src1, src2, src3)
#define UISETP_GT_U32_AND_D1(dst0, dst1, src1, src2, src3) ISETP_GT_U32_AND_D1 (dst0, dst1, src1, src2, src3)

#define UISETP_NE_U32_AND_D0(dst0, dst1, src1, src2, src3) ISETP_NE_U32_AND_D0 (dst0, dst1, src1, src2, src3)
#define UISETP_NE_U32_AND_D1(dst0, dst1, src1, src2, src3) ISETP_NE_U32_AND_D1 (dst0, dst1, src1, src2, src3)

#define UISETP_LT_AND_D0(dst0, dst1, src1, src2, src3) ISETP_LT_AND_D0 (dst0, dst1, src1, src2, src3)
#define UISETP_LT_AND_D1(dst0, dst1, src1, src2, src3) ISETP_LT_AND_D1 (dst0, dst1, src1, src2, src3)

#define MOV(dst, src)                            dst = src
#define UMOV(dst, src)                           dst = src
#define UIADD3(dst, src1, src2, src3)            dst = src1 + src2 + src3
#define LOP3_LUT(dst, src1, src2, src3, immLut)  dst = logical_op3 (src1, src2, src3, immLut)
#define ULOP3_LUT(dst, src1, src2, src3, immLut) dst = logical_op3 (src1, src2, src3, immLut)

// pretend PLOP3 is LOP3?
#define PLOP3_LUT(dst1, dst2, src1, src2, src3, immLut, src4) dst1 = logical_op3 (src1, src2, src3, immLut)

#define CS2R(dst, src) dst = src
#define IMNMX_U32(dst, src1, src2, mnpred)                                                                             \
        if (mnpred) {                                                                                                  \
                dst = src1 < src2 ? src1 : src2;                                                                       \
        } else {                                                                                                       \
                dst = src1 > src2 ? src1 : src2;                                                                       \
        } // TODO
#define ULDC(dst, src) dst = src
#define ULDC_64(dst1, dst2, src)                                                                                       \
        dst2 = (src & 0xffffffffL);                                                                                    \
        dst1 = (src >> 32)

#define concat_u32(hi, lo) ((((uint64_t)hi) << 32) | (uint64_t)lo)

#define rotate_right_64(val, rot) rot == 64 ? val : ((val << (64 - rot)) | (val >> rot))
#define rotate_left_64(val, rot)  rot == 64 ? val : ((val << rot) | (val >> (64 - rot)))

#define SHF_L_U32(dst, src1, rot, src2)  dst = rotate_left_64 (concat_u32 (src2, src1), rot)
#define USHF_L_U32(dst, src1, rot, src2) SHF_L_U32 (dst, src1, rot, src2)

#define SHF_R_U32_HI(dst, src1, rot, src2) dst = ((rotate_right_64 (concat_u32 (src2, src1), rot) >> 32) & 0xFFFFFFFFUL)
#define USHF_R_U32_HI(dst, src1, rot, src2) SHF_R_U32_HI (dst, src1, rot, src2)

#define SHF_R_S32_HI(dst, src1, rot, src2)                                                                             \
        dst = (int32_t)((rotate_right_64 ((int64_t)concat_u32 (src2, src1), rot) >> 32))
#define USHF_R_S32_HI(dst, src1, rot, src2) SHF_R_S32_HI (dst, src1, rot, src2)

#define SEL(dst, src1, src2, pred)  dst = pred ? src1 : src2
#define USEL(dst, src1, src2, pred) SEL (dst, src1, src2, pred)

#define LEA(dst, src, b, imm_shift) dst = (src << imm_shift) + b

#define LEA_HI(dst, alo, b, ahi, imm_shift) dst = ((concat_u32 (ahi, alo) << imm_shift) >> 32) + b

#define ULEA_HI(dst, alo, b, ahi, imm_shift) LEA_HI (dst, alo, b, ahi, imm_shift)

// TODO: not enough interesting inputs, need to handle SX32
// #define LEA_HI_SX32(dst, src, imm1, imm_shift) LEA_HI(dst, src, imm1, 0, imm_shift)
#define ULEA_HI_SX32(dst, src, imm1, imm_shift) LEA_HI_SX32 (dst, src, imm1, imm_shift)
#define LEA_HI_SX32(dst, alo, b, imm_shift)     dst = LEA_HI (dst, alo, b, (alo >> 31) ? 0xFFFFFFFF : 0, imm_shift)

#define P2R(dst, ign_PR, ign_RZ, pred_set) dst = pred_set

// =========================================================================
// Instructions below derived from NVBit register traces (traces.txt)
// =========================================================================

// --- Float arithmetic ---

#define FADD(dst, src1, src2)            dst = FROM_FLOAT (AS_FLOAT (src1) + AS_FLOAT (src2))
#define FADD_FTZ(dst, src1, src2)        dst = FROM_FLOAT (FTZf (AS_FLOAT (src1)) + FTZf (AS_FLOAT (src2)))
#define FFMA(dst, src1, src2, src3)      dst = FROM_FLOAT (fmaf (AS_FLOAT (src1), AS_FLOAT (src2), AS_FLOAT (src3)))
#define FMUL(dst, src1, src2)            dst = FROM_FLOAT (AS_FLOAT (src1) * AS_FLOAT (src2))
// predicated variant: @!P0 FMUL R0, R0, 0.5
#define FMUL_PRED(pred, dst, src1, src2) if (pred) { FMUL (dst, src1, src2); }

// --- Float special functions ---

// MUFU.EX2: base-2 exponential  (2^x)
// Trace: src=0x3fc00000 (1.5f) -> dst=0x403504f3 (2.8284271...) = 2^1.5
#define MUFU_EX2(dst, src) dst = FROM_FLOAT (exp2f (AS_FLOAT (src)))

// --- Float predicates ---

// FSETP.GEU.AND: Greater-or-Equal Unordered (true for >= OR NaN)
// Trace: 1.5f >= -126.0f -> true for all threads
#define FSETP_GEU_AND_D0(dst0, dst1, src1, src2, pred_src) \
        dst0 = (AS_FLOAT (src1) >= AS_FLOAT (src2) || isnan (AS_FLOAT (src1)) || isnan (AS_FLOAT (src2))) && pred_src

// FSETP.NEU.AND: Not-Equal Unordered (true for != OR NaN)
// Trace: |1.5f| != +INF -> true for all threads
#define FSETP_NEU_AND_D0(dst0, dst1, src1, src2, pred_src) \
        dst0 = (AS_FLOAT (src1) != AS_FLOAT (src2) || isnan (AS_FLOAT (src1)) || isnan (AS_FLOAT (src2))) && pred_src

// FSETP.NAN.AND: true if either operand is NaN
// Trace: |1.5f|, |1.5f| -> false (1.5 is not NaN)
#define FSETP_NAN_AND_D0(dst0, dst1, src1, src2, pred_src) \
        dst0 = (isnan (AS_FLOAT (src1)) || isnan (AS_FLOAT (src2))) && pred_src

// --- Float conversion ---

// F2FP.BF16.F32.PACK_AB: pack two f32 values into one bf16x2
// BF16 is upper 16 bits of f32. srcA -> upper half, srcB -> lower half.
// Trace: srcA=0x3fc00000, srcB=0x3fc00000 -> dst=0x3fc03fc0
#define F2FP_BF16_F32_PACK_AB(dst, srcA, srcB) \
        dst = ((srcA) & 0xFFFF0000u) | (((srcB) >> 16) & 0xFFFFu)

// --- Half-precision FMA ---

// HFMA2 (constant-loading form): -RZ * RZ + {immA, immB} = {immA, immB}
// Trace: -RZ=0, RZ=0, imm=5.96e-08 (fp16: 0x0001) -> dst=0x00010001
#define HFMA2_CONST(dst, immLo, immHi) dst = (((uint32_t)(immHi) << 16) | (uint32_t)(immLo))

// HFMA2 (FMA form): packed fp16 multiply-add on two 16-bit lanes
// Trace: src1=0x3c003c00(1.0,1.0), src2=same, src3=same -> dst=0x40004000(2.0,2.0)
// 1.0*1.0+1.0 = 2.0 per lane
#define HFMA2(dst, src1, src2, src3) do {                              \
        uint16_t s1l = (src1) & 0xFFFF, s1h = ((src1) >> 16) & 0xFFFF; \
        uint16_t s2l = (src2) & 0xFFFF, s2h = ((src2) >> 16) & 0xFFFF; \
        uint16_t s3l = (src3) & 0xFFFF, s3h = ((src3) >> 16) & 0xFFFF; \
        uint16_t rl  = half_fma (s1l, s2l, s3l);                       \
        uint16_t rh  = half_fma (s1h, s2h, s3h);                       \
        dst = ((uint32_t)rh << 16) | rl;                               \
} while (0)

// --- Integer min/max (video instructions) ---

// VIMNMX.U32: unsigned min (pred=true) or max (pred=false)
// Trace: src1=6, imm=2, PT -> dst=min(6,2)=2
#define VIMNMX_U32(dst, src1, src2, pred) do {                                                             \
        if (pred) { dst = (uint32_t)(src1) < (uint32_t)(src2) ? (uint32_t)(src1) : (uint32_t)(src2); }    \
        else      { dst = (uint32_t)(src1) > (uint32_t)(src2) ? (uint32_t)(src1) : (uint32_t)(src2); }    \
} while (0)

// VIMNMX.S32: signed min (pred=true) or max (pred=false)
// Trace: src1=16, src2=16, PT -> dst=min(16,16)=16
#define VIMNMX_S32(dst, src1, src2, pred) do {                                                             \
        if (pred) { dst = (int32_t)(src1) < (int32_t)(src2) ? (int32_t)(src1) : (int32_t)(src2); }        \
        else      { dst = (int32_t)(src1) > (int32_t)(src2) ? (int32_t)(src1) : (int32_t)(src2); }        \
} while (0)

// VIMNMX.S32.RELU: signed min/max then clamp to >= 0
// Trace: src1=16, src2=7, PT -> min(16,7)=7, max(0,7)=7
#define VIMNMX_S32_RELU(dst, src1, src2, pred) do {                                                        \
        int32_t _v;                                                                                        \
        if (pred) { _v = (int32_t)(src1) < (int32_t)(src2) ? (int32_t)(src1) : (int32_t)(src2); }         \
        else      { _v = (int32_t)(src1) > (int32_t)(src2) ? (int32_t)(src1) : (int32_t)(src2); }         \
        dst = _v > 0 ? _v : 0;                                                                            \
} while (0)

// --- Integer predicate (OR combiner) ---

// ISETP.GT.OR: (src1 > src2) || pred_src
// Trace: R2=7, R5=7, P0=false -> 7>7=false, false||false=false
// Trace: R2=7, RZ=0, P0=false -> 7>0=true,  true||false=true
#define ISETP_GT_OR_D0(dst0, dst1, src1, src2, pred_src) \
        dst0 = ((int32_t)(src1) > (int32_t)(src2)) || (pred_src)
#define ISETP_GT_OR_D1(dst0, dst1, src1, src2, pred_src) dst1 = !(dst0)

// --- VOTE ---

// VOTE.ALL: warp-wide AND of predicate (true only if ALL active threads have pred=true)
// In single-thread C simulation: identity
#define VOTE_ALL(dst_pred, src_pred) dst_pred = src_pred

// --- Uniform LEA ---

// ULEA: dst = (src1 << imm_shift) + src2
#define ULEA(dst, src1, src2, imm_shift) dst = ((uint32_t)(src1) << (imm_shift)) + (src2)

// --- Memory loads (semantics are just pointer dereferences) ---

#define LDG_E(dst, addr)                dst = *(uint32_t *)(addr)
#define LDG_E_CONSTANT(dst, addr)       dst = *(uint32_t *)(addr)
#define LDG_E_U8_CONSTANT(dst, addr)    dst = (uint32_t)(*(uint8_t *)(addr))
#define LDS(dst, addr)                  dst = *(uint32_t *)(addr)
#define LDS_U8(dst, addr)              dst = (uint32_t)(*(uint8_t *)(addr))
