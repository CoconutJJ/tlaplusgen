#pragma once
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

static inline float TRUNCf(float x) {
  float i, f;
  f = modff(x, &i);
  return i;
}

static inline float NTZf(float x) {
  if(signbit(x))
    return 0.0;

  return x;
}

static inline float FTZf(float x) {
  if(fpclassify(x) == FP_SUBNORMAL) {
	return copysignf(0.0, x);
  }

  return x;
}

static float AS_FLOAT(uint32_t r) {
  union {
    float f;
    uint32_t i;
  } x = { .i = r};

  return x.f;
}

static uint32_t FROM_FLOAT(float f) {
  union {
    float f;
    uint32_t i;
  } x = { .f = f};

  return x.i;
}

#define S2R(dst, src) dst = src

#define IABS(dst, src) dst = abs(src)


// TODO: RP
#define I2F_RP(dst, src) dst = FROM_FLOAT((float) src)

// TODO
#define MUFU_RCP(dst, src) dst = FROM_FLOAT((1.0 / AS_FLOAT(src)))

#define IMAD_U32(dst, src1, src2, src3) dst = src1 * src2 + src3
#define IMAD(dst, src1, src2, src3) dst = src1 * src2 + src3

#define IMAD_MOV_U32(dst, src1, src2, src3) IMAD(dst, src1, src2, src3)
#define IMAD_IADD_U32(dst, src1, src2, src3) IMAD(dst, src1, src2, src3)
#define IMAD_SHL_U32(dst, src1, src2, src3) IMAD(dst, src1, src2, src3)
#define IMAD_MOV(dst, src1, src2, src3) IMAD(dst, src1, src2, src3)
#define IMAD_IADD(dst, src1, src2, src3) IMAD(dst, src1, src2, src3)

// TODO: nvbit doesn't seem to capture the right value
// see also Incomprehensible IMAD post on the dev forums
#define IMAD_HI_U32(dst, src1, src2, src3) dst = (((uint64_t) src1 * src2) + ((uint64_t) dst << 32 | src3)) >> 32

// TODO: FTZ, TRUNC, and NTZ
#define F2I_FTZ_U32_TRUNC_NTZ(dst, src) dst = (uint32_t) NTZf(TRUNCf(FTZf(AS_FLOAT(src))))

#define IADD3(dst, src1, src2, src3) dst = src1 + src2 + src3

#define EXIT() goto label_exit
#define BRA(label) goto label
#define CALL_REL_NOINC(label) goto label

#define ISETP_GE_OR_D0(dst0, dst1, src1, src2, src3) dst0 = ((int32_t) src1 >= (int32_t) src2) || src3

#define ISETP_GT_AND_D0(dst0, dst1, src1, src2, src3) dst0 = ((int32_t) src1 > (int32_t) src2) && src3
#define ISETP_GT_AND_D1(dst0, dst1, src1, src2, src3) dst1 = !dst0

#define ISETP_LT_AND_D0(dst0, dst1, src1, src2, src3) dst0 = ((int32_t) src1 < (int32_t) src2) && src3
#define ISETP_LT_AND_D1(dst0, dst1, src1, src2, src3) dst1 = !dst0

#define ISETP_GE_AND_D0(dst0, dst1, src1, src2, src3) dst0 = ((int32_t) src1 >= (int32_t) src2) && src3
#define ISETP_GE_AND_D1(dst0, dst1, src1, src2, src3) dst1 = !dst0

#define ISETP_GE_U32_AND_D0(dst0, dst1, src1, src2, src3) dst0 = (src1 >= src2) && src3
#define ISETP_GE_U32_AND_D1(dst0, dst1, src1, src2, src3) dst1 = !dst0

#define ISETP_GT_U32_AND_D0(dst0, dst1, src1, src2, src3) dst0 = (src1 > src2) && src3
#define ISETP_GT_U32_AND_D1(dst0, dst1, src1, src2, src3) dst1 = !dst0

#define ISETP_NE_U32_AND_D0(dst0, dst1, src1, src2, src3) dst0 = (src1 != src2) && src3
#define ISETP_NE_U32_AND_D1(dst0, dst1, src1, src2, src3) dst1 = !dst0

#define ISETP_NE_AND_D0(dst0, dst1, src1, src2, src3) dst0 = ((int32_t) src1 != (int32_t) src2) && src3
#define ISETP_NE_AND_D1(dst0, dst1, src1, src2, src3) dst1 = !dst0


#define ISETP_EQ_U32_AND_D0(dst0, dst1, src1, src2, src3) dst0 = (src1 == src2) && src3
#define ISETP_EQ_U32_AND_D1(dst0, dst1, src1, src2, src3) dst1 = !dst0


#define UISETP_GE_U32_AND_D0(dst0, dst1, src1, src2, src3) ISETP_GE_U32_AND_D0(dst0, dst1, src1, src2, src3)
#define UISETP_GE_U32_AND_D1(dst0, dst1, src1, src2, src3) ISETP_GE_U32_AND_D1(dst0, dst1, src1, src2, src3)

#define UISETP_GT_U32_AND_D0(dst0, dst1, src1, src2, src3) ISETP_GT_U32_AND_D0(dst0, dst1, src1, src2, src3)
#define UISETP_GT_U32_AND_D1(dst0, dst1, src1, src2, src3) ISETP_GT_U32_AND_D1(dst0, dst1, src1, src2, src3)


#define UISETP_NE_U32_AND_D0(dst0, dst1, src1, src2, src3) ISETP_NE_U32_AND_D0(dst0, dst1, src1, src2, src3)
#define UISETP_NE_U32_AND_D1(dst0, dst1, src1, src2, src3) ISETP_NE_U32_AND_D1(dst0, dst1, src1, src2, src3)

#define UISETP_LT_AND_D0(dst0, dst1, src1, src2, src3) ISETP_LT_AND_D0(dst0, dst1, src1, src2, src3)
#define UISETP_LT_AND_D1(dst0, dst1, src1, src2, src3) ISETP_LT_AND_D1(dst0, dst1, src1, src2, src3)


#define UMOV(dst, src) dst = src
#define UIADD3(dst, src1, src2, src3) dst = src1 + src2 + src3
#define LOP3_LUT(dst, src1, src2, src3, immLut) dst = logical_op3(src1, src2, src3, immLut)
#define ULOP3_LUT(dst, src1, src2, src3, immLut) dst = logical_op3(src1, src2, src3, immLut)

// pretend PLOP3 is LOP3?
#define PLOP3_LUT(dst1, dst2, src1, src2, src3, immLut, src4) dst1 = logical_op3(src1, src2, src3, immLut)

#define CS2R(dst, src) dst = src
#define IMNMX_U32(dst, src1, src2, mnpred) if(mnpred) { dst = src1 < src2 ? src1 : src2; } else { dst = src1 > src2 ? src1 : src2; } // TODO
#define ULDC(dst, src) dst = src
#define ULDC_64(dst1, dst2, src) dst2 = (src & 0xffffffffL); dst1 = (src >> 32)


#define concat_u32(hi, lo) ((((uint64_t) hi) << 32) | (uint64_t) lo)

#define rotate_right_64(val, rot) rot == 64 ? val : ((val << (64 - rot)) | (val >> rot))


#define SHF_R_U32_HI(dst, src1, rot, src2) dst = ((rotate_right_64(concat_u32(src2, src1), rot) >> 32) & 0xFFFFFFFFUL)
#define USHF_R_U32_HI(dst, src1, rot, src2) SHF_R_U32_HI(dst, src1, rot, src2)

#define SHF_R_S32_HI(dst, src1, rot, src2) dst = (int32_t) ((rotate_right_64((int64_t) concat_u32(src2, src1), rot) >> 32))
#define USHF_R_S32_HI(dst, src1, rot, src2) SHF_R_S32_HI(dst, src1, rot, src2)


#define SEL(dst, src1, src2, pred) dst = pred ? src1 : src2
#define USEL(dst, src1, src2, pred) SEL(dst, src1, src2, pred)

#define LEA_HI(dst, alo, b, ahi, imm_shift) dst = ((concat_u32(ahi, alo) << imm_shift) >> 32) + b

#define ULEA_HI(dst, alo, b, ahi, imm_shift) LEA_HI(dst, alo, b, ahi, imm_shift)

// TODO: not enough interesting inputs, need to handle SX32
#define LEA_HI_SX32(dst, src, imm1, imm_shift) LEA_HI(dst, src, imm1, 0, imm_shift)
#define ULEA_HI_SX32(dst, src, imm1, imm_shift) LEA_HI_SX32(dst, src, imm1, imm_shift)

#define P2R(dst, ign_PR, ign_RZ, pred_set) dst = pred_set
