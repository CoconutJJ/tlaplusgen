/*
 * missing_sass_kernels.cu
 *
 * CUDA kernels designed to compile down to the SASS instructions listed in
 * src/missing_sass.txt. Compile for the appropriate target architecture:
 *
 *   nvcc -arch=sm_100a -o missing_sass_kernels missing_sass_kernels.cu
 *   cuobjdump -sass missing_sass_kernels
 *
 * IMPORTANT: The exact SASS generated depends on nvcc version and target arch.
 * These kernels are best-effort attempts to elicit the target instructions.
 * Some instructions (UTCATOMSWS, R2UR.OR, UP2UR, UPLOP3.LUT) are generated
 * by internal compiler machinery or CUTLASS/CuTe and cannot be reliably
 * produced from plain CUDA source.
 */

#include <cuda_bf16.h>
#include <cuda_fp16.h>

/* =======================================================================
 * Float arithmetic: FADD, FADD.FTZ, FFMA, FFMA2.FTZ.RZ, FMUL
 * ======================================================================= */

/* FADD: dst = src1 + src2 */
__global__ void kernel_fadd (const float *__restrict__ a, const float *__restrict__ b, float *__restrict__ out, int n)
{
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n)
                out[idx] = a[idx] + b[idx];
}

/* FFMA: dst = src1 * src2 + src3 */
__global__ void kernel_ffma (const float *__restrict__ a,
                             const float *__restrict__ b,
                             const float *__restrict__ c,
                             float *__restrict__ out,
                             int n)
{
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n)
                out[idx] = __fmaf_rn (a[idx], b[idx], c[idx]);
}

/* FMUL: dst = src1 * src2 */
__global__ void kernel_fmul (const float *__restrict__ a, const float *__restrict__ b, float *__restrict__ out, int n)
{
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n)
                out[idx] = a[idx] * b[idx];
}

/* FFMA2.FTZ.RZ: packed FP32x2 fused multiply-add
 * LOW CONFIDENCE — this is a packed instruction the compiler emits for
 * vectorized FP32 code paths. Difficult to force from plain CUDA. */

/* =======================================================================
 * Float predicate: FSETP.GTU.FTZ.AND, FSETP.NEU.FTZ.AND, FCHK
 * ======================================================================= */

/* FSETP.GTU.FTZ.AND: float set-predicate, greater-than-unordered
 * Used by the compiler to check for NaN/Inf: |x| > +INF is true for NaN */
__global__ void kernel_fsetp_gtu (const float *__restrict__ in, int *__restrict__ out, int n)
{
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
                float val = in[idx];
                /* isnan check compiles to FSETP.GTU |val|, +INF */
                out[idx]  = isnan (val) ? 1 : 0;
        }
}

/* FSETP.NEU.FTZ.AND: float set-predicate, not-equal-unordered */
__global__ void kernel_fsetp_neu (const float *__restrict__ in, int *__restrict__ out, int n)
{
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
                float val = in[idx];
                /* isinf check — FSETP.NEU |val|, +INF is true when val is finite or NaN */
                out[idx]  = isinf (val) ? 0 : 1;
        }
}

/* FCHK: float range check (sets predicate)
 * LOW CONFIDENCE — compiler-internal instruction for FP exception checking */

/* =======================================================================
 * Float conversion: F2FP.BF16.F32.PACK_AB, F2FP.SATFINITE.BF16.*, HFMA2
 * ======================================================================= */

/* F2FP.BF16.F32.PACK_AB: float32 to BF16 packed conversion */
__global__ void kernel_f2fp_bf16_pack (const float *__restrict__ a,
                                       const float *__restrict__ b,
                                       __nv_bfloat162 *__restrict__ out,
                                       int n)
{
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
                /* Pack two float32 values into one BF16x2 */
                out[idx] = __floats2bfloat162_rn (a[idx], b[idx]);
        }
}

/* F2FP.SATFINITE.BF16.S2_6.UNPACK_B: float to BF16 with saturation */
__global__ void kernel_f2fp_bf16_sat (const float *__restrict__ in, __nv_bfloat16 *__restrict__ out, int n)
{
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n)
                out[idx] = __float2bfloat16_rn (in[idx]);
}

/* HFMA2: packed half-precision fused multiply-add */
__global__ void kernel_hfma2 (const __half2 *__restrict__ a,
                              const __half2 *__restrict__ b,
                              const __half2 *__restrict__ c,
                              __half2 *__restrict__ out,
                              int n)
{
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n)
                out[idx] = __hfma2 (a[idx], b[idx], c[idx]);
}

/* HFMA2 constant-loading pattern seen in real kernels:
 *   HFMA2 Rx, RZ, RZ, 0, <small_constant>
 * i.e., 0 * 0 + eps = eps */
__global__ void kernel_hfma2_const (__half2 *__restrict__ out, int n)
{
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n)
                out[idx] = __float2half2_rn (5.9604644775390625e-08f);
}

/* =======================================================================
 * Float special: MUFU.EX2
 * ======================================================================= */

/* MUFU.EX2: multi-function unit base-2 exponential (2^x) */
__global__ void kernel_mufu_ex2 (const float *__restrict__ in, float *__restrict__ out, int n)
{
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n)
                out[idx] = exp2f (in[idx]);
}

/* =======================================================================
 * Integer (video/packed): VIADD, VIADDMNMX, VIADDMNMX.U32, VIMNMX
 * ======================================================================= */

/* VIADD: integer add (likely scalar on Blackwell/Hopper) */
__global__ void kernel_viadd (int *__restrict__ data, int n, int stride)
{
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
                int val   = data[idx];
                val       = val + 1;
                val       = val + stride;
                val       = val - 2;
                data[idx] = val;
        }
}

/* VIADDMNMX.U32: unsigned integer add + min/max */
__global__ void kernel_viaddmnmx_u32 (unsigned int *__restrict__ data, int n)
{
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
                unsigned int a       = data[idx];
                unsigned int b       = data[idx + n];
                unsigned int clamped = min (a + b, 2u);
                data[idx]            = clamped;
        }
}

/* VIADDMNMX: signed integer add + min/max */
__global__ void kernel_viaddmnmx (int *__restrict__ data, int n)
{
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
                int a       = data[idx];
                int b       = data[idx + n];
                int clamped = min (a + b, n);
                data[idx]   = clamped;
        }
}

/* VIMNMX: integer min/max (video variant) */
__global__ void kernel_vimnmx (int *__restrict__ data, int n)
{
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
                int a     = data[idx];
                int b     = data[idx + n];
                data[idx] = max (0, min (a, b));
        }
}

/* =======================================================================
 * Integer predicate: ISETP.GT.OR, ISETP.GT.U32.OR, ISETP.NE.OR,
 *                    UISETP.EQ.XOR
 * ======================================================================= */

/* ISETP.GT.OR / ISETP.GT.U32.OR: (src1 > src2) OR pred */
__global__ void kernel_isetp_gt_or (const int *__restrict__ a, const int *__restrict__ b, int *__restrict__ out, int n)
{
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
                int va = a[idx], vb = b[idx];
                /* Chain of OR'd comparisons — compiler should emit ISETP.GT.OR */
                bool p   = (va > 0) || (va > vb) || (vb > n);
                out[idx] = p ? 1 : 0;
        }
}

/* ISETP.NE.OR: (src1 != src2) OR pred */
__global__ void kernel_isetp_ne_or (const int *__restrict__ a, int *__restrict__ out, int n)
{
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
                int v    = a[idx];
                bool p   = (v != 0) || (v != n);
                out[idx] = p ? 1 : 0;
        }
}

/* UISETP.EQ.XOR: (src1 == src2) XOR pred (uniform)
 * Uniform predicates with XOR combiner appear in warp-specialization
 * scheduling code. Difficult to force from plain CUDA. */

/* =======================================================================
 * Memory loads: LDG.E, LDS, LDS.64, LDS.128, LDS.U8, LDL, LDL.LU,
 *               LDSM.16.MT88.4
 * ======================================================================= */

/* LDG.E: load from global memory */
__global__ void kernel_ldg (const int *__restrict__ in, int *__restrict__ out, int n)
{
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n)
                out[idx] = in[idx];
}

/* LDS / LDS.64 / LDS.128: load from shared memory */
__global__ void kernel_lds (const int *__restrict__ in, int *__restrict__ out, int n)
{
        __shared__ int smem[256];
        __shared__ long long smem64[128];
        int tid = threadIdx.x;
        if (tid < n) {
                smem[tid]   = in[tid];
                smem64[tid] = (long long)in[tid];
        }
        __syncthreads ();
        if (tid < n) {
                /* LDS: 32-bit shared load */
                int a       = smem[tid];
                /* LDS.64: 64-bit shared load */
                long long b = smem64[tid];
                out[tid]    = a + (int)b;
        }
}

/* LDS.U8: unsigned 8-bit shared memory load */
__global__ void kernel_lds_u8 (const unsigned char *__restrict__ in, int *__restrict__ out, int n)
{
        __shared__ unsigned char smem[256];
        int tid = threadIdx.x;
        if (tid < n)
                smem[tid] = in[tid];
        __syncthreads ();
        if (tid < n) {
                /* 8-bit load from shared memory — should emit LDS.U8 */
                out[tid] = (int)smem[tid];
        }
}

/* LDL / LDL.LU: load from local memory
 * Local memory is used for register spills. Force spills with large
 * arrays that exceed register file capacity. */
__global__ void kernel_ldl (int *__restrict__ out, int n)
{
        int local_array[64];
        int tid = threadIdx.x;
        for (int i = 0; i < 64; i++)
                local_array[i] = tid + i;
        if (tid < n) {
                /* Accessing spilled local array should emit LDL */
                int sum = 0;
                for (int i = 0; i < 64; i++)
                        sum += local_array[i];
                out[tid] = sum;
        }
}

/* LDSM.16.MT88.4: load shared memory matrix
 * This is a matrix load instruction used with tensor cores.
 * LOW CONFIDENCE — requires specific memory layout and mma usage. */

/* =======================================================================
 * Register movement: R2UR, R2UR.OR, UP2UR, ULEA
 * ======================================================================= */

/* R2UR: move regular register to uniform register */
__global__ void kernel_r2ur (const int *__restrict__ in, int *__restrict__ out, int n)
{
        int block_offset = blockIdx.x * blockDim.x;
        int tid          = threadIdx.x;
        int uniform_val  = block_offset + blockDim.x;
        if (tid < n)
                out[tid] = in[tid + uniform_val];
}

/* R2UR.OR: warp-level OR-reduction into uniform register
 * LOW CONFIDENCE — may come from __ballot_sync or similar */
__global__ void kernel_r2ur_or (const int *__restrict__ in, unsigned int *__restrict__ out, int n)
{
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
                unsigned int mask = __ballot_sync (0xffffffff, in[idx] > 0);
                if (threadIdx.x == 0)
                        out[blockIdx.x] = mask;
        }
}

/* UP2UR: uniform predicate to uniform register
 * LOW CONFIDENCE — compiler-internal for predicate materialization */

/* ULEA: uniform load effective address (dst = (src << shift) + b)
 * Generated when the compiler computes uniform addresses for
 * descriptor-based memory accesses. Difficult to isolate from CUDA. */

/* =======================================================================
 * Uniform predicate logic: UPLOP3.LUT
 * ======================================================================= */

/* UPLOP3.LUT: uniform predicate 3-input LUT
 * LOW CONFIDENCE — compiler-internal for uniform predicate manipulation.
 * Cannot be reliably generated from plain CUDA. */

/* =======================================================================
 * Warp/CTA ops: ELECT, UTCATOMSWS, VOTE.ALL
 * ======================================================================= */

/* ELECT: elect one lane in warp
 * The compiler emits ELECT for warp-leader patterns. */
__global__ void kernel_elect (int *__restrict__ out)
{
        /* Only one thread per warp writes — compiler may emit ELECT */
        int warp_id = threadIdx.x / 32;
        int lane    = threadIdx.x % 32;
        if (lane == 0)
                out[warp_id] = 1;
}

/* VOTE.ALL: warp vote (all lanes)
 * __all_sync intrinsic should emit VOTE.ALL */
__global__ void kernel_vote_all (const int *__restrict__ in, int *__restrict__ out, int n)
{
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
                int val          = in[idx];
                int all_positive = __all_sync (0xffffffff, val > 0);
                if (threadIdx.x == 0)
                        out[blockIdx.x] = all_positive;
        }
}

/* UTCATOMSWS.FIND_AND_SET.ALIGN / UTCATOMSWS.2CTA.FIND_AND_SET.ALIGN
 * Blackwell warp-specialization instructions. Generated by CUTLASS/CuTe
 * warp specialization runtime. NO CUDA SOURCE can reliably emit these.
 * To obtain SASS containing UTCATOMSWS, compile a CuTe DSL kernel that
 * uses warp specialization. */

/* =======================================================================
 * Main: call every kernel with example inputs
 * ======================================================================= */

#include <cstdio>
#include <cstring>

#define N     256
#define BLOCK 256
#define GRID  1

#define CHECK(call)                                                                                                    \
        do {                                                                                                           \
                cudaError_t err = (call);                                                                              \
                if (err != cudaSuccess) {                                                                              \
                        fprintf (stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__, cudaGetErrorString (err));  \
                        exit (1);                                                                                      \
                }                                                                                                      \
        } while (0)

template <typename T>
static T *alloc (int count)
{
        T *d;
        CHECK (cudaMalloc (&d, count * sizeof (T)));
        CHECK (cudaMemset (d, 0, count * sizeof (T)));
        return d;
}

template <typename T>
static T *alloc (int count, T fill)
{
        T *d;
        CHECK (cudaMalloc (&d, count * sizeof (T)));
        T *h = (T *)malloc (count * sizeof (T));
        for (int i = 0; i < count; i++)
                h[i] = fill;
        CHECK (cudaMemcpy (d, h, count * sizeof (T), cudaMemcpyHostToDevice));
        free (h);
        return d;
}

int main ()
{
        /* --- Allocate device buffers --- */
        float *d_float          = alloc<float> (N * 2, 1.5f);
        float *d_float_b        = d_float + N;
        int *d_int              = alloc<int> (N * 2, 7);
        int *d_int_b            = d_int + N;
        unsigned int *d_uint    = alloc<unsigned int> (N * 2, 3u);
        unsigned char *d_uchar  = alloc<unsigned char> (N, 42);
        __half2 *d_half2        = alloc<__half2> (N * 2, __float2half2_rn (1.0f));
        __half2 *d_half2_b      = d_half2 + N;
        __nv_bfloat16 *d_bf16   = alloc<__nv_bfloat16> (N);
        __nv_bfloat162 *d_bf162 = alloc<__nv_bfloat162> (N);
        long long *d_ll         = alloc<long long> (N, 0LL);

        float *d_fout           = alloc<float> (N);
        int *d_iout             = alloc<int> (N);
        unsigned int *d_uout    = alloc<unsigned int> (N);
        __half2 *d_hout         = alloc<__half2> (N);

        printf ("Running kernels...\n");

        /* Float arithmetic */
        kernel_fadd<<<GRID, BLOCK>>> (d_float, d_float_b, d_fout, N);
        kernel_ffma<<<GRID, BLOCK>>> (d_float, d_float_b, d_fout, d_fout, N);
        kernel_fmul<<<GRID, BLOCK>>> (d_float, d_float_b, d_fout, N);

        /* Float predicate */
        kernel_fsetp_gtu<<<GRID, BLOCK>>> (d_float, d_iout, N);
        kernel_fsetp_neu<<<GRID, BLOCK>>> (d_float, d_iout, N);

        /* Float conversion */
        kernel_f2fp_bf16_pack<<<GRID, BLOCK>>> (d_float, d_float_b, d_bf162, N);
        kernel_f2fp_bf16_sat<<<GRID, BLOCK>>> (d_float, d_bf16, N);
        kernel_hfma2<<<GRID, BLOCK>>> (d_half2, d_half2_b, d_half2, d_hout, N);
        kernel_hfma2_const<<<GRID, BLOCK>>> (d_hout, N);

        /* Float special */
        kernel_mufu_ex2<<<GRID, BLOCK>>> (d_float, d_fout, N);

        /* Integer (video/packed) */
        kernel_viadd<<<GRID, BLOCK>>> (d_int, N, 3);
        kernel_viaddmnmx_u32<<<GRID, BLOCK>>> (d_uint, N);
        kernel_viaddmnmx<<<GRID, BLOCK>>> (d_int, N);
        kernel_vimnmx<<<GRID, BLOCK>>> (d_int, N);

        /* Integer predicate */
        kernel_isetp_gt_or<<<GRID, BLOCK>>> (d_int, d_int_b, d_iout, N);
        kernel_isetp_ne_or<<<GRID, BLOCK>>> (d_int, d_iout, N);

        /* Memory loads */
        kernel_ldg<<<GRID, BLOCK>>> (d_int, d_iout, N);
        kernel_lds<<<GRID, BLOCK>>> (d_int, d_iout, N);
        kernel_lds_u8<<<GRID, BLOCK>>> (d_uchar, d_iout, N);
        kernel_ldl<<<GRID, BLOCK>>> (d_iout, N);

        /* Register movement */
        kernel_r2ur<<<GRID, BLOCK>>> (d_int, d_iout, N);
        kernel_r2ur_or<<<GRID, BLOCK>>> (d_int, d_uout, N);

        /* Warp/CTA ops */
        kernel_elect<<<GRID, BLOCK>>> (d_iout);
        kernel_vote_all<<<GRID, BLOCK>>> (d_int, d_iout, N);

        CHECK (cudaDeviceSynchronize ());
        printf ("All kernels completed successfully.\n");

        /* --- Cleanup --- */
        cudaFree (d_float);
        cudaFree (d_int);
        cudaFree (d_uint);
        cudaFree (d_uchar);
        cudaFree (d_half2);
        cudaFree (d_bf16);
        cudaFree (d_bf162);
        cudaFree (d_ll);
        cudaFree (d_fout);
        cudaFree (d_iout);
        cudaFree (d_uout);
        cudaFree (d_hout);

        return 0;
}
