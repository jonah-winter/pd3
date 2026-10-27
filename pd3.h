/*
Copyright (c) 2026 Jonah Winter
This is a single header library for computing various decompositions on 3x3
matrices. The matrices are stored as naive float arrays. The library is
available in the public domain or under an MIT license (See LICENSE).

Sources:
https://minds.wisc.edu/server/api/core/bitstreams/3e736430-d9da-439d-905c-8c53f4fa182e/content
https://github.com/mlourakis/svd3x3
https://gitlab.com/libeigen/eigen/-/tree/master/Eigen/src/SVD

Options:
1. PD3_ASSERT: Change assertion behavior. Automatically turns off if NDEBUG flag
is set unless behavior is specified.
2. PD3_STATIC: Change linkage of functions that are exposed to the user.
3. PD3_SCALAR: Change scalar type used. Default is float.
*/
#ifndef PD3_H
#define PD3_H

#if defined(__cplusplus)
extern "C" {
#endif /* C++ */

#include <math.h>
#include <string.h>
#include <float.h>

#ifndef PD3_ASSERT
#    include <assert.h>
#    define PD3_ASSERT(x) assert((x))
#endif

#define PD3_INTERNAL_STR_IMPL(x) #x
#define PD3_STR(x) PD3_INTERNAL_STR_IMPL(x)
#define PD3_PRAGMA(x) _Pragma(PD3_STR(x))

#if defined(__clang__)
#    define PD3_UNROLL_FULL PD3_PRAGMA(clang loop unroll(full))
#    define PD3_UNROLL(n) PD3_PRAGMA(clang loop unroll_count(n))
#elif defined(__GNUC__) && (__GNUC__ >= 8)
#    define PD3_UNROLL_FULL PD3_PRAGMA(GCC unroll 8)
#    define PD3_UNROLL(n) PD3_PRAGMA(GCC unroll n)
#elif defined(__CUDACC__) || defined(__INTEL_COMPILER) || \
    defined(__INTEL_LLVM_COMPILER)
#    define PD3_UNROLL_FULL PD3_PRAGMA(unroll)
#    define PD3_UNROLL(n) PD3_PRAGMA(unroll n)
#else
#    define PD3_UNROLL_FULL
#    define PD3_UNROLL(n)
#endif /* check compilers */

#if defined(PD3_STATIC)
#    define PD3_EXTERN static
#else
#    if defined(__cplusplus)
#        define PD3_EXTERN extern "C"
#    else
#        define PD3_EXTERN extern
#    endif /* C++ */
#endif     /* !defined(PD3_STATIC) */

/* stays defined */
#if !defined(PD3_SCALAR)
#    define PD3_SCALAR float
#endif                    /* PD3_SCALAR */
#define Scalar PD3_SCALAR /* just for editors for convenience */

#define PD3_SUCC 0
#define PD3_ERRC 1

/*
== SIMD BACKEND ===================================
Operations like loading, storing, arithmetic using portable and dynamic SIMD
functions + macros.
*/

/* CONFIG */

#if !defined(PD3_NO_SIMD)
#    if (defined(__AVX__) || defined(__AVX2__)) && \
        !defined(                                  \
            PD3_AVX1) /* check avx2 and then avx if avx2 support matters */
#        include <immintrin.h>
#        define PD3_AVX1 1
#    elif (defined(__SSE__) || defined(_M_X64) ||       \
           (defined(_M_IX86_FP) && _M_IX86_FP >= 1)) && \
        !defined(PD3_SSE1) /* change to == when sse2 support comes */
#        include <xmmintrin.h>
#        define PD3_SSE1 1
#    elif (defined(__aarch64__) || defined(_M_ARM64)) && !defined(PD3_NEON)
#        include <arm_neon.h>
#        define PD3_NEON 1
#    endif
#endif

#if !defined(PD3_NO_SIMD) && !defined(PD3_AVX1) && !defined(PD3_SSE1) && \
    !defined(PD3_NEON)
#    error \
        "No SIMD implementation is available for this platform. Opt out by defining the PD3_NO_SIMD option."
#endif

/* TYPES */

#if defined(PD3_SSE1)
typedef __m128 pd64; /* treat like 2 values, in first two register slots; always
                        remember to set top 2 vals to 0.0f */
typedef __m128 pd128;
typedef struct {
  __m128 lo;
  __m128 hi;
} pd256;
#elif defined(PD3_AVX1)
typedef __m128 pd64; /* see prev. comment */
typedef __m128 pd128;
typedef __m256 pd256;
#elif defined(PD3_NEON)
typedef float32x2_t pd64;
typedef float32x4_t pd128;
typedef struct {
  float32x4_t lo;
  float32x4_t hi;
} pd256;
#else
typedef struct {
  float v[2];
} pd64;
typedef struct {
  float v[4];
} pd128;
typedef struct {
  float v[8];
} pd256;
#endif

#if defined(PD3_NEON)
static void pd64_zero(pd64* vec) { *vec = vdup_n_f32(0.0f); }
static void pd128_zero(pd128* vec) { *vec = vdupq_n_f32(0.0f); }
static void pd256_zero(pd256* vec) {
  (*vec).lo = vdupq_n_f32(0.0f);
  (*vec).hi = vdupq_n_f32(0.0f);
}

#elif defined(PD3_SSE1)
static void pd64_zero(pd64* vec) { *vec = _mm_setzero_ps(); }
static void pd128_zero(pd128* vec) { *vec = _mm_setzero_ps(); }
static void pd256_zero(pd256* vec) {
  (*vec).lo = _mm_setzero_ps();
  (*vec).hi = _mm_setzero_ps();
}

#elif defined(PD3_AVX1)
static void pd64_zero(pd64* vec) { *vec = _mm_setzero_ps(); }
static void pd128_zero(pd128* vec) { *vec = _mm_setzero_ps(); }
static void pd256_zero(pd256* vec) { *vec = _mm256_setzero_ps(); }

#else
static void pd64_zero(pd64* vec) { *vec = {0}; }
static void pd128_zero(pd128* vec) { *vec = {0}; }
static void pd256_zero(pd256* vec) { *vec = {0}; }

#endif /* SIMD FUNCTIONS */

/*
== END OF SIMD BACKEND ============================
*/

typedef float pdmat3[9];

static void pd3_make_mat(const float data[9], pdmat3 out) {
  const pdmat3 tmp = {
      data[0], data[3], data[6], data[1], data[4],
      data[7], data[2], data[5], data[8],
  };
  memcpy(out, tmp, sizeof(pdmat3));
}

static inline float pd3_det(const pdmat3 x) {
  const float a = x[4] * x[8] - x[5] * x[7];
  const float b = x[1] * x[8] - x[2] * x[7];
  const float c = x[1] * x[5] - x[2] * x[4];
  return x[0] * a - x[3] * b + x[6] * c;
}

static void pd3_adj(const pdmat3 x, pdmat3 out) {
  const pdmat3 tmp = {
      x[4] * x[8] - x[5] * x[7], x[5] * x[6] - x[3] * x[8],
      x[3] * x[7] - x[4] * x[6], x[2] * x[7] - x[1] * x[8],
      x[0] * x[8] - x[2] * x[6], x[1] * x[6] - x[0] * x[7],
      x[1] * x[5] - x[2] * x[4], x[2] * x[3] - x[0] * x[5],
      x[0] * x[4] - x[1] * x[3],
  };
  memcpy(out, tmp, sizeof(pdmat3));
}

static void pd3_scale_mat(pdmat3 out, const float s) {
  /*PD3_UNROLL(9)*/
  for(int i = 0; i < 9; i++) { out[i] *= s; }
}

static int pd3_inv(const pdmat3 x, pdmat3 out, const float tol) {
  const float c00 = x[4] * x[8] - x[5] * x[7];
  const float c01 = x[5] * x[6] - x[3] * x[8];
  const float c02 = x[3] * x[7] - x[4] * x[6];
  const float det = x[0] * c00 + x[1] * c01 + x[2] * c02;
  if(!isfinite(det) || fabsf(det) < tol) { return PD3_ERRC; }
  const float d = 1.0f / det;
  out[0] = c00 * d;
  out[1] = c01 * d;
  out[2] = c02 * d;
  out[3] = (x[7] * x[2] - x[1] * x[8]) * d;
  out[4] = (x[0] * x[8] - x[2] * x[6]) * d;
  out[5] = (x[1] * x[6] - x[0] * x[7]) * d;
  out[6] = (x[1] * x[5] - x[4] * x[2]) * d;
  out[7] = (x[2] * x[3] - x[0] * x[5]) * d;
  out[8] = (x[0] * x[4] - x[1] * x[3]) * d;
  return PD3_SUCC;
}

static int pd3_inv_t(const pdmat3 x, pdmat3 out, const float tol) {
  const float c00 = x[4] * x[8] - x[5] * x[7];
  const float c01 = x[5] * x[6] - x[3] * x[8];
  const float c02 = x[3] * x[7] - x[4] * x[6];

  const float det = x[0] * c00 + x[1] * c01 + x[2] * c02;
  if(!isfinite(det) || fabsf(det) < tol) { return PD3_ERRC; }
  const float d = 1.0f / det;

  const float c10 = x[7] * x[2] - x[1] * x[8];
  const float c11 = x[0] * x[8] - x[2] * x[6];
  const float c12 = x[1] * x[6] - x[0] * x[7];
  const float c20 = x[1] * x[5] - x[4] * x[2];
  const float c21 = x[2] * x[3] - x[0] * x[5];
  const float c22 = x[0] * x[4] - x[1] * x[3];

  out[0] = c00 * d;
  out[3] = c10 * d;
  out[6] = c20 * d;
  out[1] = c01 * d;
  out[4] = c11 * d;
  out[7] = c21 * d;
  out[2] = c02 * d;
  out[5] = c12 * d;
  out[8] = c22 * d;
  return PD3_SUCC;
}

static inline float pd3_mag(const pdmat3 a) {
  /*
  float32x4_t v1 = vld1q_float(a);
  float32x4_t v2 = vld1q_float(a+4);

  float32x4_t v1sq = vmulq_float(v1, v1);
  v1sq = vmlaq_float(v1sq, v2, v2);

  float s = vaddvq_float(v1sq);
  s += a[8]*a[8];
  */
  float s = 0;
  /*PD3_UNROLL(9)*/
  for(int i = 0; i < 9; i++) { s += a[i] * a[i]; }
  return sqrtf(s);
}

static int pd3_add_mat(const pdmat3 a, const pdmat3 b, pdmat3 c) {
  PD3_UNROLL(9)
  for(int i = 0; i < 9; i++) { c[i] = a[i] + b[i]; }
  return PD3_SUCC;
}

/* frobenius norm */
static float pd3_frob(const pdmat3 x) {
  float acc = 0;
  for(int i = 0; i < 9; i++) { acc += x[i] * x[i]; }
  return sqrtf(acc);
}

/* max abs diff */
static float pd3_mad(const pdmat3 a, const pdmat3 b) {
  float max = 0;
  for(int i = 0; i < 9; i++) {
    float v = fabsf(a[i] - b[i]);
    max = (v > max) ? v : max;
  }
  return max;
}

PD3_EXTERN int pd3_is_nonsingular(const pdmat3 a) {}

PD3_EXTERN int pd3_newton(const pdmat3 a, pdmat3 u, pdmat3 p, const float tol,
                   const int max_iter, char* err) {
  pdmat3 r;
  memcpy(r, a, sizeof(pdmat3));
  pdmat3 r_ti = {0};
  pdmat3 next = {0};
  int loops = 0;
  float norm_r = pd3_frob(r);

  while(loops < max_iter) {
    if(!isfinite(norm_r) || norm_r < FLT_MIN) {
        strcpy(err, "frobenius norm of matrix is too small or 0");
        return PD3_ERRC; 
    }
    if(pd3_inv_t(r, r_ti, tol) == PD3_ERRC) { 
        strcpy(err, "failed to inverse transpose");
        return PD3_ERRC; 
    }
    float gamma = sqrtf(pd3_frob(r_ti) / norm_r);
    // PD3_UNROLL(9)
    for(int i = 0; i < 9; i++) {
      next[i] = (r[i] * gamma + r_ti[i] * (1.0f / gamma)) * 0.5f;
    }
    loops++;
    if(pd3_mad(next, r) <= tol * fmaxf(pd3_frob(next), 1.0f)) {
      memcpy(u, next, sizeof(pdmat3));
      /* add p */

      return PD3_SUCC;
    }
    memcpy(r, next, sizeof(pdmat3));
    norm_r = pd3_frob(r);
  }
  strcpy(err, "not enough iterations");
  return PD3_ERRC;
}

#if defined(PD3_IMPLEMENTATION)

#endif /* PD3_IMPLEMENTATION */

#if defined(__cplusplus)
}
#endif /* __cplusplus */

#undef Scalar
#undef PD3_PRAGMA
#undef PD3_STR
#undef PD3_INTERNAL_STR_IMPL
/*#undef PD3_SUCC
#undef PD3_ERRC*/
#undef PD3_ASSERT
#undef PD3_UNROLL
#undef PD3_UNROLL_FULL
#undef PD3_EXTERN
#endif /* PD3_H */