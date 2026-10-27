#ifndef PD3_SIMD_H
#define PD3_SIMD_H

#if !defined(PD3_NO_SIMD)
#if defined(__cplusplus)
extern "C" {
#endif

#define PD3_NEON 0
#define PD3_SSE  0
#define PD3_AVX  0
#define PD3_NO_SIMD 0

#if defined(__AVX__) || defined(__AVX2__)
#  include <immintrin.h>
#  undef  PD3_AVX
#  define PD3_AVX 1
#elif defined(__SSE__) || \
     (defined(_M_IX86_FP) && _M_IX86_FP >= 1) || \
      defined(_M_X64)
#  include <xmmintrin.h>
#  undef  PD3_SSE
#  define PD3_SSE 1
#elif defined(__aarch64__) || defined(_M_ARM64)
#  include <arm_neon.h>
#  undef  PD3_NEON
#  define PD3_NEON 1
#else
#  undef  PD3_NO_SIMD
#  define PD3_NO_SIMD 1
#endif

#if (PD3_AVX == 1) || (PD3_SSE == 1)
#  include "pd3/x86.h"
#elif (PD3_NEON == 1)
#  include "pd3/neon.h"
#endif

static inline f32x4 broadcast4_f32(float val) {
#if (PD3_AVX == 1) || (PD3_SSE == 1)
  return _mm_set1_ps(val);
#elif (PD3_NEON == 1)
  return vdupq_n_f32(val);
#else
#  if defined(__cplusplus)
  return f32x4{val, val, val, val};
#  else
  f32x4 fallback = { .val = {val, val, val, val} };
  return fallback;
#  endif
#endif
}

#if defined(__cplusplus)
}
#endif

#else
typedef struct {
  float val[4];
} f32x4;
#endif // PD3_SIMD
#endif // PD3_SIMD_H
