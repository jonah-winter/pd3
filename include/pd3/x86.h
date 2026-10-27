#ifndef PD3_SIMD_X86_H
#define PD3_SIMD_X86_H

#if defined(__cplusplus)
extern "C" {
#endif // c++

#if (PD3_AVX == 1) || (PD3_SSE == 1)
typedef __m128  f32x4;
typedef __m128d f64x2;
typedef __m128i i8x16;
typedef __m128i i16x8;
typedef __m128i i32x4;
typedef __m128i i64x2;
typedef __m128i u8x16;
typedef __m128i u16x8;
typedef __m128i u32x4;
typedef __m128i u64x2;
#endif // typedefs

#if defined(__cplusplus)
}
#endif // c++
#endif // PD3_SIMD_X86_H