#ifndef PD3_SIMD_NEON_H
#define PD3_SIMD_NEON_H

#if defined(__cplusplus)
extern "C" {
#endif // c++

#if (PD3_NEON == 1)
typedef float32x4_t f32x4;
typedef float64x2_t f64x2;
typedef int8x16_t   i8x16;
typedef int16x8_t   i16x8;
typedef int32x4_t   i32x4;
typedef int64x2_t   i64x2;
typedef uint8x16_t  u8x16;
typedef uint16x8_t  u16x8;
typedef uint32x4_t  u32x4;
typedef uint64x2_t  u64x2;
#endif // typedefs

#if defined(__cplusplus)
}
#endif // c++
#endif // PD3_SIMD_NEON_H