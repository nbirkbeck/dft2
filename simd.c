// Num expressions: 27
// Weights: 3
void dft_8_simd(const float* input, float* real_out, int stride) {
  const __m256 kWeight2 = _mm256_set1_ps(0.707107);
  const __m256 i0 = _mm256_load_ps(input);
  const __m256 i1 = _mm256_load_ps(input + stride);
  const __m256 i2 = _mm256_load_ps(input + 2 * stride);
  const __m256 i3 = _mm256_load_ps(input + 3 * stride);
  const __m256 i4 = _mm256_load_ps(input + 4 * stride);
  const __m256 i5 = _mm256_load_ps(input + 5 * stride);
  const __m256 i6 = _mm256_load_ps(input + 6 * stride);
  const __m256 i7 = _mm256_load_ps(input + 7 * stride);

  const __m256 w8 = i1  -i5;

  const __m256 w2 = i2  + i6;

  {
  const __m256 w9 = i3  + i7;
  const __m256 w7 = i1  + i5;

  {
    const __m256 w0 = i0  + i4;

    const __m256 w4 = w0  + w2 ;
    _mm256_store_ps(real_out + 16, w0 - w2);

    const __m256 w11 = w7  + w9 ;
    _mm256_store_ps(real_out + 0, w4 + w11);
    //_mm256_store_ps(imag_out + 0, _mm256_setzero_ps());
    _mm256_store_ps(real_out + 4 * 8, w4 - w11);
    //_mm256_store_ps(imag_out + 8, _mm256_setzero_ps());
  }

  _mm256_store_ps(real_out + 16 + 5 * 8 - 8, w9 - w7);
  // Done with w5, w12, w7, w9
  }

  const __m256 w10 = i3 - i7;
  {
    const __m256 f_real = (w8 - w10) * kWeight2;
    const __m256 w1 = i0  -i4;

    _mm256_store_ps(real_out + 8, w1 + f_real);
    _mm256_store_ps(real_out + 24, w1 - f_real);
  }

  {
    const __m256 f_imag = (-w10 - w8) * kWeight2;
    const __m256 w3 = i2 - i6;
    _mm256_store_ps(real_out + 8  + 5 * 8 - 8, f_imag - w3);
    _mm256_store_ps(real_out + 24 + 5 * 8 - 8, f_imag + w3);
  }
}


void dft_8_sse(const float* input, float* output) {
  //const T kWeight2 = SimdHelper<T>::constant(0.707107);
  const __m128 i03 = _mm_load_ps(input);
  const __m128 i47 = _mm_load_ps(input + 4);
  const __m128 w0_7_2_9 = i03 + i47;
  const __m128 w1_8_3_10 = i03 - i47;

  //const T w4  = w0 +w2;
  //const T w5  = w0 +-w2;
  //const T w11  = w7 +w9;
  //const T w12  = w7 +-w9;
  __m128 w0_0_7_7 = _mm_unpacklo_ps(w0_7_2_9, w0_7_2_9);
  __m128 w2_2_9_9 = _mm_unpackhi_ps(w0_7_2_9, _mm_setzero_ps()) -
      _mm_unpackhi_ps(_mm_setzero_ps(), w0_7_2_9);
  __m128 w4_5_11_12 = w0_0_7_7 + w2_2_9_9;

  _mm_store_ps(output, w0_0_7_7);
  _mm_store_ps(output + 4, w4_5_11_12);
  return;

  // const T w6[2]  =  {w1, -w3};
  //const T w13[2]  =  {w8, w10};
  __m128 w13 = _mm_shuffle_ps(w1_8_3_10, w1_8_3_10, 1 + (1<<2) + (3<<4) + (3<<6));
  // kWeight2, -kWeight2, kWeight2, -kWeight2
  //   w13[0]    w13[0]     w13[1]    w13[1]
  __m128 t1 = _mm_shuffle_ps(w13, _mm_setzero_ps(), 2 + (2 << 2));
  __m128 t2 = _mm_shuffle_ps(_mm_setzero_ps(), w13, 0);
  __m128 kWeight2 = _mm_set_ps(0.707107, -0.707107, 0.707107, 0.707107);
  t1 = kWeight2 * (w13 - t1 + t2);
  // w4, w6[0], w5, w6[0] =
  // out[0:3] = [w4, w1, w5 w1] + [w11, x, 0, -x]
  //            [w4, x, w5, -x] + [0, w1, 0, w1] + [w11, 0, 0, 0]
  // out[4:7] = [w4, -w3, -w12, w3] + [-w11, y, 0, y]
  //__m128 w11_only =_mm_movehl_ps(_mm_setzero_ps(), w4_5_11_12);
  //w11_only = _mm_shuffle_ps(w11_only, w11_only, 0 + (2<<2) + (2<<4) + (2<<6));

  const __m128 out03 = _mm_unpacklo_ps(w4_5_11_12, t1) +
      _mm_unpacklo_ps(_mm_setzero_ps(), w1_8_3_10);// + w11_only;
  ///__m128 out47 = _mm_unpacklo_ps(w4_5_11_12, t1) -
  //    _mm_unpacklo_ps(_mm_setzero_ps(), w1_8_3_10) - w11_only;
  _mm_store_ps(output, out03);

  /*
  SimdHelper<T>::store(output, 0, w4 + w11);
  SimdHelper<T>::store(output, 1, w6[0] + kWeight2*(w13[0] - w13[1]));
  SimdHelper<T>::store(output, 2, w5);
  SimdHelper<T>::store(output, 3, w6[0] - kWeight2*(w13[0] - w13[1]));

  SimdHelper<T>::store(output, 4, w4 - w11);
  SimdHelper<T>::store(output, 5, w6[1] -  kWeight2*(w13[1] + w13[0]))
  SimdHelper<T>::store(output, 6, -w12);
  SimdHelper<T>::store(output, 7, -w6[1] - kWeight2*(w13[1] + w13[0]));
  */
}

/*
  SimdHelper<T>::store(output, 0, w4 +w11);
  SimdHelper<T>::store(output, 1, w6[0] +( kWeight2*w13[0] -  kWeight2*w13[1]));
  SimdHelper<T>::store(output, 2, w5);
  SimdHelper<T>::store(output, 3, w6[0] +(-kWeight2*w13[0] +  kWeight2*w13[1]));
  SimdHelper<T>::store(output, 4, w4 +-w11);
  SimdHelper<T>::store(output, 5, w6[1]  - (kWeight2*w13[1] + kWeight2*w13[0]));
  SimdHelper<T>::store(output, 6, -w12);
  SimdHelper<T>::store(output, 7, -w6[1] - (kWeight2*w13[1] + kWeight2*w13[0]));
*/
