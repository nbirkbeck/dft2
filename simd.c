// Num expressions: 27
// Weights: 3
void dft_8_simd(float* input,
                float* real_out) {
  const __m256 kWeight2 = _mm256_set1_ps(0.707107);
  const __m256 i0 = _mm256_load_ps(input);
  const __m256 i1 = _mm256_load_ps(input + 8);
  const __m256 i2 = _mm256_load_ps(input + 16);
  const __m256 i3 = _mm256_load_ps(input + 24);
  const __m256 i4 = _mm256_load_ps(input + 32);
  const __m256 i5 = _mm256_load_ps(input + 40);
  const __m256 i6 = _mm256_load_ps(input + 48);
  const __m256 i7 = _mm256_load_ps(input + 56);

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
