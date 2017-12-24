// Num expressions: 4
// Weights: 2
void dft_2(float* input, std::complex<float>* output) {
  output[0] = input[0]  + input[1] ;
  output[1] = input[0]  + -input[1] ;
}

// Num expressions: 11
// Weights: 2
template <class T>
void dft_4(T* input, std::complex<T>* output) {
  const T w0 = input[0]  + input[2] ;
  const T w1 = input[0]  + -input[2] ;
  const T w2 = input[1]  + input[3] ;
  const T w3 = input[1]  + -input[3] ;
  output[0] = w0  + w2 ;
  output[2] = w0  + -w2 ;
  output[1] =  {w1   ,   -w3};
}

// Num expressions: 11
// Weights: 2
void dft_4x4(float* input, float* output) {
  __m128 i0 = _mm_load_ps(input);
  __m128 i1 = _mm_load_ps(input + 4);
  __m128 i2 = _mm_load_ps(input + 8);
  __m128 i3 = _mm_load_ps(input + 12);

  __m128 w0 = i0 + i2;
  __m128 w1 = i0 - i2;
  __m128 w2 = i1 + i3;
  __m128 w3 = i1 - i3;

  __m128 o0_r = w0 + w2;
  __m128 o1_r = w1;
  __m128 o1_i = -w3;
  __m128 o2_r = w0 - w2;
  __m128 o3_r = w1;

  _MM_TRANSPOSE4_PS(o0_r, o1_r, o2_r, o3_r);
  w0 = o0_r + o2_r;
  w1 = o0_r - o2_r;
  w2 = o1_r + o3_r;
  w3 = o1_r - o3_r;

  o0_r = w0 + w2;
  o1_r = w1;
  o1_i = -w3;
  o2_r = w0 - w2;
  o3_r = o1_r;

  _MM_TRANSPOSE4_PS(o0_r, o1_r, o2_r, o3_r);
  _mm_store_ps(output, o0_r);
  _mm_store_ps(output + 4, o1_r);
  _mm_store_ps(output + 8, o2_r);
  _mm_store_ps(output + 12, o3_r);
}

// Num expressions: 27
// Weights: 3
template <class T>
void dft_8(T* input, std::complex<T>* output) {
  const T kWeight2 = load<T>(0.707107);
  // t0 = 0, 1, 2, 3
  // t1 = 4, 5, 6, 7
  // w0 = t0 + t1
  // w1 = t0 - t1
  const T w0 = input[0]  + input[4];
  const T w7 = input[1]  + input[5];
  const T w2 = input[2]  + input[6];
  const T w9 = input[3]  + input[7];

  const T w1 = input[0]  -input[4];
  const T w8 = input[1]  -input[5];
  const T w3 = input[2]  -input[6];
  const T w10 = input[3]  -input[7];

  const T w4 = w0  + w2 ;
  const T w5 = w0  + -w2 ;
  const std::complex<T> w6 =  {w1   ,   -w3};

  const T w11 = w7  + w9 ;
  const T w12 = w7  + -w9 ;
  const std::complex<T> w13 =  {w8   ,   -w10};

  std::complex<T> f = std::complex<T>(kWeight2, -kWeight2) * w13;
  output[0] = w4  + w11 ;
  output[4] = w4  + -w11 ;
  output[1] = w6  +  f;
  output[2] =  {w5   ,   -w12};
  output[3] = std::complex<T>(w6.real()-f.real(), f.imag()-w6.imag());
  // kWeight2, kWeight2) * std::conj(w13);
}

// Num expressions: 63
// Weights: 5
template <typename T>
inline void dft_16(T* input, std::complex<T>* output) {
  const T kWeight2 = load<T>(0.707107);
  const T kWeight3 = load<T>(0.92388);
  const T kWeight4 = load<T>(0.382683);

  const T w0 = input[0]  + input[8] ;
  const T w19 = input[1]  + input[9] ;
  const T w7 = input[2]  + input[10] ;
  const T w26 = input[3]  + input[11] ;

  const T w1 = input[0]  + -input[8] ;
  const T w20 = input[1]  + -input[9] ;
  const T w8 = input[2]  + -input[10] ;
  const T w27 = input[3]  + -input[11] ;

  const T w2 = input[4]  + input[12] ;
  const T w21 = input[5]  + input[13] ;
  const T w9 = input[6]  + input[14] ;
  const T w28 = input[7]  + input[15] ;

  const T w3 = input[4]  + -input[12] ;
  const T w22 = input[5]  + -input[13] ;
  const T w10 = input[6]  + -input[14] ;
  const T w29 = input[7]  + -input[15] ;

  const T w4 = w0  + w2 ;
  const T w5 = w0  + -w2 ;
  const std::complex<T> w6 =  {w1   ,   -w3};


  const T w11 = w7  + w9 ;
  const T w12 = w7  + -w9 ;
  const std::complex<T> w13 =  {w8   ,   -w10};
  const T w14 = w4  + w11 ;
  const T w15 = w4  + -w11 ;

  T f1 = kWeight2 * w13.real();
  T f2 = kWeight2 * w13.imag();
  const std::complex<T> c13 {f1 + f2, f2 - f1}; // kWeight2 * w13.real(), = std::complex<T>( kWeight2, -kWeight2) * w13;
  const std::complex<T> w16 = w6  + c13;
  const std::complex<T> w17 =  {w5   ,   -w12};
  const std::complex<T> w18 = std::conj(w6 - c13);//  -  std::complex<T>(c13.real(), -kWeight2, -kWeight2) * std::conj(w13);


  const T w23 = w19  + w21 ;
  const T w24 = w19  + -w21 ;
  const std::complex<T> w25 =  {w20   ,   -w22};

  const T w30 = w26  + w28 ;
  const T w31 = w26  + -w28 ;
  const std::complex<T> w32 =  {w27   ,   -w29};
  const T w33 = w23  + w30 ;
  const T w34 = w23  + -w30 ;

  f1 = kWeight2 * w32.real();
  f2 = kWeight2 * w32.imag();

  const std::complex<T> c32 = {f1 + f2, f2 - f1}; // std::complex<T>( kWeight2, -kWeight2) * w32;
  const std::complex<T> w35 = w25  + c32;
  const std::complex<T> w36 =  {w24   ,   -w31};
  const std::complex<T> w37 = std::conj(w25 - c32);

  f1 = kWeight2 * w36.real();
  f2 = kWeight2 * w36.imag();

  const std::complex<T> c36(f1 + f2, f2 - f1); // = std::complex<T>( kWeight2, -kWeight2) * w36;
  const std::complex<T> c35 = std::complex<T>( kWeight3, -kWeight4) * w35;
  const std::complex<T> c37 = std::complex<T>( kWeight4, -kWeight3) * w37;

  output[0] = w14  + w33;
  output[8] = w14  + -w33;
  output[1] = w16  + c35; // std::complex<float>( kWeight3, -kWeight4) * w35;
  output[2] = w17  + c36; // std::complex<float>( kWeight2, -kWeight2) * w36;
  output[3] = w18  + c37;
  output[4] =  {w15   ,   -w34};
  output[5] = std::conj(w18 - c37);//  +  std::complex<float>(-kWeight4, -kWeight3) * std::conj(w37);
  output[6] = std::conj(w17 - c36);//  +  std::complex<float>(-kWeight2, -kWeight2) * std::conj(w36);
  output[7] = std::conj(w16 - c35);//  +  std::complex<float>(-kWeight3, -kWeight4) * std::conj(w35);
}
// Num expressions: 143
// Weights: 9
void dft_32(float* input, std::complex<float>* output) {
  const float kWeight2 = 0.707107;
  const float kWeight3 = 0.92388;
  const float kWeight4 = 0.382683;
  const float kWeight5 = 0.980785;
  const float kWeight6 = 0.19509;
  const float kWeight7 = 0.83147;
  const float kWeight8 = 0.55557;
  const float w0 = input[0]  + input[16] ;
  const float w1 = input[0]  + -input[16] ;
  const float w2 = input[8]  + input[24] ;
  const float w3 = input[8]  + -input[24] ;
  const float w4 = w0  + w2 ;
  const float w5 = w0  + -w2 ;
  const std::complex<float> w6 =  {w1   ,   -w3};
  const float w7 = input[4]  + input[20] ;
  const float w8 = input[4]  + -input[20] ;
  const float w9 = input[12]  + input[28] ;
  const float w10 = input[12]  + -input[28] ;
  const float w11 = w7  + w9 ;
  const float w12 = w7  + -w9 ;
  const std::complex<float> w13 =  {w8   ,   -w10};
  const float w14 = w4  + w11 ;
  const float w15 = w4  + -w11 ;
  const std::complex<float> w16 = w6  +  std::complex<float>( kWeight2, -kWeight2) * w13;
  const std::complex<float> w17 =  {w5   ,   -w12};
  const std::complex<float> w18 = std::conj(w6)  +  std::complex<float>(-kWeight2, -kWeight2) * std::conj(w13);
  const float w19 = input[2]  + input[18] ;
  const float w20 = input[2]  + -input[18] ;
  const float w21 = input[10]  + input[26] ;
  const float w22 = input[10]  + -input[26] ;
  const float w23 = w19  + w21 ;
  const float w24 = w19  + -w21 ;
  const std::complex<float> w25 =  {w20   ,   -w22};
  const float w26 = input[6]  + input[22] ;
  const float w27 = input[6]  + -input[22] ;
  const float w28 = input[14]  + input[30] ;
  const float w29 = input[14]  + -input[30] ;
  const float w30 = w26  + w28 ;
  const float w31 = w26  + -w28 ;
  const std::complex<float> w32 =  {w27   ,   -w29};
  const float w33 = w23  + w30 ;
  const float w34 = w23  + -w30 ;
  const std::complex<float> w35 = w25  +  std::complex<float>( kWeight2, -kWeight2) * w32;
  const std::complex<float> w36 =  {w24   ,   -w31};
  const std::complex<float> w37 = std::conj(w25)  +  std::complex<float>(-kWeight2, -kWeight2) * std::conj(w32);
  const float w38 = w14  + w33 ;
  const float w39 = w14  + -w33 ;
  const std::complex<float> w40 = w16  +  std::complex<float>( kWeight3, -kWeight4) * w35;
  const std::complex<float> w41 = w17  +  std::complex<float>( kWeight2, -kWeight2) * w36;
  const std::complex<float> w42 = w18  +  std::complex<float>( kWeight4, -kWeight3) * w37;
  const std::complex<float> w43 =  {w15   ,   -w34};
  const std::complex<float> w44 = std::conj(w18)  +  std::complex<float>(-kWeight4, -kWeight3) * std::conj(w37);
  const std::complex<float> w45 = std::conj(w17)  +  std::complex<float>(-kWeight2, -kWeight2) * std::conj(w36);
  const std::complex<float> w46 = std::conj(w16)  +  std::complex<float>(-kWeight3, -kWeight4) * std::conj(w35);
  const float w47 = input[1]  + input[17] ;
  const float w48 = input[1]  + -input[17] ;
  const float w49 = input[9]  + input[25] ;
  const float w50 = input[9]  + -input[25] ;
  const float w51 = w47  + w49 ;
  const float w52 = w47  + -w49 ;
  const std::complex<float> w53 =  {w48   ,   -w50};
  const float w54 = input[5]  + input[21] ;
  const float w55 = input[5]  + -input[21] ;
  const float w56 = input[13]  + input[29] ;
  const float w57 = input[13]  + -input[29] ;
  const float w58 = w54  + w56 ;
  const float w59 = w54  + -w56 ;
  const std::complex<float> w60 =  {w55   ,   -w57};
  const float w61 = w51  + w58 ;
  const float w62 = w51  + -w58 ;
  const std::complex<float> w63 = w53  +  std::complex<float>( kWeight2, -kWeight2) * w60;
  const std::complex<float> w64 =  {w52   ,   -w59};
  const std::complex<float> w65 = std::conj(w53)  +  std::complex<float>(-kWeight2, -kWeight2) * std::conj(w60);
  const float w66 = input[3]  + input[19] ;
  const float w67 = input[3]  + -input[19] ;
  const float w68 = input[11]  + input[27] ;
  const float w69 = input[11]  + -input[27] ;
  const float w70 = w66  + w68 ;
  const float w71 = w66  + -w68 ;
  const std::complex<float> w72 =  {w67   ,   -w69};
  const float w73 = input[7]  + input[23] ;
  const float w74 = input[7]  + -input[23] ;
  const float w75 = input[15]  + input[31] ;
  const float w76 = input[15]  + -input[31] ;
  const float w77 = w73  + w75 ;
  const float w78 = w73  + -w75 ;
  const std::complex<float> w79 =  {w74   ,   -w76};
  const float w80 = w70  + w77 ;
  const float w81 = w70  + -w77 ;
  const std::complex<float> w82 = w72  +  std::complex<float>( kWeight2, -kWeight2) * w79;
  const std::complex<float> w83 =  {w71   ,   -w78};
  const std::complex<float> w84 = std::conj(w72)  +  std::complex<float>(-kWeight2, -kWeight2) * std::conj(w79);
  const float w85 = w61  + w80 ;
  const float w86 = w61  + -w80 ;
  const std::complex<float> w87 = w63  +  std::complex<float>( kWeight3, -kWeight4) * w82;
  const std::complex<float> w88 = w64  +  std::complex<float>( kWeight2, -kWeight2) * w83;
  const std::complex<float> w89 = w65  +  std::complex<float>( kWeight4, -kWeight3) * w84;
  const std::complex<float> w90 =  {w62   ,   -w81};
  const std::complex<float> w91 = std::conj(w65)  +  std::complex<float>(-kWeight4, -kWeight3) * std::conj(w84);
  const std::complex<float> w92 = std::conj(w64)  +  std::complex<float>(-kWeight2, -kWeight2) * std::conj(w83);
  const std::complex<float> w93 = std::conj(w63)  +  std::complex<float>(-kWeight3, -kWeight4) * std::conj(w82);
  output[0] = w38  + w85 ;
  output[16] = w38  + -w85 ;
  output[1] = w40  +  std::complex<float>( kWeight5, -kWeight6) * w87;
  output[2] = w41  +  std::complex<float>( kWeight3, -kWeight4) * w88;
  output[3] = w42  +  std::complex<float>( kWeight7, -kWeight8) * w89;
  output[4] = w43  +  std::complex<float>( kWeight2, -kWeight2) * w90;
  output[5] = w44  +  std::complex<float>( kWeight8, -kWeight7) * w91;
  output[6] = w45  +  std::complex<float>( kWeight4, -kWeight3) * w92;
  output[7] = w46  +  std::complex<float>( kWeight6, -kWeight5) * w93;
  output[8] =  {w39   ,   -w86};
  output[9] = std::conj(w46)  +  std::complex<float>(-kWeight6, -kWeight5) * std::conj(w93);
  output[10] = std::conj(w45)  +  std::complex<float>(-kWeight4, -kWeight3) * std::conj(w92);
  output[11] = std::conj(w44)  +  std::complex<float>(-kWeight8, -kWeight7) * std::conj(w91);
  output[12] = std::conj(w43)  +  std::complex<float>(-kWeight2, -kWeight2) * std::conj(w90);
  output[13] = std::conj(w42)  +  std::complex<float>(-kWeight7, -kWeight8) * std::conj(w89);
  output[14] = std::conj(w41)  +  std::complex<float>(-kWeight3, -kWeight4) * std::conj(w88);
  output[15] = std::conj(w40)  +  std::complex<float>(-kWeight5, -kWeight6) * std::conj(w87);
}
