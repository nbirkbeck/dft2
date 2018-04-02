template <typename T, typename I = float>
void idft_2_compact(const I* input, I* output, int stride = 1) {
  const T i0 = SimdHelper<T, I>::load(input + 0 * stride);
  const T i1 = SimdHelper<T, I>::load(input + 1 * stride);
  SimdHelper<T, I>::store(output + 0 * stride, i0 + i1);
  SimdHelper<T, I>::store(output + 1 * stride, i0 + -i1);
}
template <typename T, typename I = float>
void idft_4_compact(const I* input, I* output, int stride = 1) {
  const T i0 = SimdHelper<T, I>::load(input + 0 * stride);
  const T i1 = SimdHelper<T, I>::load(input + 1 * stride);
  const T i2 = SimdHelper<T, I>::load(input + 2 * stride);
  const T i3 = SimdHelper<T, I>::load(input + 3 * stride);
  const T w2 = i0 + i2;
  const T w3 = i0 + -i2;
  const T w4 = i1 + i1;
  const T w5 = -i3 + -i3;
  SimdHelper<T, I>::store(output + 0 * stride, w2 + w4);
  SimdHelper<T, I>::store(output + 1 * stride, w3 + w5);
  SimdHelper<T, I>::store(output + 2 * stride, w2 + -w4);
  SimdHelper<T, I>::store(output + 3 * stride, w3 + -w5);
}
template <typename T, typename I = float>
void idft_8_compact(const I* input, I* output, int stride = 1) {
  const T kWeight2 = SimdHelper<T, I>::constant(0.707107);
  const T i0 = SimdHelper<T, I>::load(input + 0 * stride);
  const T i1 = SimdHelper<T, I>::load(input + 1 * stride);
  const T i2 = SimdHelper<T, I>::load(input + 2 * stride);
  const T i3 = SimdHelper<T, I>::load(input + 3 * stride);
  const T i4 = SimdHelper<T, I>::load(input + 4 * stride);
  const T i5 = SimdHelper<T, I>::load(input + 5 * stride);
  const T i6 = SimdHelper<T, I>::load(input + 6 * stride);
  const T i7 = SimdHelper<T, I>::load(input + 7 * stride);
  const T w0[2] = {i1, i5};
  const T w1[2] = {i2, i6};
  const T w2[2] = {i3, i7};
  const T w3[2] = {i3, -i7};
  const T w4[2] = {i2, -i6};
  const T w5[2] = {i1, -i5};
  const T w6 = i0 + i4;
  const T w7 = i0 + -i4;
  const T w8[2] = {w4[0] + w1[0], w4[1] + w1[1]};
  const T w9[2] = {w4[0] + -w1[0], w4[1] + -w1[1]};
  const T w10[2] = {w6 + w8[0], w8[1]};
  const T w11[2] = {w6 + -w8[0], -w8[1]};
  const T w12[2] = {w7 + w9[1], -w9[0]};
  const T w13[2] = {w7 + -w9[1], w9[0]};
  const T w14[2] = {w5[0] + w2[0], w5[1] + w2[1]};
  const T w15[2] = {w5[0] + -w2[0], w5[1] + -w2[1]};
  const T w16[2] = {w3[0] + w0[0], w3[1] + w0[1]};
  const T w17[2] = {w3[0] + -w0[0], w3[1] + -w0[1]};
  const T w18[2] = {w14[0] + w16[0], w14[1] + w16[1]};
  const T w19[2] = {w14[0] + -w16[0], w14[1] + -w16[1]};
  const T w20[2] = {w15[0] + w17[1], w15[1] + -w17[0]};
  const T w21[2] = {w15[0] + -w17[1], w15[1] + w17[0]};
  SimdHelper<T, I>::store(output + 0 * stride, w10[0] + w18[0]);
  SimdHelper<T, I>::store(output + 1 * stride,
                          w12[0] + (kWeight2 * w20[0] - -kWeight2 * w20[1]));
  SimdHelper<T, I>::store(output + 2 * stride, w11[0] + w19[1]);
  SimdHelper<T, I>::store(output + 3 * stride,
                          w13[0] + (-kWeight2 * w21[0] - -kWeight2 * w21[1]));
  SimdHelper<T, I>::store(output + 4 * stride, w10[0] + -w18[0]);
  SimdHelper<T, I>::store(output + 5 * stride,
                          w12[0] + (-kWeight2 * w20[0] - kWeight2 * w20[1]));
  SimdHelper<T, I>::store(output + 6 * stride, w11[0] + -w19[1]);
  SimdHelper<T, I>::store(output + 7 * stride,
                          w13[0] + (kWeight2 * w21[0] - kWeight2 * w21[1]));
}
template <typename T, typename I = float>
void idft_16_compact(const I* input, I* output, int stride = 1) {
  const T kWeight2 = SimdHelper<T, I>::constant(0.707107);
  const T kWeight3 = SimdHelper<T, I>::constant(0.92388);
  const T kWeight4 = SimdHelper<T, I>::constant(0.382683);
  const T i0 = SimdHelper<T, I>::load(input + 0 * stride);
  const T i1 = SimdHelper<T, I>::load(input + 1 * stride);
  const T i2 = SimdHelper<T, I>::load(input + 2 * stride);
  const T i3 = SimdHelper<T, I>::load(input + 3 * stride);
  const T i4 = SimdHelper<T, I>::load(input + 4 * stride);
  const T i5 = SimdHelper<T, I>::load(input + 5 * stride);
  const T i6 = SimdHelper<T, I>::load(input + 6 * stride);
  const T i7 = SimdHelper<T, I>::load(input + 7 * stride);
  const T i8 = SimdHelper<T, I>::load(input + 8 * stride);
  const T i9 = SimdHelper<T, I>::load(input + 9 * stride);
  const T i10 = SimdHelper<T, I>::load(input + 10 * stride);
  const T i11 = SimdHelper<T, I>::load(input + 11 * stride);
  const T i12 = SimdHelper<T, I>::load(input + 12 * stride);
  const T i13 = SimdHelper<T, I>::load(input + 13 * stride);
  const T i14 = SimdHelper<T, I>::load(input + 14 * stride);
  const T i15 = SimdHelper<T, I>::load(input + 15 * stride);
  const T w0[2] = {i1, i9};
  const T w1[2] = {i2, i10};
  const T w2[2] = {i3, i11};
  const T w3[2] = {i4, i12};
  const T w4[2] = {i5, i13};
  const T w5[2] = {i6, i14};
  const T w6[2] = {i7, i15};
  const T w7[2] = {i7, -i15};
  const T w8[2] = {i6, -i14};
  const T w9[2] = {i5, -i13};
  const T w10[2] = {i4, -i12};
  const T w11[2] = {i3, -i11};
  const T w12[2] = {i2, -i10};
  const T w13[2] = {i1, -i9};
  const T w14 = i0 + i8;
  const T w15 = i0 + -i8;
  const T w16[2] = {w10[0] + w3[0], w10[1] + w3[1]};
  const T w17[2] = {w10[0] + -w3[0], w10[1] + -w3[1]};
  const T w18[2] = {w14 + w16[0], w16[1]};
  const T w19[2] = {w14 + -w16[0], -w16[1]};
  const T w20[2] = {w15 + w17[1], -w17[0]};
  const T w21[2] = {w15 + -w17[1], w17[0]};
  const T w22[2] = {w12[0] + w5[0], w12[1] + w5[1]};
  const T w23[2] = {w12[0] + -w5[0], w12[1] + -w5[1]};
  const T w24[2] = {w8[0] + w1[0], w8[1] + w1[1]};
  const T w25[2] = {w8[0] + -w1[0], w8[1] + -w1[1]};
  const T w26[2] = {w22[0] + w24[0], w22[1] + w24[1]};
  const T w27[2] = {w22[0] + -w24[0], w22[1] + -w24[1]};
  const T w28[2] = {w23[0] + w25[1], w23[1] + -w25[0]};
  const T w29[2] = {w23[0] + -w25[1], w23[1] + w25[0]};
  const T w30[2] = {w18[0] + w26[0], w18[1] + w26[1]};
  const T w31[2] = {w18[0] + -w26[0], w18[1] + -w26[1]};
  const T w32[2] = {w20[0] + (kWeight2 * w28[0] - -kWeight2 * w28[1]),
                    w20[1] + (kWeight2 * w28[1] + -kWeight2 * w28[0])};
  const T w33[2] = {w20[0] + (-kWeight2 * w28[0] - kWeight2 * w28[1]),
                    w20[1] + (-kWeight2 * w28[1] + kWeight2 * w28[0])};
  const T w34[2] = {w19[0] + w27[1], w19[1] + -w27[0]};
  const T w35[2] = {w19[0] + -w27[1], w19[1] + w27[0]};
  const T w36[2] = {w21[0] + (-kWeight2 * w29[0] - -kWeight2 * w29[1]),
                    w21[1] + (-kWeight2 * w29[1] + -kWeight2 * w29[0])};
  const T w37[2] = {w21[0] + (kWeight2 * w29[0] - kWeight2 * w29[1]),
                    w21[1] + (kWeight2 * w29[1] + kWeight2 * w29[0])};
  const T w38[2] = {w13[0] + w6[0], w13[1] + w6[1]};
  const T w39[2] = {w13[0] + -w6[0], w13[1] + -w6[1]};
  const T w40[2] = {w9[0] + w2[0], w9[1] + w2[1]};
  const T w41[2] = {w9[0] + -w2[0], w9[1] + -w2[1]};
  const T w42[2] = {w38[0] + w40[0], w38[1] + w40[1]};
  const T w43[2] = {w38[0] + -w40[0], w38[1] + -w40[1]};
  const T w44[2] = {w39[0] + w41[1], w39[1] + -w41[0]};
  const T w45[2] = {w39[0] + -w41[1], w39[1] + w41[0]};
  const T w46[2] = {w11[0] + w4[0], w11[1] + w4[1]};
  const T w47[2] = {w11[0] + -w4[0], w11[1] + -w4[1]};
  const T w48[2] = {w7[0] + w0[0], w7[1] + w0[1]};
  const T w49[2] = {w7[0] + -w0[0], w7[1] + -w0[1]};
  const T w50[2] = {w46[0] + w48[0], w46[1] + w48[1]};
  const T w51[2] = {w46[0] + -w48[0], w46[1] + -w48[1]};
  const T w52[2] = {w47[0] + w49[1], w47[1] + -w49[0]};
  const T w53[2] = {w47[0] + -w49[1], w47[1] + w49[0]};
  const T w54[2] = {w42[0] + w50[0], w42[1] + w50[1]};
  const T w55[2] = {w42[0] + -w50[0], w42[1] + -w50[1]};
  const T w56[2] = {w44[0] + (kWeight2 * w52[0] - -kWeight2 * w52[1]),
                    w44[1] + (kWeight2 * w52[1] + -kWeight2 * w52[0])};
  const T w57[2] = {w44[0] + (-kWeight2 * w52[0] - kWeight2 * w52[1]),
                    w44[1] + (-kWeight2 * w52[1] + kWeight2 * w52[0])};
  const T w58[2] = {w43[0] + w51[1], w43[1] + -w51[0]};
  const T w59[2] = {w43[0] + -w51[1], w43[1] + w51[0]};
  const T w60[2] = {w45[0] + (-kWeight2 * w53[0] - -kWeight2 * w53[1]),
                    w45[1] + (-kWeight2 * w53[1] + -kWeight2 * w53[0])};
  const T w61[2] = {w45[0] + (kWeight2 * w53[0] - kWeight2 * w53[1]),
                    w45[1] + (kWeight2 * w53[1] + kWeight2 * w53[0])};
  SimdHelper<T, I>::store(output + 0 * stride, w30[0] + w54[0]);
  SimdHelper<T, I>::store(output + 1 * stride,
                          w32[0] + (kWeight3 * w56[0] - -kWeight4 * w56[1]));
  SimdHelper<T, I>::store(output + 2 * stride,
                          w34[0] + (kWeight2 * w58[0] - -kWeight2 * w58[1]));
  SimdHelper<T, I>::store(output + 3 * stride,
                          w36[0] + (kWeight4 * w60[0] - -kWeight3 * w60[1]));
  SimdHelper<T, I>::store(output + 4 * stride, w31[0] + w55[1]);
  SimdHelper<T, I>::store(output + 5 * stride,
                          w33[0] + (-kWeight4 * w57[0] - -kWeight3 * w57[1]));
  SimdHelper<T, I>::store(output + 6 * stride,
                          w35[0] + (-kWeight2 * w59[0] - -kWeight2 * w59[1]));
  SimdHelper<T, I>::store(output + 7 * stride,
                          w37[0] + (-kWeight3 * w61[0] - -kWeight4 * w61[1]));
  SimdHelper<T, I>::store(output + 8 * stride, w30[0] + -w54[0]);
  SimdHelper<T, I>::store(output + 9 * stride,
                          w32[0] + (-kWeight3 * w56[0] - kWeight4 * w56[1]));
  SimdHelper<T, I>::store(output + 10 * stride,
                          w34[0] + (-kWeight2 * w58[0] - kWeight2 * w58[1]));
  SimdHelper<T, I>::store(output + 11 * stride,
                          w36[0] + (-kWeight4 * w60[0] - kWeight3 * w60[1]));
  SimdHelper<T, I>::store(output + 12 * stride, w31[0] + -w55[1]);
  SimdHelper<T, I>::store(output + 13 * stride,
                          w33[0] + (kWeight4 * w57[0] - kWeight3 * w57[1]));
  SimdHelper<T, I>::store(output + 14 * stride,
                          w35[0] + (kWeight2 * w59[0] - kWeight2 * w59[1]));
  SimdHelper<T, I>::store(output + 15 * stride,
                          w37[0] + (kWeight3 * w61[0] - kWeight4 * w61[1]));
}
template <typename T, typename I = float>
void idft_32_compact(const I* input, I* output, int stride = 1) {
  const T kWeight2 = SimdHelper<T, I>::constant(0.707107);
  const T kWeight3 = SimdHelper<T, I>::constant(0.92388);
  const T kWeight4 = SimdHelper<T, I>::constant(0.382683);
  const T kWeight5 = SimdHelper<T, I>::constant(0.980785);
  const T kWeight6 = SimdHelper<T, I>::constant(0.19509);
  const T kWeight7 = SimdHelper<T, I>::constant(0.83147);
  const T kWeight8 = SimdHelper<T, I>::constant(0.55557);
  const T i0 = SimdHelper<T, I>::load(input + 0 * stride);
  const T i1 = SimdHelper<T, I>::load(input + 1 * stride);
  const T i2 = SimdHelper<T, I>::load(input + 2 * stride);
  const T i3 = SimdHelper<T, I>::load(input + 3 * stride);
  const T i4 = SimdHelper<T, I>::load(input + 4 * stride);
  const T i5 = SimdHelper<T, I>::load(input + 5 * stride);
  const T i6 = SimdHelper<T, I>::load(input + 6 * stride);
  const T i7 = SimdHelper<T, I>::load(input + 7 * stride);
  const T i8 = SimdHelper<T, I>::load(input + 8 * stride);
  const T i9 = SimdHelper<T, I>::load(input + 9 * stride);
  const T i10 = SimdHelper<T, I>::load(input + 10 * stride);
  const T i11 = SimdHelper<T, I>::load(input + 11 * stride);
  const T i12 = SimdHelper<T, I>::load(input + 12 * stride);
  const T i13 = SimdHelper<T, I>::load(input + 13 * stride);
  const T i14 = SimdHelper<T, I>::load(input + 14 * stride);
  const T i15 = SimdHelper<T, I>::load(input + 15 * stride);
  const T i16 = SimdHelper<T, I>::load(input + 16 * stride);
  const T i17 = SimdHelper<T, I>::load(input + 17 * stride);
  const T i18 = SimdHelper<T, I>::load(input + 18 * stride);
  const T i19 = SimdHelper<T, I>::load(input + 19 * stride);
  const T i20 = SimdHelper<T, I>::load(input + 20 * stride);
  const T i21 = SimdHelper<T, I>::load(input + 21 * stride);
  const T i22 = SimdHelper<T, I>::load(input + 22 * stride);
  const T i23 = SimdHelper<T, I>::load(input + 23 * stride);
  const T i24 = SimdHelper<T, I>::load(input + 24 * stride);
  const T i25 = SimdHelper<T, I>::load(input + 25 * stride);
  const T i26 = SimdHelper<T, I>::load(input + 26 * stride);
  const T i27 = SimdHelper<T, I>::load(input + 27 * stride);
  const T i28 = SimdHelper<T, I>::load(input + 28 * stride);
  const T i29 = SimdHelper<T, I>::load(input + 29 * stride);
  const T i30 = SimdHelper<T, I>::load(input + 30 * stride);
  const T i31 = SimdHelper<T, I>::load(input + 31 * stride);
  const T w0[2] = {i1, i17};
  const T w1[2] = {i2, i18};
  const T w2[2] = {i3, i19};
  const T w3[2] = {i4, i20};
  const T w4[2] = {i5, i21};
  const T w5[2] = {i6, i22};
  const T w6[2] = {i7, i23};
  const T w7[2] = {i8, i24};
  const T w8[2] = {i9, i25};
  const T w9[2] = {i10, i26};
  const T w10[2] = {i11, i27};
  const T w11[2] = {i12, i28};
  const T w12[2] = {i13, i29};
  const T w13[2] = {i14, i30};
  const T w14[2] = {i15, i31};
  const T w15[2] = {i15, -i31};
  const T w16[2] = {i14, -i30};
  const T w17[2] = {i13, -i29};
  const T w18[2] = {i12, -i28};
  const T w19[2] = {i11, -i27};
  const T w20[2] = {i10, -i26};
  const T w21[2] = {i9, -i25};
  const T w22[2] = {i8, -i24};
  const T w23[2] = {i7, -i23};
  const T w24[2] = {i6, -i22};
  const T w25[2] = {i5, -i21};
  const T w26[2] = {i4, -i20};
  const T w27[2] = {i3, -i19};
  const T w28[2] = {i2, -i18};
  const T w29[2] = {i1, -i17};
  const T w30 = i0 + i16;
  const T w31 = i0 + -i16;
  const T w32[2] = {w22[0] + w7[0], w22[1] + w7[1]};
  const T w33[2] = {w22[0] + -w7[0], w22[1] + -w7[1]};
  const T w34[2] = {w30 + w32[0], w32[1]};
  const T w35[2] = {w30 + -w32[0], -w32[1]};
  const T w36[2] = {w31 + w33[1], -w33[0]};
  const T w37[2] = {w31 + -w33[1], w33[0]};
  const T w38[2] = {w26[0] + w11[0], w26[1] + w11[1]};
  const T w39[2] = {w26[0] + -w11[0], w26[1] + -w11[1]};
  const T w40[2] = {w18[0] + w3[0], w18[1] + w3[1]};
  const T w41[2] = {w18[0] + -w3[0], w18[1] + -w3[1]};
  const T w42[2] = {w38[0] + w40[0], w38[1] + w40[1]};
  const T w43[2] = {w38[0] + -w40[0], w38[1] + -w40[1]};
  const T w44[2] = {w39[0] + w41[1], w39[1] + -w41[0]};
  const T w45[2] = {w39[0] + -w41[1], w39[1] + w41[0]};
  const T w46[2] = {w34[0] + w42[0], w34[1] + w42[1]};
  const T w47[2] = {w34[0] + -w42[0], w34[1] + -w42[1]};
  const T w48[2] = {w36[0] + (kWeight2 * w44[0] - -kWeight2 * w44[1]),
                    w36[1] + (kWeight2 * w44[1] + -kWeight2 * w44[0])};
  const T w49[2] = {w36[0] + (-kWeight2 * w44[0] - kWeight2 * w44[1]),
                    w36[1] + (-kWeight2 * w44[1] + kWeight2 * w44[0])};
  const T w50[2] = {w35[0] + w43[1], w35[1] + -w43[0]};
  const T w51[2] = {w35[0] + -w43[1], w35[1] + w43[0]};
  const T w52[2] = {w37[0] + (-kWeight2 * w45[0] - -kWeight2 * w45[1]),
                    w37[1] + (-kWeight2 * w45[1] + -kWeight2 * w45[0])};
  const T w53[2] = {w37[0] + (kWeight2 * w45[0] - kWeight2 * w45[1]),
                    w37[1] + (kWeight2 * w45[1] + kWeight2 * w45[0])};
  const T w54[2] = {w28[0] + w13[0], w28[1] + w13[1]};
  const T w55[2] = {w28[0] + -w13[0], w28[1] + -w13[1]};
  const T w56[2] = {w20[0] + w5[0], w20[1] + w5[1]};
  const T w57[2] = {w20[0] + -w5[0], w20[1] + -w5[1]};
  const T w58[2] = {w54[0] + w56[0], w54[1] + w56[1]};
  const T w59[2] = {w54[0] + -w56[0], w54[1] + -w56[1]};
  const T w60[2] = {w55[0] + w57[1], w55[1] + -w57[0]};
  const T w61[2] = {w55[0] + -w57[1], w55[1] + w57[0]};
  const T w62[2] = {w24[0] + w9[0], w24[1] + w9[1]};
  const T w63[2] = {w24[0] + -w9[0], w24[1] + -w9[1]};
  const T w64[2] = {w16[0] + w1[0], w16[1] + w1[1]};
  const T w65[2] = {w16[0] + -w1[0], w16[1] + -w1[1]};
  const T w66[2] = {w62[0] + w64[0], w62[1] + w64[1]};
  const T w67[2] = {w62[0] + -w64[0], w62[1] + -w64[1]};
  const T w68[2] = {w63[0] + w65[1], w63[1] + -w65[0]};
  const T w69[2] = {w63[0] + -w65[1], w63[1] + w65[0]};
  const T w70[2] = {w58[0] + w66[0], w58[1] + w66[1]};
  const T w71[2] = {w58[0] + -w66[0], w58[1] + -w66[1]};
  const T w72[2] = {w60[0] + (kWeight2 * w68[0] - -kWeight2 * w68[1]),
                    w60[1] + (kWeight2 * w68[1] + -kWeight2 * w68[0])};
  const T w73[2] = {w60[0] + (-kWeight2 * w68[0] - kWeight2 * w68[1]),
                    w60[1] + (-kWeight2 * w68[1] + kWeight2 * w68[0])};
  const T w74[2] = {w59[0] + w67[1], w59[1] + -w67[0]};
  const T w75[2] = {w59[0] + -w67[1], w59[1] + w67[0]};
  const T w76[2] = {w61[0] + (-kWeight2 * w69[0] - -kWeight2 * w69[1]),
                    w61[1] + (-kWeight2 * w69[1] + -kWeight2 * w69[0])};
  const T w77[2] = {w61[0] + (kWeight2 * w69[0] - kWeight2 * w69[1]),
                    w61[1] + (kWeight2 * w69[1] + kWeight2 * w69[0])};
  const T w78[2] = {w46[0] + w70[0], w46[1] + w70[1]};
  const T w79[2] = {w46[0] + -w70[0], w46[1] + -w70[1]};
  const T w80[2] = {w48[0] + (kWeight3 * w72[0] - -kWeight4 * w72[1]),
                    w48[1] + (kWeight3 * w72[1] + -kWeight4 * w72[0])};
  const T w81[2] = {w48[0] + (-kWeight3 * w72[0] - kWeight4 * w72[1]),
                    w48[1] + (-kWeight3 * w72[1] + kWeight4 * w72[0])};
  const T w82[2] = {w50[0] + (kWeight2 * w74[0] - -kWeight2 * w74[1]),
                    w50[1] + (kWeight2 * w74[1] + -kWeight2 * w74[0])};
  const T w83[2] = {w50[0] + (-kWeight2 * w74[0] - kWeight2 * w74[1]),
                    w50[1] + (-kWeight2 * w74[1] + kWeight2 * w74[0])};
  const T w84[2] = {w52[0] + (kWeight4 * w76[0] - -kWeight3 * w76[1]),
                    w52[1] + (kWeight4 * w76[1] + -kWeight3 * w76[0])};
  const T w85[2] = {w52[0] + (-kWeight4 * w76[0] - kWeight3 * w76[1]),
                    w52[1] + (-kWeight4 * w76[1] + kWeight3 * w76[0])};
  const T w86[2] = {w47[0] + w71[1], w47[1] + -w71[0]};
  const T w87[2] = {w47[0] + -w71[1], w47[1] + w71[0]};
  const T w88[2] = {w49[0] + (-kWeight4 * w73[0] - -kWeight3 * w73[1]),
                    w49[1] + (-kWeight4 * w73[1] + -kWeight3 * w73[0])};
  const T w89[2] = {w49[0] + (kWeight4 * w73[0] - kWeight3 * w73[1]),
                    w49[1] + (kWeight4 * w73[1] + kWeight3 * w73[0])};
  const T w90[2] = {w51[0] + (-kWeight2 * w75[0] - -kWeight2 * w75[1]),
                    w51[1] + (-kWeight2 * w75[1] + -kWeight2 * w75[0])};
  const T w91[2] = {w51[0] + (kWeight2 * w75[0] - kWeight2 * w75[1]),
                    w51[1] + (kWeight2 * w75[1] + kWeight2 * w75[0])};
  const T w92[2] = {w53[0] + (-kWeight3 * w77[0] - -kWeight4 * w77[1]),
                    w53[1] + (-kWeight3 * w77[1] + -kWeight4 * w77[0])};
  const T w93[2] = {w53[0] + (kWeight3 * w77[0] - kWeight4 * w77[1]),
                    w53[1] + (kWeight3 * w77[1] + kWeight4 * w77[0])};
  const T w94[2] = {w29[0] + w14[0], w29[1] + w14[1]};
  const T w95[2] = {w29[0] + -w14[0], w29[1] + -w14[1]};
  const T w96[2] = {w21[0] + w6[0], w21[1] + w6[1]};
  const T w97[2] = {w21[0] + -w6[0], w21[1] + -w6[1]};
  const T w98[2] = {w94[0] + w96[0], w94[1] + w96[1]};
  const T w99[2] = {w94[0] + -w96[0], w94[1] + -w96[1]};
  const T w100[2] = {w95[0] + w97[1], w95[1] + -w97[0]};
  const T w101[2] = {w95[0] + -w97[1], w95[1] + w97[0]};
  const T w102[2] = {w25[0] + w10[0], w25[1] + w10[1]};
  const T w103[2] = {w25[0] + -w10[0], w25[1] + -w10[1]};
  const T w104[2] = {w17[0] + w2[0], w17[1] + w2[1]};
  const T w105[2] = {w17[0] + -w2[0], w17[1] + -w2[1]};
  const T w106[2] = {w102[0] + w104[0], w102[1] + w104[1]};
  const T w107[2] = {w102[0] + -w104[0], w102[1] + -w104[1]};
  const T w108[2] = {w103[0] + w105[1], w103[1] + -w105[0]};
  const T w109[2] = {w103[0] + -w105[1], w103[1] + w105[0]};
  const T w110[2] = {w98[0] + w106[0], w98[1] + w106[1]};
  const T w111[2] = {w98[0] + -w106[0], w98[1] + -w106[1]};
  const T w112[2] = {w100[0] + (kWeight2 * w108[0] - -kWeight2 * w108[1]),
                     w100[1] + (kWeight2 * w108[1] + -kWeight2 * w108[0])};
  const T w113[2] = {w100[0] + (-kWeight2 * w108[0] - kWeight2 * w108[1]),
                     w100[1] + (-kWeight2 * w108[1] + kWeight2 * w108[0])};
  const T w114[2] = {w99[0] + w107[1], w99[1] + -w107[0]};
  const T w115[2] = {w99[0] + -w107[1], w99[1] + w107[0]};
  const T w116[2] = {w101[0] + (-kWeight2 * w109[0] - -kWeight2 * w109[1]),
                     w101[1] + (-kWeight2 * w109[1] + -kWeight2 * w109[0])};
  const T w117[2] = {w101[0] + (kWeight2 * w109[0] - kWeight2 * w109[1]),
                     w101[1] + (kWeight2 * w109[1] + kWeight2 * w109[0])};
  const T w118[2] = {w27[0] + w12[0], w27[1] + w12[1]};
  const T w119[2] = {w27[0] + -w12[0], w27[1] + -w12[1]};
  const T w120[2] = {w19[0] + w4[0], w19[1] + w4[1]};
  const T w121[2] = {w19[0] + -w4[0], w19[1] + -w4[1]};
  const T w122[2] = {w118[0] + w120[0], w118[1] + w120[1]};
  const T w123[2] = {w118[0] + -w120[0], w118[1] + -w120[1]};
  const T w124[2] = {w119[0] + w121[1], w119[1] + -w121[0]};
  const T w125[2] = {w119[0] + -w121[1], w119[1] + w121[0]};
  const T w126[2] = {w23[0] + w8[0], w23[1] + w8[1]};
  const T w127[2] = {w23[0] + -w8[0], w23[1] + -w8[1]};
  const T w128[2] = {w15[0] + w0[0], w15[1] + w0[1]};
  const T w129[2] = {w15[0] + -w0[0], w15[1] + -w0[1]};
  const T w130[2] = {w126[0] + w128[0], w126[1] + w128[1]};
  const T w131[2] = {w126[0] + -w128[0], w126[1] + -w128[1]};
  const T w132[2] = {w127[0] + w129[1], w127[1] + -w129[0]};
  const T w133[2] = {w127[0] + -w129[1], w127[1] + w129[0]};
  const T w134[2] = {w122[0] + w130[0], w122[1] + w130[1]};
  const T w135[2] = {w122[0] + -w130[0], w122[1] + -w130[1]};
  const T w136[2] = {w124[0] + (kWeight2 * w132[0] - -kWeight2 * w132[1]),
                     w124[1] + (kWeight2 * w132[1] + -kWeight2 * w132[0])};
  const T w137[2] = {w124[0] + (-kWeight2 * w132[0] - kWeight2 * w132[1]),
                     w124[1] + (-kWeight2 * w132[1] + kWeight2 * w132[0])};
  const T w138[2] = {w123[0] + w131[1], w123[1] + -w131[0]};
  const T w139[2] = {w123[0] + -w131[1], w123[1] + w131[0]};
  const T w140[2] = {w125[0] + (-kWeight2 * w133[0] - -kWeight2 * w133[1]),
                     w125[1] + (-kWeight2 * w133[1] + -kWeight2 * w133[0])};
  const T w141[2] = {w125[0] + (kWeight2 * w133[0] - kWeight2 * w133[1]),
                     w125[1] + (kWeight2 * w133[1] + kWeight2 * w133[0])};
  const T w142[2] = {w110[0] + w134[0], w110[1] + w134[1]};
  const T w143[2] = {w110[0] + -w134[0], w110[1] + -w134[1]};
  const T w144[2] = {w112[0] + (kWeight3 * w136[0] - -kWeight4 * w136[1]),
                     w112[1] + (kWeight3 * w136[1] + -kWeight4 * w136[0])};
  const T w145[2] = {w112[0] + (-kWeight3 * w136[0] - kWeight4 * w136[1]),
                     w112[1] + (-kWeight3 * w136[1] + kWeight4 * w136[0])};
  const T w146[2] = {w114[0] + (kWeight2 * w138[0] - -kWeight2 * w138[1]),
                     w114[1] + (kWeight2 * w138[1] + -kWeight2 * w138[0])};
  const T w147[2] = {w114[0] + (-kWeight2 * w138[0] - kWeight2 * w138[1]),
                     w114[1] + (-kWeight2 * w138[1] + kWeight2 * w138[0])};
  const T w148[2] = {w116[0] + (kWeight4 * w140[0] - -kWeight3 * w140[1]),
                     w116[1] + (kWeight4 * w140[1] + -kWeight3 * w140[0])};
  const T w149[2] = {w116[0] + (-kWeight4 * w140[0] - kWeight3 * w140[1]),
                     w116[1] + (-kWeight4 * w140[1] + kWeight3 * w140[0])};
  const T w150[2] = {w111[0] + w135[1], w111[1] + -w135[0]};
  const T w151[2] = {w111[0] + -w135[1], w111[1] + w135[0]};
  const T w152[2] = {w113[0] + (-kWeight4 * w137[0] - -kWeight3 * w137[1]),
                     w113[1] + (-kWeight4 * w137[1] + -kWeight3 * w137[0])};
  const T w153[2] = {w113[0] + (kWeight4 * w137[0] - kWeight3 * w137[1]),
                     w113[1] + (kWeight4 * w137[1] + kWeight3 * w137[0])};
  const T w154[2] = {w115[0] + (-kWeight2 * w139[0] - -kWeight2 * w139[1]),
                     w115[1] + (-kWeight2 * w139[1] + -kWeight2 * w139[0])};
  const T w155[2] = {w115[0] + (kWeight2 * w139[0] - kWeight2 * w139[1]),
                     w115[1] + (kWeight2 * w139[1] + kWeight2 * w139[0])};
  const T w156[2] = {w117[0] + (-kWeight3 * w141[0] - -kWeight4 * w141[1]),
                     w117[1] + (-kWeight3 * w141[1] + -kWeight4 * w141[0])};
  const T w157[2] = {w117[0] + (kWeight3 * w141[0] - kWeight4 * w141[1]),
                     w117[1] + (kWeight3 * w141[1] + kWeight4 * w141[0])};
  SimdHelper<T, I>::store(output + 0 * stride, w78[0] + w142[0]);
  SimdHelper<T, I>::store(output + 1 * stride,
                          w80[0] + (kWeight5 * w144[0] - -kWeight6 * w144[1]));
  SimdHelper<T, I>::store(output + 2 * stride,
                          w82[0] + (kWeight3 * w146[0] - -kWeight4 * w146[1]));
  SimdHelper<T, I>::store(output + 3 * stride,
                          w84[0] + (kWeight7 * w148[0] - -kWeight8 * w148[1]));
  SimdHelper<T, I>::store(output + 4 * stride,
                          w86[0] + (kWeight2 * w150[0] - -kWeight2 * w150[1]));
  SimdHelper<T, I>::store(output + 5 * stride,
                          w88[0] + (kWeight8 * w152[0] - -kWeight7 * w152[1]));
  SimdHelper<T, I>::store(output + 6 * stride,
                          w90[0] + (kWeight4 * w154[0] - -kWeight3 * w154[1]));
  SimdHelper<T, I>::store(output + 7 * stride,
                          w92[0] + (kWeight6 * w156[0] - -kWeight5 * w156[1]));
  SimdHelper<T, I>::store(output + 8 * stride, w79[0] + w143[1]);
  SimdHelper<T, I>::store(output + 9 * stride,
                          w81[0] + (-kWeight6 * w145[0] - -kWeight5 * w145[1]));
  SimdHelper<T, I>::store(output + 10 * stride,
                          w83[0] + (-kWeight4 * w147[0] - -kWeight3 * w147[1]));
  SimdHelper<T, I>::store(output + 11 * stride,
                          w85[0] + (-kWeight8 * w149[0] - -kWeight7 * w149[1]));
  SimdHelper<T, I>::store(output + 12 * stride,
                          w87[0] + (-kWeight2 * w151[0] - -kWeight2 * w151[1]));
  SimdHelper<T, I>::store(output + 13 * stride,
                          w89[0] + (-kWeight7 * w153[0] - -kWeight8 * w153[1]));
  SimdHelper<T, I>::store(output + 14 * stride,
                          w91[0] + (-kWeight3 * w155[0] - -kWeight4 * w155[1]));
  SimdHelper<T, I>::store(output + 15 * stride,
                          w93[0] + (-kWeight5 * w157[0] - -kWeight6 * w157[1]));
  SimdHelper<T, I>::store(output + 16 * stride, w78[0] + -w142[0]);
  SimdHelper<T, I>::store(output + 17 * stride,
                          w80[0] + (-kWeight5 * w144[0] - kWeight6 * w144[1]));
  SimdHelper<T, I>::store(output + 18 * stride,
                          w82[0] + (-kWeight3 * w146[0] - kWeight4 * w146[1]));
  SimdHelper<T, I>::store(output + 19 * stride,
                          w84[0] + (-kWeight7 * w148[0] - kWeight8 * w148[1]));
  SimdHelper<T, I>::store(output + 20 * stride,
                          w86[0] + (-kWeight2 * w150[0] - kWeight2 * w150[1]));
  SimdHelper<T, I>::store(output + 21 * stride,
                          w88[0] + (-kWeight8 * w152[0] - kWeight7 * w152[1]));
  SimdHelper<T, I>::store(output + 22 * stride,
                          w90[0] + (-kWeight4 * w154[0] - kWeight3 * w154[1]));
  SimdHelper<T, I>::store(output + 23 * stride,
                          w92[0] + (-kWeight6 * w156[0] - kWeight5 * w156[1]));
  SimdHelper<T, I>::store(output + 24 * stride, w79[0] + -w143[1]);
  SimdHelper<T, I>::store(output + 25 * stride,
                          w81[0] + (kWeight6 * w145[0] - kWeight5 * w145[1]));
  SimdHelper<T, I>::store(output + 26 * stride,
                          w83[0] + (kWeight4 * w147[0] - kWeight3 * w147[1]));
  SimdHelper<T, I>::store(output + 27 * stride,
                          w85[0] + (kWeight8 * w149[0] - kWeight7 * w149[1]));
  SimdHelper<T, I>::store(output + 28 * stride,
                          w87[0] + (kWeight2 * w151[0] - kWeight2 * w151[1]));
  SimdHelper<T, I>::store(output + 29 * stride,
                          w89[0] + (kWeight7 * w153[0] - kWeight8 * w153[1]));
  SimdHelper<T, I>::store(output + 30 * stride,
                          w91[0] + (kWeight3 * w155[0] - kWeight4 * w155[1]));
  SimdHelper<T, I>::store(output + 31 * stride,
                          w93[0] + (kWeight5 * w157[0] - kWeight6 * w157[1]));
}
template <typename T, typename I = float>
void idft_64_compact(const I* input, I* output, int stride = 1) {
  const T kWeight2 = SimdHelper<T, I>::constant(0.707107);
  const T kWeight3 = SimdHelper<T, I>::constant(0.92388);
  const T kWeight4 = SimdHelper<T, I>::constant(0.382683);
  const T kWeight5 = SimdHelper<T, I>::constant(0.980785);
  const T kWeight6 = SimdHelper<T, I>::constant(0.19509);
  const T kWeight7 = SimdHelper<T, I>::constant(0.83147);
  const T kWeight8 = SimdHelper<T, I>::constant(0.55557);
  const T kWeight9 = SimdHelper<T, I>::constant(0.995185);
  const T kWeight10 = SimdHelper<T, I>::constant(0.0980171);
  const T kWeight11 = SimdHelper<T, I>::constant(0.95694);
  const T kWeight12 = SimdHelper<T, I>::constant(0.290285);
  const T kWeight13 = SimdHelper<T, I>::constant(0.881921);
  const T kWeight14 = SimdHelper<T, I>::constant(0.471397);
  const T kWeight15 = SimdHelper<T, I>::constant(0.77301);
  const T kWeight16 = SimdHelper<T, I>::constant(0.634393);
  const T i0 = SimdHelper<T, I>::load(input + 0 * stride);
  const T i1 = SimdHelper<T, I>::load(input + 1 * stride);
  const T i2 = SimdHelper<T, I>::load(input + 2 * stride);
  const T i3 = SimdHelper<T, I>::load(input + 3 * stride);
  const T i4 = SimdHelper<T, I>::load(input + 4 * stride);
  const T i5 = SimdHelper<T, I>::load(input + 5 * stride);
  const T i6 = SimdHelper<T, I>::load(input + 6 * stride);
  const T i7 = SimdHelper<T, I>::load(input + 7 * stride);
  const T i8 = SimdHelper<T, I>::load(input + 8 * stride);
  const T i9 = SimdHelper<T, I>::load(input + 9 * stride);
  const T i10 = SimdHelper<T, I>::load(input + 10 * stride);
  const T i11 = SimdHelper<T, I>::load(input + 11 * stride);
  const T i12 = SimdHelper<T, I>::load(input + 12 * stride);
  const T i13 = SimdHelper<T, I>::load(input + 13 * stride);
  const T i14 = SimdHelper<T, I>::load(input + 14 * stride);
  const T i15 = SimdHelper<T, I>::load(input + 15 * stride);
  const T i16 = SimdHelper<T, I>::load(input + 16 * stride);
  const T i17 = SimdHelper<T, I>::load(input + 17 * stride);
  const T i18 = SimdHelper<T, I>::load(input + 18 * stride);
  const T i19 = SimdHelper<T, I>::load(input + 19 * stride);
  const T i20 = SimdHelper<T, I>::load(input + 20 * stride);
  const T i21 = SimdHelper<T, I>::load(input + 21 * stride);
  const T i22 = SimdHelper<T, I>::load(input + 22 * stride);
  const T i23 = SimdHelper<T, I>::load(input + 23 * stride);
  const T i24 = SimdHelper<T, I>::load(input + 24 * stride);
  const T i25 = SimdHelper<T, I>::load(input + 25 * stride);
  const T i26 = SimdHelper<T, I>::load(input + 26 * stride);
  const T i27 = SimdHelper<T, I>::load(input + 27 * stride);
  const T i28 = SimdHelper<T, I>::load(input + 28 * stride);
  const T i29 = SimdHelper<T, I>::load(input + 29 * stride);
  const T i30 = SimdHelper<T, I>::load(input + 30 * stride);
  const T i31 = SimdHelper<T, I>::load(input + 31 * stride);
  const T i32 = SimdHelper<T, I>::load(input + 32 * stride);
  const T i33 = SimdHelper<T, I>::load(input + 33 * stride);
  const T i34 = SimdHelper<T, I>::load(input + 34 * stride);
  const T i35 = SimdHelper<T, I>::load(input + 35 * stride);
  const T i36 = SimdHelper<T, I>::load(input + 36 * stride);
  const T i37 = SimdHelper<T, I>::load(input + 37 * stride);
  const T i38 = SimdHelper<T, I>::load(input + 38 * stride);
  const T i39 = SimdHelper<T, I>::load(input + 39 * stride);
  const T i40 = SimdHelper<T, I>::load(input + 40 * stride);
  const T i41 = SimdHelper<T, I>::load(input + 41 * stride);
  const T i42 = SimdHelper<T, I>::load(input + 42 * stride);
  const T i43 = SimdHelper<T, I>::load(input + 43 * stride);
  const T i44 = SimdHelper<T, I>::load(input + 44 * stride);
  const T i45 = SimdHelper<T, I>::load(input + 45 * stride);
  const T i46 = SimdHelper<T, I>::load(input + 46 * stride);
  const T i47 = SimdHelper<T, I>::load(input + 47 * stride);
  const T i48 = SimdHelper<T, I>::load(input + 48 * stride);
  const T i49 = SimdHelper<T, I>::load(input + 49 * stride);
  const T i50 = SimdHelper<T, I>::load(input + 50 * stride);
  const T i51 = SimdHelper<T, I>::load(input + 51 * stride);
  const T i52 = SimdHelper<T, I>::load(input + 52 * stride);
  const T i53 = SimdHelper<T, I>::load(input + 53 * stride);
  const T i54 = SimdHelper<T, I>::load(input + 54 * stride);
  const T i55 = SimdHelper<T, I>::load(input + 55 * stride);
  const T i56 = SimdHelper<T, I>::load(input + 56 * stride);
  const T i57 = SimdHelper<T, I>::load(input + 57 * stride);
  const T i58 = SimdHelper<T, I>::load(input + 58 * stride);
  const T i59 = SimdHelper<T, I>::load(input + 59 * stride);
  const T i60 = SimdHelper<T, I>::load(input + 60 * stride);
  const T i61 = SimdHelper<T, I>::load(input + 61 * stride);
  const T i62 = SimdHelper<T, I>::load(input + 62 * stride);
  const T i63 = SimdHelper<T, I>::load(input + 63 * stride);
  const T w0[2] = {i1, i33};
  const T w1[2] = {i2, i34};
  const T w2[2] = {i3, i35};
  const T w3[2] = {i4, i36};
  const T w4[2] = {i5, i37};
  const T w5[2] = {i6, i38};
  const T w6[2] = {i7, i39};
  const T w7[2] = {i8, i40};
  const T w8[2] = {i9, i41};
  const T w9[2] = {i10, i42};
  const T w10[2] = {i11, i43};
  const T w11[2] = {i12, i44};
  const T w12[2] = {i13, i45};
  const T w13[2] = {i14, i46};
  const T w14[2] = {i15, i47};
  const T w15[2] = {i16, i48};
  const T w16[2] = {i17, i49};
  const T w17[2] = {i18, i50};
  const T w18[2] = {i19, i51};
  const T w19[2] = {i20, i52};
  const T w20[2] = {i21, i53};
  const T w21[2] = {i22, i54};
  const T w22[2] = {i23, i55};
  const T w23[2] = {i24, i56};
  const T w24[2] = {i25, i57};
  const T w25[2] = {i26, i58};
  const T w26[2] = {i27, i59};
  const T w27[2] = {i28, i60};
  const T w28[2] = {i29, i61};
  const T w29[2] = {i30, i62};
  const T w30[2] = {i31, i63};
  const T w31[2] = {i31, -i63};
  const T w32[2] = {i30, -i62};
  const T w33[2] = {i29, -i61};
  const T w34[2] = {i28, -i60};
  const T w35[2] = {i27, -i59};
  const T w36[2] = {i26, -i58};
  const T w37[2] = {i25, -i57};
  const T w38[2] = {i24, -i56};
  const T w39[2] = {i23, -i55};
  const T w40[2] = {i22, -i54};
  const T w41[2] = {i21, -i53};
  const T w42[2] = {i20, -i52};
  const T w43[2] = {i19, -i51};
  const T w44[2] = {i18, -i50};
  const T w45[2] = {i17, -i49};
  const T w46[2] = {i16, -i48};
  const T w47[2] = {i15, -i47};
  const T w48[2] = {i14, -i46};
  const T w49[2] = {i13, -i45};
  const T w50[2] = {i12, -i44};
  const T w51[2] = {i11, -i43};
  const T w52[2] = {i10, -i42};
  const T w53[2] = {i9, -i41};
  const T w54[2] = {i8, -i40};
  const T w55[2] = {i7, -i39};
  const T w56[2] = {i6, -i38};
  const T w57[2] = {i5, -i37};
  const T w58[2] = {i4, -i36};
  const T w59[2] = {i3, -i35};
  const T w60[2] = {i2, -i34};
  const T w61[2] = {i1, -i33};
  const T w62 = i0 + i32;
  const T w63 = i0 + -i32;
  const T w64[2] = {w46[0] + w15[0], w46[1] + w15[1]};
  const T w65[2] = {w46[0] + -w15[0], w46[1] + -w15[1]};
  const T w66[2] = {w62 + w64[0], w64[1]};
  const T w67[2] = {w62 + -w64[0], -w64[1]};
  const T w68[2] = {w63 + w65[1], -w65[0]};
  const T w69[2] = {w63 + -w65[1], w65[0]};
  const T w70[2] = {w54[0] + w23[0], w54[1] + w23[1]};
  const T w71[2] = {w54[0] + -w23[0], w54[1] + -w23[1]};
  const T w72[2] = {w38[0] + w7[0], w38[1] + w7[1]};
  const T w73[2] = {w38[0] + -w7[0], w38[1] + -w7[1]};
  const T w74[2] = {w70[0] + w72[0], w70[1] + w72[1]};
  const T w75[2] = {w70[0] + -w72[0], w70[1] + -w72[1]};
  const T w76[2] = {w71[0] + w73[1], w71[1] + -w73[0]};
  const T w77[2] = {w71[0] + -w73[1], w71[1] + w73[0]};
  const T w78[2] = {w66[0] + w74[0], w66[1] + w74[1]};
  const T w79[2] = {w66[0] + -w74[0], w66[1] + -w74[1]};
  const T w80[2] = {w68[0] + (kWeight2 * w76[0] - -kWeight2 * w76[1]),
                    w68[1] + (kWeight2 * w76[1] + -kWeight2 * w76[0])};
  const T w81[2] = {w68[0] + (-kWeight2 * w76[0] - kWeight2 * w76[1]),
                    w68[1] + (-kWeight2 * w76[1] + kWeight2 * w76[0])};
  const T w82[2] = {w67[0] + w75[1], w67[1] + -w75[0]};
  const T w83[2] = {w67[0] + -w75[1], w67[1] + w75[0]};
  const T w84[2] = {w69[0] + (-kWeight2 * w77[0] - -kWeight2 * w77[1]),
                    w69[1] + (-kWeight2 * w77[1] + -kWeight2 * w77[0])};
  const T w85[2] = {w69[0] + (kWeight2 * w77[0] - kWeight2 * w77[1]),
                    w69[1] + (kWeight2 * w77[1] + kWeight2 * w77[0])};
  const T w86[2] = {w58[0] + w27[0], w58[1] + w27[1]};
  const T w87[2] = {w58[0] + -w27[0], w58[1] + -w27[1]};
  const T w88[2] = {w42[0] + w11[0], w42[1] + w11[1]};
  const T w89[2] = {w42[0] + -w11[0], w42[1] + -w11[1]};
  const T w90[2] = {w86[0] + w88[0], w86[1] + w88[1]};
  const T w91[2] = {w86[0] + -w88[0], w86[1] + -w88[1]};
  const T w92[2] = {w87[0] + w89[1], w87[1] + -w89[0]};
  const T w93[2] = {w87[0] + -w89[1], w87[1] + w89[0]};
  const T w94[2] = {w50[0] + w19[0], w50[1] + w19[1]};
  const T w95[2] = {w50[0] + -w19[0], w50[1] + -w19[1]};
  const T w96[2] = {w34[0] + w3[0], w34[1] + w3[1]};
  const T w97[2] = {w34[0] + -w3[0], w34[1] + -w3[1]};
  const T w98[2] = {w94[0] + w96[0], w94[1] + w96[1]};
  const T w99[2] = {w94[0] + -w96[0], w94[1] + -w96[1]};
  const T w100[2] = {w95[0] + w97[1], w95[1] + -w97[0]};
  const T w101[2] = {w95[0] + -w97[1], w95[1] + w97[0]};
  const T w102[2] = {w90[0] + w98[0], w90[1] + w98[1]};
  const T w103[2] = {w90[0] + -w98[0], w90[1] + -w98[1]};
  const T w104[2] = {w92[0] + (kWeight2 * w100[0] - -kWeight2 * w100[1]),
                     w92[1] + (kWeight2 * w100[1] + -kWeight2 * w100[0])};
  const T w105[2] = {w92[0] + (-kWeight2 * w100[0] - kWeight2 * w100[1]),
                     w92[1] + (-kWeight2 * w100[1] + kWeight2 * w100[0])};
  const T w106[2] = {w91[0] + w99[1], w91[1] + -w99[0]};
  const T w107[2] = {w91[0] + -w99[1], w91[1] + w99[0]};
  const T w108[2] = {w93[0] + (-kWeight2 * w101[0] - -kWeight2 * w101[1]),
                     w93[1] + (-kWeight2 * w101[1] + -kWeight2 * w101[0])};
  const T w109[2] = {w93[0] + (kWeight2 * w101[0] - kWeight2 * w101[1]),
                     w93[1] + (kWeight2 * w101[1] + kWeight2 * w101[0])};
  const T w110[2] = {w78[0] + w102[0], w78[1] + w102[1]};
  const T w111[2] = {w78[0] + -w102[0], w78[1] + -w102[1]};
  const T w112[2] = {w80[0] + (kWeight3 * w104[0] - -kWeight4 * w104[1]),
                     w80[1] + (kWeight3 * w104[1] + -kWeight4 * w104[0])};
  const T w113[2] = {w80[0] + (-kWeight3 * w104[0] - kWeight4 * w104[1]),
                     w80[1] + (-kWeight3 * w104[1] + kWeight4 * w104[0])};
  const T w114[2] = {w82[0] + (kWeight2 * w106[0] - -kWeight2 * w106[1]),
                     w82[1] + (kWeight2 * w106[1] + -kWeight2 * w106[0])};
  const T w115[2] = {w82[0] + (-kWeight2 * w106[0] - kWeight2 * w106[1]),
                     w82[1] + (-kWeight2 * w106[1] + kWeight2 * w106[0])};
  const T w116[2] = {w84[0] + (kWeight4 * w108[0] - -kWeight3 * w108[1]),
                     w84[1] + (kWeight4 * w108[1] + -kWeight3 * w108[0])};
  const T w117[2] = {w84[0] + (-kWeight4 * w108[0] - kWeight3 * w108[1]),
                     w84[1] + (-kWeight4 * w108[1] + kWeight3 * w108[0])};
  const T w118[2] = {w79[0] + w103[1], w79[1] + -w103[0]};
  const T w119[2] = {w79[0] + -w103[1], w79[1] + w103[0]};
  const T w120[2] = {w81[0] + (-kWeight4 * w105[0] - -kWeight3 * w105[1]),
                     w81[1] + (-kWeight4 * w105[1] + -kWeight3 * w105[0])};
  const T w121[2] = {w81[0] + (kWeight4 * w105[0] - kWeight3 * w105[1]),
                     w81[1] + (kWeight4 * w105[1] + kWeight3 * w105[0])};
  const T w122[2] = {w83[0] + (-kWeight2 * w107[0] - -kWeight2 * w107[1]),
                     w83[1] + (-kWeight2 * w107[1] + -kWeight2 * w107[0])};
  const T w123[2] = {w83[0] + (kWeight2 * w107[0] - kWeight2 * w107[1]),
                     w83[1] + (kWeight2 * w107[1] + kWeight2 * w107[0])};
  const T w124[2] = {w85[0] + (-kWeight3 * w109[0] - -kWeight4 * w109[1]),
                     w85[1] + (-kWeight3 * w109[1] + -kWeight4 * w109[0])};
  const T w125[2] = {w85[0] + (kWeight3 * w109[0] - kWeight4 * w109[1]),
                     w85[1] + (kWeight3 * w109[1] + kWeight4 * w109[0])};
  const T w126[2] = {w60[0] + w29[0], w60[1] + w29[1]};
  const T w127[2] = {w60[0] + -w29[0], w60[1] + -w29[1]};
  const T w128[2] = {w44[0] + w13[0], w44[1] + w13[1]};
  const T w129[2] = {w44[0] + -w13[0], w44[1] + -w13[1]};
  const T w130[2] = {w126[0] + w128[0], w126[1] + w128[1]};
  const T w131[2] = {w126[0] + -w128[0], w126[1] + -w128[1]};
  const T w132[2] = {w127[0] + w129[1], w127[1] + -w129[0]};
  const T w133[2] = {w127[0] + -w129[1], w127[1] + w129[0]};
  const T w134[2] = {w52[0] + w21[0], w52[1] + w21[1]};
  const T w135[2] = {w52[0] + -w21[0], w52[1] + -w21[1]};
  const T w136[2] = {w36[0] + w5[0], w36[1] + w5[1]};
  const T w137[2] = {w36[0] + -w5[0], w36[1] + -w5[1]};
  const T w138[2] = {w134[0] + w136[0], w134[1] + w136[1]};
  const T w139[2] = {w134[0] + -w136[0], w134[1] + -w136[1]};
  const T w140[2] = {w135[0] + w137[1], w135[1] + -w137[0]};
  const T w141[2] = {w135[0] + -w137[1], w135[1] + w137[0]};
  const T w142[2] = {w130[0] + w138[0], w130[1] + w138[1]};
  const T w143[2] = {w130[0] + -w138[0], w130[1] + -w138[1]};
  const T w144[2] = {w132[0] + (kWeight2 * w140[0] - -kWeight2 * w140[1]),
                     w132[1] + (kWeight2 * w140[1] + -kWeight2 * w140[0])};
  const T w145[2] = {w132[0] + (-kWeight2 * w140[0] - kWeight2 * w140[1]),
                     w132[1] + (-kWeight2 * w140[1] + kWeight2 * w140[0])};
  const T w146[2] = {w131[0] + w139[1], w131[1] + -w139[0]};
  const T w147[2] = {w131[0] + -w139[1], w131[1] + w139[0]};
  const T w148[2] = {w133[0] + (-kWeight2 * w141[0] - -kWeight2 * w141[1]),
                     w133[1] + (-kWeight2 * w141[1] + -kWeight2 * w141[0])};
  const T w149[2] = {w133[0] + (kWeight2 * w141[0] - kWeight2 * w141[1]),
                     w133[1] + (kWeight2 * w141[1] + kWeight2 * w141[0])};
  const T w150[2] = {w56[0] + w25[0], w56[1] + w25[1]};
  const T w151[2] = {w56[0] + -w25[0], w56[1] + -w25[1]};
  const T w152[2] = {w40[0] + w9[0], w40[1] + w9[1]};
  const T w153[2] = {w40[0] + -w9[0], w40[1] + -w9[1]};
  const T w154[2] = {w150[0] + w152[0], w150[1] + w152[1]};
  const T w155[2] = {w150[0] + -w152[0], w150[1] + -w152[1]};
  const T w156[2] = {w151[0] + w153[1], w151[1] + -w153[0]};
  const T w157[2] = {w151[0] + -w153[1], w151[1] + w153[0]};
  const T w158[2] = {w48[0] + w17[0], w48[1] + w17[1]};
  const T w159[2] = {w48[0] + -w17[0], w48[1] + -w17[1]};
  const T w160[2] = {w32[0] + w1[0], w32[1] + w1[1]};
  const T w161[2] = {w32[0] + -w1[0], w32[1] + -w1[1]};
  const T w162[2] = {w158[0] + w160[0], w158[1] + w160[1]};
  const T w163[2] = {w158[0] + -w160[0], w158[1] + -w160[1]};
  const T w164[2] = {w159[0] + w161[1], w159[1] + -w161[0]};
  const T w165[2] = {w159[0] + -w161[1], w159[1] + w161[0]};
  const T w166[2] = {w154[0] + w162[0], w154[1] + w162[1]};
  const T w167[2] = {w154[0] + -w162[0], w154[1] + -w162[1]};
  const T w168[2] = {w156[0] + (kWeight2 * w164[0] - -kWeight2 * w164[1]),
                     w156[1] + (kWeight2 * w164[1] + -kWeight2 * w164[0])};
  const T w169[2] = {w156[0] + (-kWeight2 * w164[0] - kWeight2 * w164[1]),
                     w156[1] + (-kWeight2 * w164[1] + kWeight2 * w164[0])};
  const T w170[2] = {w155[0] + w163[1], w155[1] + -w163[0]};
  const T w171[2] = {w155[0] + -w163[1], w155[1] + w163[0]};
  const T w172[2] = {w157[0] + (-kWeight2 * w165[0] - -kWeight2 * w165[1]),
                     w157[1] + (-kWeight2 * w165[1] + -kWeight2 * w165[0])};
  const T w173[2] = {w157[0] + (kWeight2 * w165[0] - kWeight2 * w165[1]),
                     w157[1] + (kWeight2 * w165[1] + kWeight2 * w165[0])};
  const T w174[2] = {w142[0] + w166[0], w142[1] + w166[1]};
  const T w175[2] = {w142[0] + -w166[0], w142[1] + -w166[1]};
  const T w176[2] = {w144[0] + (kWeight3 * w168[0] - -kWeight4 * w168[1]),
                     w144[1] + (kWeight3 * w168[1] + -kWeight4 * w168[0])};
  const T w177[2] = {w144[0] + (-kWeight3 * w168[0] - kWeight4 * w168[1]),
                     w144[1] + (-kWeight3 * w168[1] + kWeight4 * w168[0])};
  const T w178[2] = {w146[0] + (kWeight2 * w170[0] - -kWeight2 * w170[1]),
                     w146[1] + (kWeight2 * w170[1] + -kWeight2 * w170[0])};
  const T w179[2] = {w146[0] + (-kWeight2 * w170[0] - kWeight2 * w170[1]),
                     w146[1] + (-kWeight2 * w170[1] + kWeight2 * w170[0])};
  const T w180[2] = {w148[0] + (kWeight4 * w172[0] - -kWeight3 * w172[1]),
                     w148[1] + (kWeight4 * w172[1] + -kWeight3 * w172[0])};
  const T w181[2] = {w148[0] + (-kWeight4 * w172[0] - kWeight3 * w172[1]),
                     w148[1] + (-kWeight4 * w172[1] + kWeight3 * w172[0])};
  const T w182[2] = {w143[0] + w167[1], w143[1] + -w167[0]};
  const T w183[2] = {w143[0] + -w167[1], w143[1] + w167[0]};
  const T w184[2] = {w145[0] + (-kWeight4 * w169[0] - -kWeight3 * w169[1]),
                     w145[1] + (-kWeight4 * w169[1] + -kWeight3 * w169[0])};
  const T w185[2] = {w145[0] + (kWeight4 * w169[0] - kWeight3 * w169[1]),
                     w145[1] + (kWeight4 * w169[1] + kWeight3 * w169[0])};
  const T w186[2] = {w147[0] + (-kWeight2 * w171[0] - -kWeight2 * w171[1]),
                     w147[1] + (-kWeight2 * w171[1] + -kWeight2 * w171[0])};
  const T w187[2] = {w147[0] + (kWeight2 * w171[0] - kWeight2 * w171[1]),
                     w147[1] + (kWeight2 * w171[1] + kWeight2 * w171[0])};
  const T w188[2] = {w149[0] + (-kWeight3 * w173[0] - -kWeight4 * w173[1]),
                     w149[1] + (-kWeight3 * w173[1] + -kWeight4 * w173[0])};
  const T w189[2] = {w149[0] + (kWeight3 * w173[0] - kWeight4 * w173[1]),
                     w149[1] + (kWeight3 * w173[1] + kWeight4 * w173[0])};
  const T w190[2] = {w110[0] + w174[0], w110[1] + w174[1]};
  const T w191[2] = {w110[0] + -w174[0], w110[1] + -w174[1]};
  const T w192[2] = {w112[0] + (kWeight5 * w176[0] - -kWeight6 * w176[1]),
                     w112[1] + (kWeight5 * w176[1] + -kWeight6 * w176[0])};
  const T w193[2] = {w112[0] + (-kWeight5 * w176[0] - kWeight6 * w176[1]),
                     w112[1] + (-kWeight5 * w176[1] + kWeight6 * w176[0])};
  const T w194[2] = {w114[0] + (kWeight3 * w178[0] - -kWeight4 * w178[1]),
                     w114[1] + (kWeight3 * w178[1] + -kWeight4 * w178[0])};
  const T w195[2] = {w114[0] + (-kWeight3 * w178[0] - kWeight4 * w178[1]),
                     w114[1] + (-kWeight3 * w178[1] + kWeight4 * w178[0])};
  const T w196[2] = {w116[0] + (kWeight7 * w180[0] - -kWeight8 * w180[1]),
                     w116[1] + (kWeight7 * w180[1] + -kWeight8 * w180[0])};
  const T w197[2] = {w116[0] + (-kWeight7 * w180[0] - kWeight8 * w180[1]),
                     w116[1] + (-kWeight7 * w180[1] + kWeight8 * w180[0])};
  const T w198[2] = {w118[0] + (kWeight2 * w182[0] - -kWeight2 * w182[1]),
                     w118[1] + (kWeight2 * w182[1] + -kWeight2 * w182[0])};
  const T w199[2] = {w118[0] + (-kWeight2 * w182[0] - kWeight2 * w182[1]),
                     w118[1] + (-kWeight2 * w182[1] + kWeight2 * w182[0])};
  const T w200[2] = {w120[0] + (kWeight8 * w184[0] - -kWeight7 * w184[1]),
                     w120[1] + (kWeight8 * w184[1] + -kWeight7 * w184[0])};
  const T w201[2] = {w120[0] + (-kWeight8 * w184[0] - kWeight7 * w184[1]),
                     w120[1] + (-kWeight8 * w184[1] + kWeight7 * w184[0])};
  const T w202[2] = {w122[0] + (kWeight4 * w186[0] - -kWeight3 * w186[1]),
                     w122[1] + (kWeight4 * w186[1] + -kWeight3 * w186[0])};
  const T w203[2] = {w122[0] + (-kWeight4 * w186[0] - kWeight3 * w186[1]),
                     w122[1] + (-kWeight4 * w186[1] + kWeight3 * w186[0])};
  const T w204[2] = {w124[0] + (kWeight6 * w188[0] - -kWeight5 * w188[1]),
                     w124[1] + (kWeight6 * w188[1] + -kWeight5 * w188[0])};
  const T w205[2] = {w124[0] + (-kWeight6 * w188[0] - kWeight5 * w188[1]),
                     w124[1] + (-kWeight6 * w188[1] + kWeight5 * w188[0])};
  const T w206[2] = {w111[0] + w175[1], w111[1] + -w175[0]};
  const T w207[2] = {w111[0] + -w175[1], w111[1] + w175[0]};
  const T w208[2] = {w113[0] + (-kWeight6 * w177[0] - -kWeight5 * w177[1]),
                     w113[1] + (-kWeight6 * w177[1] + -kWeight5 * w177[0])};
  const T w209[2] = {w113[0] + (kWeight6 * w177[0] - kWeight5 * w177[1]),
                     w113[1] + (kWeight6 * w177[1] + kWeight5 * w177[0])};
  const T w210[2] = {w115[0] + (-kWeight4 * w179[0] - -kWeight3 * w179[1]),
                     w115[1] + (-kWeight4 * w179[1] + -kWeight3 * w179[0])};
  const T w211[2] = {w115[0] + (kWeight4 * w179[0] - kWeight3 * w179[1]),
                     w115[1] + (kWeight4 * w179[1] + kWeight3 * w179[0])};
  const T w212[2] = {w117[0] + (-kWeight8 * w181[0] - -kWeight7 * w181[1]),
                     w117[1] + (-kWeight8 * w181[1] + -kWeight7 * w181[0])};
  const T w213[2] = {w117[0] + (kWeight8 * w181[0] - kWeight7 * w181[1]),
                     w117[1] + (kWeight8 * w181[1] + kWeight7 * w181[0])};
  const T w214[2] = {w119[0] + (-kWeight2 * w183[0] - -kWeight2 * w183[1]),
                     w119[1] + (-kWeight2 * w183[1] + -kWeight2 * w183[0])};
  const T w215[2] = {w119[0] + (kWeight2 * w183[0] - kWeight2 * w183[1]),
                     w119[1] + (kWeight2 * w183[1] + kWeight2 * w183[0])};
  const T w216[2] = {w121[0] + (-kWeight7 * w185[0] - -kWeight8 * w185[1]),
                     w121[1] + (-kWeight7 * w185[1] + -kWeight8 * w185[0])};
  const T w217[2] = {w121[0] + (kWeight7 * w185[0] - kWeight8 * w185[1]),
                     w121[1] + (kWeight7 * w185[1] + kWeight8 * w185[0])};
  const T w218[2] = {w123[0] + (-kWeight3 * w187[0] - -kWeight4 * w187[1]),
                     w123[1] + (-kWeight3 * w187[1] + -kWeight4 * w187[0])};
  const T w219[2] = {w123[0] + (kWeight3 * w187[0] - kWeight4 * w187[1]),
                     w123[1] + (kWeight3 * w187[1] + kWeight4 * w187[0])};
  const T w220[2] = {w125[0] + (-kWeight5 * w189[0] - -kWeight6 * w189[1]),
                     w125[1] + (-kWeight5 * w189[1] + -kWeight6 * w189[0])};
  const T w221[2] = {w125[0] + (kWeight5 * w189[0] - kWeight6 * w189[1]),
                     w125[1] + (kWeight5 * w189[1] + kWeight6 * w189[0])};
  const T w222[2] = {w61[0] + w30[0], w61[1] + w30[1]};
  const T w223[2] = {w61[0] + -w30[0], w61[1] + -w30[1]};
  const T w224[2] = {w45[0] + w14[0], w45[1] + w14[1]};
  const T w225[2] = {w45[0] + -w14[0], w45[1] + -w14[1]};
  const T w226[2] = {w222[0] + w224[0], w222[1] + w224[1]};
  const T w227[2] = {w222[0] + -w224[0], w222[1] + -w224[1]};
  const T w228[2] = {w223[0] + w225[1], w223[1] + -w225[0]};
  const T w229[2] = {w223[0] + -w225[1], w223[1] + w225[0]};
  const T w230[2] = {w53[0] + w22[0], w53[1] + w22[1]};
  const T w231[2] = {w53[0] + -w22[0], w53[1] + -w22[1]};
  const T w232[2] = {w37[0] + w6[0], w37[1] + w6[1]};
  const T w233[2] = {w37[0] + -w6[0], w37[1] + -w6[1]};
  const T w234[2] = {w230[0] + w232[0], w230[1] + w232[1]};
  const T w235[2] = {w230[0] + -w232[0], w230[1] + -w232[1]};
  const T w236[2] = {w231[0] + w233[1], w231[1] + -w233[0]};
  const T w237[2] = {w231[0] + -w233[1], w231[1] + w233[0]};
  const T w238[2] = {w226[0] + w234[0], w226[1] + w234[1]};
  const T w239[2] = {w226[0] + -w234[0], w226[1] + -w234[1]};
  const T w240[2] = {w228[0] + (kWeight2 * w236[0] - -kWeight2 * w236[1]),
                     w228[1] + (kWeight2 * w236[1] + -kWeight2 * w236[0])};
  const T w241[2] = {w228[0] + (-kWeight2 * w236[0] - kWeight2 * w236[1]),
                     w228[1] + (-kWeight2 * w236[1] + kWeight2 * w236[0])};
  const T w242[2] = {w227[0] + w235[1], w227[1] + -w235[0]};
  const T w243[2] = {w227[0] + -w235[1], w227[1] + w235[0]};
  const T w244[2] = {w229[0] + (-kWeight2 * w237[0] - -kWeight2 * w237[1]),
                     w229[1] + (-kWeight2 * w237[1] + -kWeight2 * w237[0])};
  const T w245[2] = {w229[0] + (kWeight2 * w237[0] - kWeight2 * w237[1]),
                     w229[1] + (kWeight2 * w237[1] + kWeight2 * w237[0])};
  const T w246[2] = {w57[0] + w26[0], w57[1] + w26[1]};
  const T w247[2] = {w57[0] + -w26[0], w57[1] + -w26[1]};
  const T w248[2] = {w41[0] + w10[0], w41[1] + w10[1]};
  const T w249[2] = {w41[0] + -w10[0], w41[1] + -w10[1]};
  const T w250[2] = {w246[0] + w248[0], w246[1] + w248[1]};
  const T w251[2] = {w246[0] + -w248[0], w246[1] + -w248[1]};
  const T w252[2] = {w247[0] + w249[1], w247[1] + -w249[0]};
  const T w253[2] = {w247[0] + -w249[1], w247[1] + w249[0]};
  const T w254[2] = {w49[0] + w18[0], w49[1] + w18[1]};
  const T w255[2] = {w49[0] + -w18[0], w49[1] + -w18[1]};
  const T w256[2] = {w33[0] + w2[0], w33[1] + w2[1]};
  const T w257[2] = {w33[0] + -w2[0], w33[1] + -w2[1]};
  const T w258[2] = {w254[0] + w256[0], w254[1] + w256[1]};
  const T w259[2] = {w254[0] + -w256[0], w254[1] + -w256[1]};
  const T w260[2] = {w255[0] + w257[1], w255[1] + -w257[0]};
  const T w261[2] = {w255[0] + -w257[1], w255[1] + w257[0]};
  const T w262[2] = {w250[0] + w258[0], w250[1] + w258[1]};
  const T w263[2] = {w250[0] + -w258[0], w250[1] + -w258[1]};
  const T w264[2] = {w252[0] + (kWeight2 * w260[0] - -kWeight2 * w260[1]),
                     w252[1] + (kWeight2 * w260[1] + -kWeight2 * w260[0])};
  const T w265[2] = {w252[0] + (-kWeight2 * w260[0] - kWeight2 * w260[1]),
                     w252[1] + (-kWeight2 * w260[1] + kWeight2 * w260[0])};
  const T w266[2] = {w251[0] + w259[1], w251[1] + -w259[0]};
  const T w267[2] = {w251[0] + -w259[1], w251[1] + w259[0]};
  const T w268[2] = {w253[0] + (-kWeight2 * w261[0] - -kWeight2 * w261[1]),
                     w253[1] + (-kWeight2 * w261[1] + -kWeight2 * w261[0])};
  const T w269[2] = {w253[0] + (kWeight2 * w261[0] - kWeight2 * w261[1]),
                     w253[1] + (kWeight2 * w261[1] + kWeight2 * w261[0])};
  const T w270[2] = {w238[0] + w262[0], w238[1] + w262[1]};
  const T w271[2] = {w238[0] + -w262[0], w238[1] + -w262[1]};
  const T w272[2] = {w240[0] + (kWeight3 * w264[0] - -kWeight4 * w264[1]),
                     w240[1] + (kWeight3 * w264[1] + -kWeight4 * w264[0])};
  const T w273[2] = {w240[0] + (-kWeight3 * w264[0] - kWeight4 * w264[1]),
                     w240[1] + (-kWeight3 * w264[1] + kWeight4 * w264[0])};
  const T w274[2] = {w242[0] + (kWeight2 * w266[0] - -kWeight2 * w266[1]),
                     w242[1] + (kWeight2 * w266[1] + -kWeight2 * w266[0])};
  const T w275[2] = {w242[0] + (-kWeight2 * w266[0] - kWeight2 * w266[1]),
                     w242[1] + (-kWeight2 * w266[1] + kWeight2 * w266[0])};
  const T w276[2] = {w244[0] + (kWeight4 * w268[0] - -kWeight3 * w268[1]),
                     w244[1] + (kWeight4 * w268[1] + -kWeight3 * w268[0])};
  const T w277[2] = {w244[0] + (-kWeight4 * w268[0] - kWeight3 * w268[1]),
                     w244[1] + (-kWeight4 * w268[1] + kWeight3 * w268[0])};
  const T w278[2] = {w239[0] + w263[1], w239[1] + -w263[0]};
  const T w279[2] = {w239[0] + -w263[1], w239[1] + w263[0]};
  const T w280[2] = {w241[0] + (-kWeight4 * w265[0] - -kWeight3 * w265[1]),
                     w241[1] + (-kWeight4 * w265[1] + -kWeight3 * w265[0])};
  const T w281[2] = {w241[0] + (kWeight4 * w265[0] - kWeight3 * w265[1]),
                     w241[1] + (kWeight4 * w265[1] + kWeight3 * w265[0])};
  const T w282[2] = {w243[0] + (-kWeight2 * w267[0] - -kWeight2 * w267[1]),
                     w243[1] + (-kWeight2 * w267[1] + -kWeight2 * w267[0])};
  const T w283[2] = {w243[0] + (kWeight2 * w267[0] - kWeight2 * w267[1]),
                     w243[1] + (kWeight2 * w267[1] + kWeight2 * w267[0])};
  const T w284[2] = {w245[0] + (-kWeight3 * w269[0] - -kWeight4 * w269[1]),
                     w245[1] + (-kWeight3 * w269[1] + -kWeight4 * w269[0])};
  const T w285[2] = {w245[0] + (kWeight3 * w269[0] - kWeight4 * w269[1]),
                     w245[1] + (kWeight3 * w269[1] + kWeight4 * w269[0])};
  const T w286[2] = {w59[0] + w28[0], w59[1] + w28[1]};
  const T w287[2] = {w59[0] + -w28[0], w59[1] + -w28[1]};
  const T w288[2] = {w43[0] + w12[0], w43[1] + w12[1]};
  const T w289[2] = {w43[0] + -w12[0], w43[1] + -w12[1]};
  const T w290[2] = {w286[0] + w288[0], w286[1] + w288[1]};
  const T w291[2] = {w286[0] + -w288[0], w286[1] + -w288[1]};
  const T w292[2] = {w287[0] + w289[1], w287[1] + -w289[0]};
  const T w293[2] = {w287[0] + -w289[1], w287[1] + w289[0]};
  const T w294[2] = {w51[0] + w20[0], w51[1] + w20[1]};
  const T w295[2] = {w51[0] + -w20[0], w51[1] + -w20[1]};
  const T w296[2] = {w35[0] + w4[0], w35[1] + w4[1]};
  const T w297[2] = {w35[0] + -w4[0], w35[1] + -w4[1]};
  const T w298[2] = {w294[0] + w296[0], w294[1] + w296[1]};
  const T w299[2] = {w294[0] + -w296[0], w294[1] + -w296[1]};
  const T w300[2] = {w295[0] + w297[1], w295[1] + -w297[0]};
  const T w301[2] = {w295[0] + -w297[1], w295[1] + w297[0]};
  const T w302[2] = {w290[0] + w298[0], w290[1] + w298[1]};
  const T w303[2] = {w290[0] + -w298[0], w290[1] + -w298[1]};
  const T w304[2] = {w292[0] + (kWeight2 * w300[0] - -kWeight2 * w300[1]),
                     w292[1] + (kWeight2 * w300[1] + -kWeight2 * w300[0])};
  const T w305[2] = {w292[0] + (-kWeight2 * w300[0] - kWeight2 * w300[1]),
                     w292[1] + (-kWeight2 * w300[1] + kWeight2 * w300[0])};
  const T w306[2] = {w291[0] + w299[1], w291[1] + -w299[0]};
  const T w307[2] = {w291[0] + -w299[1], w291[1] + w299[0]};
  const T w308[2] = {w293[0] + (-kWeight2 * w301[0] - -kWeight2 * w301[1]),
                     w293[1] + (-kWeight2 * w301[1] + -kWeight2 * w301[0])};
  const T w309[2] = {w293[0] + (kWeight2 * w301[0] - kWeight2 * w301[1]),
                     w293[1] + (kWeight2 * w301[1] + kWeight2 * w301[0])};
  const T w310[2] = {w55[0] + w24[0], w55[1] + w24[1]};
  const T w311[2] = {w55[0] + -w24[0], w55[1] + -w24[1]};
  const T w312[2] = {w39[0] + w8[0], w39[1] + w8[1]};
  const T w313[2] = {w39[0] + -w8[0], w39[1] + -w8[1]};
  const T w314[2] = {w310[0] + w312[0], w310[1] + w312[1]};
  const T w315[2] = {w310[0] + -w312[0], w310[1] + -w312[1]};
  const T w316[2] = {w311[0] + w313[1], w311[1] + -w313[0]};
  const T w317[2] = {w311[0] + -w313[1], w311[1] + w313[0]};
  const T w318[2] = {w47[0] + w16[0], w47[1] + w16[1]};
  const T w319[2] = {w47[0] + -w16[0], w47[1] + -w16[1]};
  const T w320[2] = {w31[0] + w0[0], w31[1] + w0[1]};
  const T w321[2] = {w31[0] + -w0[0], w31[1] + -w0[1]};
  const T w322[2] = {w318[0] + w320[0], w318[1] + w320[1]};
  const T w323[2] = {w318[0] + -w320[0], w318[1] + -w320[1]};
  const T w324[2] = {w319[0] + w321[1], w319[1] + -w321[0]};
  const T w325[2] = {w319[0] + -w321[1], w319[1] + w321[0]};
  const T w326[2] = {w314[0] + w322[0], w314[1] + w322[1]};
  const T w327[2] = {w314[0] + -w322[0], w314[1] + -w322[1]};
  const T w328[2] = {w316[0] + (kWeight2 * w324[0] - -kWeight2 * w324[1]),
                     w316[1] + (kWeight2 * w324[1] + -kWeight2 * w324[0])};
  const T w329[2] = {w316[0] + (-kWeight2 * w324[0] - kWeight2 * w324[1]),
                     w316[1] + (-kWeight2 * w324[1] + kWeight2 * w324[0])};
  const T w330[2] = {w315[0] + w323[1], w315[1] + -w323[0]};
  const T w331[2] = {w315[0] + -w323[1], w315[1] + w323[0]};
  const T w332[2] = {w317[0] + (-kWeight2 * w325[0] - -kWeight2 * w325[1]),
                     w317[1] + (-kWeight2 * w325[1] + -kWeight2 * w325[0])};
  const T w333[2] = {w317[0] + (kWeight2 * w325[0] - kWeight2 * w325[1]),
                     w317[1] + (kWeight2 * w325[1] + kWeight2 * w325[0])};
  const T w334[2] = {w302[0] + w326[0], w302[1] + w326[1]};
  const T w335[2] = {w302[0] + -w326[0], w302[1] + -w326[1]};
  const T w336[2] = {w304[0] + (kWeight3 * w328[0] - -kWeight4 * w328[1]),
                     w304[1] + (kWeight3 * w328[1] + -kWeight4 * w328[0])};
  const T w337[2] = {w304[0] + (-kWeight3 * w328[0] - kWeight4 * w328[1]),
                     w304[1] + (-kWeight3 * w328[1] + kWeight4 * w328[0])};
  const T w338[2] = {w306[0] + (kWeight2 * w330[0] - -kWeight2 * w330[1]),
                     w306[1] + (kWeight2 * w330[1] + -kWeight2 * w330[0])};
  const T w339[2] = {w306[0] + (-kWeight2 * w330[0] - kWeight2 * w330[1]),
                     w306[1] + (-kWeight2 * w330[1] + kWeight2 * w330[0])};
  const T w340[2] = {w308[0] + (kWeight4 * w332[0] - -kWeight3 * w332[1]),
                     w308[1] + (kWeight4 * w332[1] + -kWeight3 * w332[0])};
  const T w341[2] = {w308[0] + (-kWeight4 * w332[0] - kWeight3 * w332[1]),
                     w308[1] + (-kWeight4 * w332[1] + kWeight3 * w332[0])};
  const T w342[2] = {w303[0] + w327[1], w303[1] + -w327[0]};
  const T w343[2] = {w303[0] + -w327[1], w303[1] + w327[0]};
  const T w344[2] = {w305[0] + (-kWeight4 * w329[0] - -kWeight3 * w329[1]),
                     w305[1] + (-kWeight4 * w329[1] + -kWeight3 * w329[0])};
  const T w345[2] = {w305[0] + (kWeight4 * w329[0] - kWeight3 * w329[1]),
                     w305[1] + (kWeight4 * w329[1] + kWeight3 * w329[0])};
  const T w346[2] = {w307[0] + (-kWeight2 * w331[0] - -kWeight2 * w331[1]),
                     w307[1] + (-kWeight2 * w331[1] + -kWeight2 * w331[0])};
  const T w347[2] = {w307[0] + (kWeight2 * w331[0] - kWeight2 * w331[1]),
                     w307[1] + (kWeight2 * w331[1] + kWeight2 * w331[0])};
  const T w348[2] = {w309[0] + (-kWeight3 * w333[0] - -kWeight4 * w333[1]),
                     w309[1] + (-kWeight3 * w333[1] + -kWeight4 * w333[0])};
  const T w349[2] = {w309[0] + (kWeight3 * w333[0] - kWeight4 * w333[1]),
                     w309[1] + (kWeight3 * w333[1] + kWeight4 * w333[0])};
  const T w350[2] = {w270[0] + w334[0], w270[1] + w334[1]};
  const T w351[2] = {w270[0] + -w334[0], w270[1] + -w334[1]};
  const T w352[2] = {w272[0] + (kWeight5 * w336[0] - -kWeight6 * w336[1]),
                     w272[1] + (kWeight5 * w336[1] + -kWeight6 * w336[0])};
  const T w353[2] = {w272[0] + (-kWeight5 * w336[0] - kWeight6 * w336[1]),
                     w272[1] + (-kWeight5 * w336[1] + kWeight6 * w336[0])};
  const T w354[2] = {w274[0] + (kWeight3 * w338[0] - -kWeight4 * w338[1]),
                     w274[1] + (kWeight3 * w338[1] + -kWeight4 * w338[0])};
  const T w355[2] = {w274[0] + (-kWeight3 * w338[0] - kWeight4 * w338[1]),
                     w274[1] + (-kWeight3 * w338[1] + kWeight4 * w338[0])};
  const T w356[2] = {w276[0] + (kWeight7 * w340[0] - -kWeight8 * w340[1]),
                     w276[1] + (kWeight7 * w340[1] + -kWeight8 * w340[0])};
  const T w357[2] = {w276[0] + (-kWeight7 * w340[0] - kWeight8 * w340[1]),
                     w276[1] + (-kWeight7 * w340[1] + kWeight8 * w340[0])};
  const T w358[2] = {w278[0] + (kWeight2 * w342[0] - -kWeight2 * w342[1]),
                     w278[1] + (kWeight2 * w342[1] + -kWeight2 * w342[0])};
  const T w359[2] = {w278[0] + (-kWeight2 * w342[0] - kWeight2 * w342[1]),
                     w278[1] + (-kWeight2 * w342[1] + kWeight2 * w342[0])};
  const T w360[2] = {w280[0] + (kWeight8 * w344[0] - -kWeight7 * w344[1]),
                     w280[1] + (kWeight8 * w344[1] + -kWeight7 * w344[0])};
  const T w361[2] = {w280[0] + (-kWeight8 * w344[0] - kWeight7 * w344[1]),
                     w280[1] + (-kWeight8 * w344[1] + kWeight7 * w344[0])};
  const T w362[2] = {w282[0] + (kWeight4 * w346[0] - -kWeight3 * w346[1]),
                     w282[1] + (kWeight4 * w346[1] + -kWeight3 * w346[0])};
  const T w363[2] = {w282[0] + (-kWeight4 * w346[0] - kWeight3 * w346[1]),
                     w282[1] + (-kWeight4 * w346[1] + kWeight3 * w346[0])};
  const T w364[2] = {w284[0] + (kWeight6 * w348[0] - -kWeight5 * w348[1]),
                     w284[1] + (kWeight6 * w348[1] + -kWeight5 * w348[0])};
  const T w365[2] = {w284[0] + (-kWeight6 * w348[0] - kWeight5 * w348[1]),
                     w284[1] + (-kWeight6 * w348[1] + kWeight5 * w348[0])};
  const T w366[2] = {w271[0] + w335[1], w271[1] + -w335[0]};
  const T w367[2] = {w271[0] + -w335[1], w271[1] + w335[0]};
  const T w368[2] = {w273[0] + (-kWeight6 * w337[0] - -kWeight5 * w337[1]),
                     w273[1] + (-kWeight6 * w337[1] + -kWeight5 * w337[0])};
  const T w369[2] = {w273[0] + (kWeight6 * w337[0] - kWeight5 * w337[1]),
                     w273[1] + (kWeight6 * w337[1] + kWeight5 * w337[0])};
  const T w370[2] = {w275[0] + (-kWeight4 * w339[0] - -kWeight3 * w339[1]),
                     w275[1] + (-kWeight4 * w339[1] + -kWeight3 * w339[0])};
  const T w371[2] = {w275[0] + (kWeight4 * w339[0] - kWeight3 * w339[1]),
                     w275[1] + (kWeight4 * w339[1] + kWeight3 * w339[0])};
  const T w372[2] = {w277[0] + (-kWeight8 * w341[0] - -kWeight7 * w341[1]),
                     w277[1] + (-kWeight8 * w341[1] + -kWeight7 * w341[0])};
  const T w373[2] = {w277[0] + (kWeight8 * w341[0] - kWeight7 * w341[1]),
                     w277[1] + (kWeight8 * w341[1] + kWeight7 * w341[0])};
  const T w374[2] = {w279[0] + (-kWeight2 * w343[0] - -kWeight2 * w343[1]),
                     w279[1] + (-kWeight2 * w343[1] + -kWeight2 * w343[0])};
  const T w375[2] = {w279[0] + (kWeight2 * w343[0] - kWeight2 * w343[1]),
                     w279[1] + (kWeight2 * w343[1] + kWeight2 * w343[0])};
  const T w376[2] = {w281[0] + (-kWeight7 * w345[0] - -kWeight8 * w345[1]),
                     w281[1] + (-kWeight7 * w345[1] + -kWeight8 * w345[0])};
  const T w377[2] = {w281[0] + (kWeight7 * w345[0] - kWeight8 * w345[1]),
                     w281[1] + (kWeight7 * w345[1] + kWeight8 * w345[0])};
  const T w378[2] = {w283[0] + (-kWeight3 * w347[0] - -kWeight4 * w347[1]),
                     w283[1] + (-kWeight3 * w347[1] + -kWeight4 * w347[0])};
  const T w379[2] = {w283[0] + (kWeight3 * w347[0] - kWeight4 * w347[1]),
                     w283[1] + (kWeight3 * w347[1] + kWeight4 * w347[0])};
  const T w380[2] = {w285[0] + (-kWeight5 * w349[0] - -kWeight6 * w349[1]),
                     w285[1] + (-kWeight5 * w349[1] + -kWeight6 * w349[0])};
  const T w381[2] = {w285[0] + (kWeight5 * w349[0] - kWeight6 * w349[1]),
                     w285[1] + (kWeight5 * w349[1] + kWeight6 * w349[0])};
  SimdHelper<T, I>::store(output + 0 * stride, w190[0] + w350[0]);
  SimdHelper<T, I>::store(
      output + 1 * stride,
      w192[0] + (kWeight9 * w352[0] - -kWeight10 * w352[1]));
  SimdHelper<T, I>::store(output + 2 * stride,
                          w194[0] + (kWeight5 * w354[0] - -kWeight6 * w354[1]));
  SimdHelper<T, I>::store(
      output + 3 * stride,
      w196[0] + (kWeight11 * w356[0] - -kWeight12 * w356[1]));
  SimdHelper<T, I>::store(output + 4 * stride,
                          w198[0] + (kWeight3 * w358[0] - -kWeight4 * w358[1]));
  SimdHelper<T, I>::store(
      output + 5 * stride,
      w200[0] + (kWeight13 * w360[0] - -kWeight14 * w360[1]));
  SimdHelper<T, I>::store(output + 6 * stride,
                          w202[0] + (kWeight7 * w362[0] - -kWeight8 * w362[1]));
  SimdHelper<T, I>::store(
      output + 7 * stride,
      w204[0] + (kWeight15 * w364[0] - -kWeight16 * w364[1]));
  SimdHelper<T, I>::store(output + 8 * stride,
                          w206[0] + (kWeight2 * w366[0] - -kWeight2 * w366[1]));
  SimdHelper<T, I>::store(
      output + 9 * stride,
      w208[0] + (kWeight16 * w368[0] - -kWeight15 * w368[1]));
  SimdHelper<T, I>::store(output + 10 * stride,
                          w210[0] + (kWeight8 * w370[0] - -kWeight7 * w370[1]));
  SimdHelper<T, I>::store(
      output + 11 * stride,
      w212[0] + (kWeight14 * w372[0] - -kWeight13 * w372[1]));
  SimdHelper<T, I>::store(output + 12 * stride,
                          w214[0] + (kWeight4 * w374[0] - -kWeight3 * w374[1]));
  SimdHelper<T, I>::store(
      output + 13 * stride,
      w216[0] + (kWeight12 * w376[0] - -kWeight11 * w376[1]));
  SimdHelper<T, I>::store(output + 14 * stride,
                          w218[0] + (kWeight6 * w378[0] - -kWeight5 * w378[1]));
  SimdHelper<T, I>::store(
      output + 15 * stride,
      w220[0] + (kWeight10 * w380[0] - -kWeight9 * w380[1]));
  SimdHelper<T, I>::store(output + 16 * stride, w191[0] + w351[1]);
  SimdHelper<T, I>::store(
      output + 17 * stride,
      w193[0] + (-kWeight10 * w353[0] - -kWeight9 * w353[1]));
  SimdHelper<T, I>::store(
      output + 18 * stride,
      w195[0] + (-kWeight6 * w355[0] - -kWeight5 * w355[1]));
  SimdHelper<T, I>::store(
      output + 19 * stride,
      w197[0] + (-kWeight12 * w357[0] - -kWeight11 * w357[1]));
  SimdHelper<T, I>::store(
      output + 20 * stride,
      w199[0] + (-kWeight4 * w359[0] - -kWeight3 * w359[1]));
  SimdHelper<T, I>::store(
      output + 21 * stride,
      w201[0] + (-kWeight14 * w361[0] - -kWeight13 * w361[1]));
  SimdHelper<T, I>::store(
      output + 22 * stride,
      w203[0] + (-kWeight8 * w363[0] - -kWeight7 * w363[1]));
  SimdHelper<T, I>::store(
      output + 23 * stride,
      w205[0] + (-kWeight16 * w365[0] - -kWeight15 * w365[1]));
  SimdHelper<T, I>::store(
      output + 24 * stride,
      w207[0] + (-kWeight2 * w367[0] - -kWeight2 * w367[1]));
  SimdHelper<T, I>::store(
      output + 25 * stride,
      w209[0] + (-kWeight15 * w369[0] - -kWeight16 * w369[1]));
  SimdHelper<T, I>::store(
      output + 26 * stride,
      w211[0] + (-kWeight7 * w371[0] - -kWeight8 * w371[1]));
  SimdHelper<T, I>::store(
      output + 27 * stride,
      w213[0] + (-kWeight13 * w373[0] - -kWeight14 * w373[1]));
  SimdHelper<T, I>::store(
      output + 28 * stride,
      w215[0] + (-kWeight3 * w375[0] - -kWeight4 * w375[1]));
  SimdHelper<T, I>::store(
      output + 29 * stride,
      w217[0] + (-kWeight11 * w377[0] - -kWeight12 * w377[1]));
  SimdHelper<T, I>::store(
      output + 30 * stride,
      w219[0] + (-kWeight5 * w379[0] - -kWeight6 * w379[1]));
  SimdHelper<T, I>::store(
      output + 31 * stride,
      w221[0] + (-kWeight9 * w381[0] - -kWeight10 * w381[1]));
  SimdHelper<T, I>::store(output + 32 * stride, w190[0] + -w350[0]);
  SimdHelper<T, I>::store(
      output + 33 * stride,
      w192[0] + (-kWeight9 * w352[0] - kWeight10 * w352[1]));
  SimdHelper<T, I>::store(output + 34 * stride,
                          w194[0] + (-kWeight5 * w354[0] - kWeight6 * w354[1]));
  SimdHelper<T, I>::store(
      output + 35 * stride,
      w196[0] + (-kWeight11 * w356[0] - kWeight12 * w356[1]));
  SimdHelper<T, I>::store(output + 36 * stride,
                          w198[0] + (-kWeight3 * w358[0] - kWeight4 * w358[1]));
  SimdHelper<T, I>::store(
      output + 37 * stride,
      w200[0] + (-kWeight13 * w360[0] - kWeight14 * w360[1]));
  SimdHelper<T, I>::store(output + 38 * stride,
                          w202[0] + (-kWeight7 * w362[0] - kWeight8 * w362[1]));
  SimdHelper<T, I>::store(
      output + 39 * stride,
      w204[0] + (-kWeight15 * w364[0] - kWeight16 * w364[1]));
  SimdHelper<T, I>::store(output + 40 * stride,
                          w206[0] + (-kWeight2 * w366[0] - kWeight2 * w366[1]));
  SimdHelper<T, I>::store(
      output + 41 * stride,
      w208[0] + (-kWeight16 * w368[0] - kWeight15 * w368[1]));
  SimdHelper<T, I>::store(output + 42 * stride,
                          w210[0] + (-kWeight8 * w370[0] - kWeight7 * w370[1]));
  SimdHelper<T, I>::store(
      output + 43 * stride,
      w212[0] + (-kWeight14 * w372[0] - kWeight13 * w372[1]));
  SimdHelper<T, I>::store(output + 44 * stride,
                          w214[0] + (-kWeight4 * w374[0] - kWeight3 * w374[1]));
  SimdHelper<T, I>::store(
      output + 45 * stride,
      w216[0] + (-kWeight12 * w376[0] - kWeight11 * w376[1]));
  SimdHelper<T, I>::store(output + 46 * stride,
                          w218[0] + (-kWeight6 * w378[0] - kWeight5 * w378[1]));
  SimdHelper<T, I>::store(
      output + 47 * stride,
      w220[0] + (-kWeight10 * w380[0] - kWeight9 * w380[1]));
  SimdHelper<T, I>::store(output + 48 * stride, w191[0] + -w351[1]);
  SimdHelper<T, I>::store(output + 49 * stride,
                          w193[0] + (kWeight10 * w353[0] - kWeight9 * w353[1]));
  SimdHelper<T, I>::store(output + 50 * stride,
                          w195[0] + (kWeight6 * w355[0] - kWeight5 * w355[1]));
  SimdHelper<T, I>::store(
      output + 51 * stride,
      w197[0] + (kWeight12 * w357[0] - kWeight11 * w357[1]));
  SimdHelper<T, I>::store(output + 52 * stride,
                          w199[0] + (kWeight4 * w359[0] - kWeight3 * w359[1]));
  SimdHelper<T, I>::store(
      output + 53 * stride,
      w201[0] + (kWeight14 * w361[0] - kWeight13 * w361[1]));
  SimdHelper<T, I>::store(output + 54 * stride,
                          w203[0] + (kWeight8 * w363[0] - kWeight7 * w363[1]));
  SimdHelper<T, I>::store(
      output + 55 * stride,
      w205[0] + (kWeight16 * w365[0] - kWeight15 * w365[1]));
  SimdHelper<T, I>::store(output + 56 * stride,
                          w207[0] + (kWeight2 * w367[0] - kWeight2 * w367[1]));
  SimdHelper<T, I>::store(
      output + 57 * stride,
      w209[0] + (kWeight15 * w369[0] - kWeight16 * w369[1]));
  SimdHelper<T, I>::store(output + 58 * stride,
                          w211[0] + (kWeight7 * w371[0] - kWeight8 * w371[1]));
  SimdHelper<T, I>::store(
      output + 59 * stride,
      w213[0] + (kWeight13 * w373[0] - kWeight14 * w373[1]));
  SimdHelper<T, I>::store(output + 60 * stride,
                          w215[0] + (kWeight3 * w375[0] - kWeight4 * w375[1]));
  SimdHelper<T, I>::store(
      output + 61 * stride,
      w217[0] + (kWeight11 * w377[0] - kWeight12 * w377[1]));
  SimdHelper<T, I>::store(output + 62 * stride,
                          w219[0] + (kWeight5 * w379[0] - kWeight6 * w379[1]));
  SimdHelper<T, I>::store(output + 63 * stride,
                          w221[0] + (kWeight9 * w381[0] - kWeight10 * w381[1]));
}