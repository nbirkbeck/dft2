template <typename T, typename I=float>
void dft_2_compact(const I* input, I* output, int stride=1) {
  const T i0 = SimdHelper<T, I>::load(input + 0 * stride);
  const T i1 = SimdHelper<T, I>::load(input + 1 * stride);
  store(output + 0 * stride, i0 +i1);
  store(output + 1 * stride, i0 +-i1);
}
template <typename T, typename I=float>
void dft_4_compact(const I* input, I* output, int stride=1) {
  const T i0 = SimdHelper<T, I>::load(input + 0 * stride);
  const T i1 = SimdHelper<T, I>::load(input + 1 * stride);
  const T i2 = SimdHelper<T, I>::load(input + 2 * stride);
  const T i3 = SimdHelper<T, I>::load(input + 3 * stride);
  const T w0  = i0 +i2;
  const T w1  = i0 +-i2;
  const T w2  = i1 +i3;
  const T w3  = i1 +-i3;
  store(output + 0 * stride, w0 +w2);
  store(output + 1 * stride, w1);
  store(output + 2 * stride, w0 +-w2);
  store(output + 3 * stride, -w3);
}
template <typename T, typename I=float>
void dft_8_compact(const I* input, I* output, int stride=1) {
  const T kWeight2 = SimdHelper<T, I>::constant(0.707107);
  const T i0 = SimdHelper<T, I>::load(input + 0 * stride);
  const T i1 = SimdHelper<T, I>::load(input + 1 * stride);
  const T i2 = SimdHelper<T, I>::load(input + 2 * stride);
  const T i3 = SimdHelper<T, I>::load(input + 3 * stride);
  const T i4 = SimdHelper<T, I>::load(input + 4 * stride);
  const T i5 = SimdHelper<T, I>::load(input + 5 * stride);
  const T i6 = SimdHelper<T, I>::load(input + 6 * stride);
  const T i7 = SimdHelper<T, I>::load(input + 7 * stride);
  const T w0  = i0 +i4;
  const T w1  = i0 +-i4;
  const T w2  = i2 +i6;
  const T w3  = i2 +-i6;
  const T w4  = w0 +w2;
  const T w5  = w0 +-w2;
  const T w7  = i1 +i5;
  const T w8  = i1 +-i5;
  const T w9  = i3 +i7;
  const T w10  = i3 +-i7;
  const T w11  = w7 +w9;
  const T w12  = w7 +-w9;
  store(output + 0 * stride, w4 +w11);
  store(output + 1 * stride, w1 +( kWeight2*w8 -  kWeight2*w10));
  store(output + 2 * stride, w5);
  store(output + 3 * stride, w1 +(-kWeight2*w8 - -kWeight2*w10));
  store(output + 4 * stride, w4 +-w11);
  store(output + 5 * stride, -w3 +-( kWeight2*w10 +  kWeight2*w8));
  store(output + 6 * stride, -w12);
  store(output + 7 * stride, w3 +(-kWeight2*w10 + -kWeight2*w8));
}
template <typename T, typename I=float>
void dft_16_compact(const I* input, I* output, int stride=1) {
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
  const T w0  = i0 +i8;
  const T w1  = i0 +-i8;
  const T w2  = i4 +i12;
  const T w3  = i4 +-i12;
  const T w4  = w0 +w2;
  const T w5  = w0 +-w2;
  const T w7  = i2 +i10;
  const T w8  = i2 +-i10;
  const T w9  = i6 +i14;
  const T w10  = i6 +-i14;
  const T w11  = w7 +w9;
  const T w12  = w7 +-w9;
  const T w14  = w4 +w11;
  const T w15  = w4 +-w11;
  const T w16[2]  =  {(T)(w1 +( kWeight2*w8 -  kWeight2*w10)), (T)(-w3 +-( kWeight2*w10 +  kWeight2*w8))};
  const T w18[2]  =  {(T)(w1 +(-kWeight2*w8 - -kWeight2*w10)), (T)(w3 +(-kWeight2*w10 + -kWeight2*w8))};
  const T w19  = i1 +i9;
  const T w20  = i1 +-i9;
  const T w21  = i5 +i13;
  const T w22  = i5 +-i13;
  const T w23  = w19 +w21;
  const T w24  = w19 +-w21;
  const T w26  = i3 +i11;
  const T w27  = i3 +-i11;
  const T w28  = i7 +i15;
  const T w29  = i7 +-i15;
  const T w30  = w26 +w28;
  const T w31  = w26 +-w28;
  const T w33  = w23 +w30;
  const T w34  = w23 +-w30;
  const T w35[2]  =  {(T)(w20 +( kWeight2*w27 -  kWeight2*w29)), (T)(-w22 +-( kWeight2*w29 +  kWeight2*w27))};
  const T w37[2]  =  {(T)(w20 +(-kWeight2*w27 - -kWeight2*w29)), (T)(w22 +(-kWeight2*w29 + -kWeight2*w27))};
  store(output + 0 * stride, w14 +w33);
  store(output + 1 * stride, w16[0] +( kWeight3*w35[0] - -kWeight4*w35[1]));
  store(output + 2 * stride, w5 +( kWeight2*w24 -  kWeight2*w31));
  store(output + 3 * stride, w18[0] +( kWeight4*w37[0] - -kWeight3*w37[1]));
  store(output + 4 * stride, w15);
  store(output + 5 * stride, w18[0] +(-kWeight4*w37[0] -  kWeight3*w37[1]));
  store(output + 6 * stride, w5 +(-kWeight2*w24 - -kWeight2*w31));
  store(output + 7 * stride, w16[0] +(-kWeight3*w35[0] -  kWeight4*w35[1]));
  store(output + 8 * stride, w14 +-w33);
  store(output + 9 * stride, w16[1] +( kWeight3*w35[1] + -kWeight4*w35[0]));
  store(output + 10 * stride, -w12 +-( kWeight2*w31 +  kWeight2*w24));
  store(output + 11 * stride, w18[1] +( kWeight4*w37[1] + -kWeight3*w37[0]));
  store(output + 12 * stride, -w34);
  store(output + 13 * stride, -w18[1] +-(-kWeight4*w37[1] +  kWeight3*w37[0]));
  store(output + 14 * stride, w12 +(-kWeight2*w31 + -kWeight2*w24));
  store(output + 15 * stride, -w16[1] +-(-kWeight3*w35[1] +  kWeight4*w35[0]));
}
template <typename T, typename I=float>
void dft_32_compact(const I* input, I* output, int stride=1) {
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
  const T w0  = i0 +i16;
  const T w1  = i0 +-i16;
  const T w2  = i8 +i24;
  const T w3  = i8 +-i24;
  const T w4  = w0 +w2;
  const T w5  = w0 +-w2;
  const T w7  = i4 +i20;
  const T w8  = i4 +-i20;
  const T w9  = i12 +i28;
  const T w10  = i12 +-i28;
  const T w11  = w7 +w9;
  const T w12  = w7 +-w9;
  const T w14  = w4 +w11;
  const T w15  = w4 +-w11;
  const T w16[2]  =  {(T)(w1 +( kWeight2*w8 -  kWeight2*w10)), (T)(-w3 +-( kWeight2*w10 +  kWeight2*w8))};
  const T w18[2]  =  {(T)(w1 +(-kWeight2*w8 - -kWeight2*w10)), (T)(w3 +(-kWeight2*w10 + -kWeight2*w8))};
  const T w19  = i2 +i18;
  const T w20  = i2 +-i18;
  const T w21  = i10 +i26;
  const T w22  = i10 +-i26;
  const T w23  = w19 +w21;
  const T w24  = w19 +-w21;
  const T w26  = i6 +i22;
  const T w27  = i6 +-i22;
  const T w28  = i14 +i30;
  const T w29  = i14 +-i30;
  const T w30  = w26 +w28;
  const T w31  = w26 +-w28;
  const T w33  = w23 +w30;
  const T w34  = w23 +-w30;
  const T w35[2]  =  {(T)(w20 +( kWeight2*w27 -  kWeight2*w29)), (T)(-w22 +-( kWeight2*w29 +  kWeight2*w27))};
  const T w37[2]  =  {(T)(w20 +(-kWeight2*w27 - -kWeight2*w29)), (T)(w22 +(-kWeight2*w29 + -kWeight2*w27))};
  const T w38  = w14 +w33;
  const T w39  = w14 +-w33;
  const T w40[2]  =  {(T)(w16[0] +( kWeight3*w35[0] - -kWeight4*w35[1])), (T)(w16[1] +( kWeight3*w35[1] + -kWeight4*w35[0]))};
  const T w41[2]  =  {(T)(w5 +( kWeight2*w24 -  kWeight2*w31)), (T)(-w12 +-( kWeight2*w31 +  kWeight2*w24))};
  const T w42[2]  =  {(T)(w18[0] +( kWeight4*w37[0] - -kWeight3*w37[1])), (T)(w18[1] +( kWeight4*w37[1] + -kWeight3*w37[0]))};
  const T w44[2]  =  {(T)(w18[0] +(-kWeight4*w37[0] -  kWeight3*w37[1])), (T)(-w18[1] +-(-kWeight4*w37[1] +  kWeight3*w37[0]))};
  const T w45[2]  =  {(T)(w5 +(-kWeight2*w24 - -kWeight2*w31)), (T)(w12 +(-kWeight2*w31 + -kWeight2*w24))};
  const T w46[2]  =  {(T)(w16[0] +(-kWeight3*w35[0] -  kWeight4*w35[1])), (T)(-w16[1] +-(-kWeight3*w35[1] +  kWeight4*w35[0]))};
  const T w47  = i1 +i17;
  const T w48  = i1 +-i17;
  const T w49  = i9 +i25;
  const T w50  = i9 +-i25;
  const T w51  = w47 +w49;
  const T w52  = w47 +-w49;
  const T w54  = i5 +i21;
  const T w55  = i5 +-i21;
  const T w56  = i13 +i29;
  const T w57  = i13 +-i29;
  const T w58  = w54 +w56;
  const T w59  = w54 +-w56;
  const T w61  = w51 +w58;
  const T w62  = w51 +-w58;
  const T w63[2]  =  {(T)(w48 +( kWeight2*w55 -  kWeight2*w57)), (T)(-w50 +-( kWeight2*w57 +  kWeight2*w55))};
  const T w65[2]  =  {(T)(w48 +(-kWeight2*w55 - -kWeight2*w57)), (T)(w50 +(-kWeight2*w57 + -kWeight2*w55))};
  const T w66  = i3 +i19;
  const T w67  = i3 +-i19;
  const T w68  = i11 +i27;
  const T w69  = i11 +-i27;
  const T w70  = w66 +w68;
  const T w71  = w66 +-w68;
  const T w73  = i7 +i23;
  const T w74  = i7 +-i23;
  const T w75  = i15 +i31;
  const T w76  = i15 +-i31;
  const T w77  = w73 +w75;
  const T w78  = w73 +-w75;
  const T w80  = w70 +w77;
  const T w81  = w70 +-w77;
  const T w82[2]  =  {(T)(w67 +( kWeight2*w74 -  kWeight2*w76)), (T)(-w69 +-( kWeight2*w76 +  kWeight2*w74))};
  const T w84[2]  =  {(T)(w67 +(-kWeight2*w74 - -kWeight2*w76)), (T)(w69 +(-kWeight2*w76 + -kWeight2*w74))};
  const T w85  = w61 +w80;
  const T w86  = w61 +-w80;
  const T w87[2]  =  {(T)(w63[0] +( kWeight3*w82[0] - -kWeight4*w82[1])), (T)(w63[1] +( kWeight3*w82[1] + -kWeight4*w82[0]))};
  const T w88[2]  =  {(T)(w52 +( kWeight2*w71 -  kWeight2*w78)), (T)(-w59 +-( kWeight2*w78 +  kWeight2*w71))};
  const T w89[2]  =  {(T)(w65[0] +( kWeight4*w84[0] - -kWeight3*w84[1])), (T)(w65[1] +( kWeight4*w84[1] + -kWeight3*w84[0]))};
  const T w91[2]  =  {(T)(w65[0] +(-kWeight4*w84[0] -  kWeight3*w84[1])), (T)(-w65[1] +-(-kWeight4*w84[1] +  kWeight3*w84[0]))};
  const T w92[2]  =  {(T)(w52 +(-kWeight2*w71 - -kWeight2*w78)), (T)(w59 +(-kWeight2*w78 + -kWeight2*w71))};
  const T w93[2]  =  {(T)(w63[0] +(-kWeight3*w82[0] -  kWeight4*w82[1])), (T)(-w63[1] +-(-kWeight3*w82[1] +  kWeight4*w82[0]))};
  store(output + 0 * stride, w38 +w85);
  store(output + 1 * stride, w40[0] +( kWeight5*w87[0] - -kWeight6*w87[1]));
  store(output + 2 * stride, w41[0] +( kWeight3*w88[0] - -kWeight4*w88[1]));
  store(output + 3 * stride, w42[0] +( kWeight7*w89[0] - -kWeight8*w89[1]));
  store(output + 4 * stride, w15 +( kWeight2*w62 -  kWeight2*w81));
  store(output + 5 * stride, w44[0] +( kWeight8*w91[0] - -kWeight7*w91[1]));
  store(output + 6 * stride, w45[0] +( kWeight4*w92[0] - -kWeight3*w92[1]));
  store(output + 7 * stride, w46[0] +( kWeight6*w93[0] - -kWeight5*w93[1]));
  store(output + 8 * stride, w39);
  store(output + 9 * stride, w46[0] +(-kWeight6*w93[0] -  kWeight5*w93[1]));
  store(output + 10 * stride, w45[0] +(-kWeight4*w92[0] -  kWeight3*w92[1]));
  store(output + 11 * stride, w44[0] +(-kWeight8*w91[0] -  kWeight7*w91[1]));
  store(output + 12 * stride, w15 +(-kWeight2*w62 - -kWeight2*w81));
  store(output + 13 * stride, w42[0] +(-kWeight7*w89[0] -  kWeight8*w89[1]));
  store(output + 14 * stride, w41[0] +(-kWeight3*w88[0] -  kWeight4*w88[1]));
  store(output + 15 * stride, w40[0] +(-kWeight5*w87[0] -  kWeight6*w87[1]));
  store(output + 16 * stride, w38 +-w85);
  store(output + 17 * stride, w40[1] +( kWeight5*w87[1] + -kWeight6*w87[0]));
  store(output + 18 * stride, w41[1] +( kWeight3*w88[1] + -kWeight4*w88[0]));
  store(output + 19 * stride, w42[1] +( kWeight7*w89[1] + -kWeight8*w89[0]));
  store(output + 20 * stride, -w34 +-( kWeight2*w81 +  kWeight2*w62));
  store(output + 21 * stride, w44[1] +( kWeight8*w91[1] + -kWeight7*w91[0]));
  store(output + 22 * stride, w45[1] +( kWeight4*w92[1] + -kWeight3*w92[0]));
  store(output + 23 * stride, w46[1] +( kWeight6*w93[1] + -kWeight5*w93[0]));
  store(output + 24 * stride, -w86);
  store(output + 25 * stride, -w46[1] +-(-kWeight6*w93[1] +  kWeight5*w93[0]));
  store(output + 26 * stride, -w45[1] +-(-kWeight4*w92[1] +  kWeight3*w92[0]));
  store(output + 27 * stride, -w44[1] +-(-kWeight8*w91[1] +  kWeight7*w91[0]));
  store(output + 28 * stride, w34 +(-kWeight2*w81 + -kWeight2*w62));
  store(output + 29 * stride, -w42[1] +-(-kWeight7*w89[1] +  kWeight8*w89[0]));
  store(output + 30 * stride, -w41[1] +-(-kWeight3*w88[1] +  kWeight4*w88[0]));
  store(output + 31 * stride, -w40[1] +-(-kWeight5*w87[1] +  kWeight6*w87[0]));
}
template <typename T, typename I=float>
void dft_64_compact(const I* input, I* output, int stride=1) {
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
  const T w0  = i0 +i32;
  const T w1  = i0 +-i32;
  const T w2  = i16 +i48;
  const T w3  = i16 +-i48;
  const T w4  = w0 +w2;
  const T w5  = w0 +-w2;
  const T w7  = i8 +i40;
  const T w8  = i8 +-i40;
  const T w9  = i24 +i56;
  const T w10  = i24 +-i56;
  const T w11  = w7 +w9;
  const T w12  = w7 +-w9;
  const T w14  = w4 +w11;
  const T w15  = w4 +-w11;
  const T w16[2]  =  {(T)(w1 +( kWeight2*w8 -  kWeight2*w10)), (T)(-w3 +-( kWeight2*w10 +  kWeight2*w8))};
  const T w18[2]  =  {(T)(w1 +(-kWeight2*w8 - -kWeight2*w10)), (T)(w3 +(-kWeight2*w10 + -kWeight2*w8))};
  const T w19  = i4 +i36;
  const T w20  = i4 +-i36;
  const T w21  = i20 +i52;
  const T w22  = i20 +-i52;
  const T w23  = w19 +w21;
  const T w24  = w19 +-w21;
  const T w26  = i12 +i44;
  const T w27  = i12 +-i44;
  const T w28  = i28 +i60;
  const T w29  = i28 +-i60;
  const T w30  = w26 +w28;
  const T w31  = w26 +-w28;
  const T w33  = w23 +w30;
  const T w34  = w23 +-w30;
  const T w35[2]  =  {(T)(w20 +( kWeight2*w27 -  kWeight2*w29)), (T)(-w22 +-( kWeight2*w29 +  kWeight2*w27))};
  const T w37[2]  =  {(T)(w20 +(-kWeight2*w27 - -kWeight2*w29)), (T)(w22 +(-kWeight2*w29 + -kWeight2*w27))};
  const T w38  = w14 +w33;
  const T w39  = w14 +-w33;
  const T w40[2]  =  {(T)(w16[0] +( kWeight3*w35[0] - -kWeight4*w35[1])), (T)(w16[1] +( kWeight3*w35[1] + -kWeight4*w35[0]))};
  const T w41[2]  =  {(T)(w5 +( kWeight2*w24 -  kWeight2*w31)), (T)(-w12 +-( kWeight2*w31 +  kWeight2*w24))};
  const T w42[2]  =  {(T)(w18[0] +( kWeight4*w37[0] - -kWeight3*w37[1])), (T)(w18[1] +( kWeight4*w37[1] + -kWeight3*w37[0]))};
  const T w44[2]  =  {(T)(w18[0] +(-kWeight4*w37[0] -  kWeight3*w37[1])), (T)(-w18[1] +-(-kWeight4*w37[1] +  kWeight3*w37[0]))};
  const T w45[2]  =  {(T)(w5 +(-kWeight2*w24 - -kWeight2*w31)), (T)(w12 +(-kWeight2*w31 + -kWeight2*w24))};
  const T w46[2]  =  {(T)(w16[0] +(-kWeight3*w35[0] -  kWeight4*w35[1])), (T)(-w16[1] +-(-kWeight3*w35[1] +  kWeight4*w35[0]))};
  const T w47  = i2 +i34;
  const T w48  = i2 +-i34;
  const T w49  = i18 +i50;
  const T w50  = i18 +-i50;
  const T w51  = w47 +w49;
  const T w52  = w47 +-w49;
  const T w54  = i10 +i42;
  const T w55  = i10 +-i42;
  const T w56  = i26 +i58;
  const T w57  = i26 +-i58;
  const T w58  = w54 +w56;
  const T w59  = w54 +-w56;
  const T w61  = w51 +w58;
  const T w62  = w51 +-w58;
  const T w63[2]  =  {(T)(w48 +( kWeight2*w55 -  kWeight2*w57)), (T)(-w50 +-( kWeight2*w57 +  kWeight2*w55))};
  const T w65[2]  =  {(T)(w48 +(-kWeight2*w55 - -kWeight2*w57)), (T)(w50 +(-kWeight2*w57 + -kWeight2*w55))};
  const T w66  = i6 +i38;
  const T w67  = i6 +-i38;
  const T w68  = i22 +i54;
  const T w69  = i22 +-i54;
  const T w70  = w66 +w68;
  const T w71  = w66 +-w68;
  const T w73  = i14 +i46;
  const T w74  = i14 +-i46;
  const T w75  = i30 +i62;
  const T w76  = i30 +-i62;
  const T w77  = w73 +w75;
  const T w78  = w73 +-w75;
  const T w80  = w70 +w77;
  const T w81  = w70 +-w77;
  const T w82[2]  =  {(T)(w67 +( kWeight2*w74 -  kWeight2*w76)), (T)(-w69 +-( kWeight2*w76 +  kWeight2*w74))};
  const T w84[2]  =  {(T)(w67 +(-kWeight2*w74 - -kWeight2*w76)), (T)(w69 +(-kWeight2*w76 + -kWeight2*w74))};
  const T w85  = w61 +w80;
  const T w86  = w61 +-w80;
  const T w87[2]  =  {(T)(w63[0] +( kWeight3*w82[0] - -kWeight4*w82[1])), (T)(w63[1] +( kWeight3*w82[1] + -kWeight4*w82[0]))};
  const T w88[2]  =  {(T)(w52 +( kWeight2*w71 -  kWeight2*w78)), (T)(-w59 +-( kWeight2*w78 +  kWeight2*w71))};
  const T w89[2]  =  {(T)(w65[0] +( kWeight4*w84[0] - -kWeight3*w84[1])), (T)(w65[1] +( kWeight4*w84[1] + -kWeight3*w84[0]))};
  const T w91[2]  =  {(T)(w65[0] +(-kWeight4*w84[0] -  kWeight3*w84[1])), (T)(-w65[1] +-(-kWeight4*w84[1] +  kWeight3*w84[0]))};
  const T w92[2]  =  {(T)(w52 +(-kWeight2*w71 - -kWeight2*w78)), (T)(w59 +(-kWeight2*w78 + -kWeight2*w71))};
  const T w93[2]  =  {(T)(w63[0] +(-kWeight3*w82[0] -  kWeight4*w82[1])), (T)(-w63[1] +-(-kWeight3*w82[1] +  kWeight4*w82[0]))};
  const T w94  = w38 +w85;
  const T w95  = w38 +-w85;
  const T w96[2]  =  {(T)(w40[0] +( kWeight5*w87[0] - -kWeight6*w87[1])), (T)(w40[1] +( kWeight5*w87[1] + -kWeight6*w87[0]))};
  const T w97[2]  =  {(T)(w41[0] +( kWeight3*w88[0] - -kWeight4*w88[1])), (T)(w41[1] +( kWeight3*w88[1] + -kWeight4*w88[0]))};
  const T w98[2]  =  {(T)(w42[0] +( kWeight7*w89[0] - -kWeight8*w89[1])), (T)(w42[1] +( kWeight7*w89[1] + -kWeight8*w89[0]))};
  const T w99[2]  =  {(T)(w15 +( kWeight2*w62 -  kWeight2*w81)), (T)(-w34 +-( kWeight2*w81 +  kWeight2*w62))};
  const T w100[2]  =  {(T)(w44[0] +( kWeight8*w91[0] - -kWeight7*w91[1])), (T)(w44[1] +( kWeight8*w91[1] + -kWeight7*w91[0]))};
  const T w101[2]  =  {(T)(w45[0] +( kWeight4*w92[0] - -kWeight3*w92[1])), (T)(w45[1] +( kWeight4*w92[1] + -kWeight3*w92[0]))};
  const T w102[2]  =  {(T)(w46[0] +( kWeight6*w93[0] - -kWeight5*w93[1])), (T)(w46[1] +( kWeight6*w93[1] + -kWeight5*w93[0]))};
  const T w104[2]  =  {(T)(w46[0] +(-kWeight6*w93[0] -  kWeight5*w93[1])), (T)(-w46[1] +-(-kWeight6*w93[1] +  kWeight5*w93[0]))};
  const T w105[2]  =  {(T)(w45[0] +(-kWeight4*w92[0] -  kWeight3*w92[1])), (T)(-w45[1] +-(-kWeight4*w92[1] +  kWeight3*w92[0]))};
  const T w106[2]  =  {(T)(w44[0] +(-kWeight8*w91[0] -  kWeight7*w91[1])), (T)(-w44[1] +-(-kWeight8*w91[1] +  kWeight7*w91[0]))};
  const T w107[2]  =  {(T)(w15 +(-kWeight2*w62 - -kWeight2*w81)), (T)(w34 +(-kWeight2*w81 + -kWeight2*w62))};
  const T w108[2]  =  {(T)(w42[0] +(-kWeight7*w89[0] -  kWeight8*w89[1])), (T)(-w42[1] +-(-kWeight7*w89[1] +  kWeight8*w89[0]))};
  const T w109[2]  =  {(T)(w41[0] +(-kWeight3*w88[0] -  kWeight4*w88[1])), (T)(-w41[1] +-(-kWeight3*w88[1] +  kWeight4*w88[0]))};
  const T w110[2]  =  {(T)(w40[0] +(-kWeight5*w87[0] -  kWeight6*w87[1])), (T)(-w40[1] +-(-kWeight5*w87[1] +  kWeight6*w87[0]))};
  const T w111  = i1 +i33;
  const T w112  = i1 +-i33;
  const T w113  = i17 +i49;
  const T w114  = i17 +-i49;
  const T w115  = w111 +w113;
  const T w116  = w111 +-w113;
  const T w118  = i9 +i41;
  const T w119  = i9 +-i41;
  const T w120  = i25 +i57;
  const T w121  = i25 +-i57;
  const T w122  = w118 +w120;
  const T w123  = w118 +-w120;
  const T w125  = w115 +w122;
  const T w126  = w115 +-w122;
  const T w127[2]  =  {(T)(w112 +( kWeight2*w119 -  kWeight2*w121)), (T)(-w114 +-( kWeight2*w121 +  kWeight2*w119))};
  const T w129[2]  =  {(T)(w112 +(-kWeight2*w119 - -kWeight2*w121)), (T)(w114 +(-kWeight2*w121 + -kWeight2*w119))};
  const T w130  = i5 +i37;
  const T w131  = i5 +-i37;
  const T w132  = i21 +i53;
  const T w133  = i21 +-i53;
  const T w134  = w130 +w132;
  const T w135  = w130 +-w132;
  const T w137  = i13 +i45;
  const T w138  = i13 +-i45;
  const T w139  = i29 +i61;
  const T w140  = i29 +-i61;
  const T w141  = w137 +w139;
  const T w142  = w137 +-w139;
  const T w144  = w134 +w141;
  const T w145  = w134 +-w141;
  const T w146[2]  =  {(T)(w131 +( kWeight2*w138 -  kWeight2*w140)), (T)(-w133 +-( kWeight2*w140 +  kWeight2*w138))};
  const T w148[2]  =  {(T)(w131 +(-kWeight2*w138 - -kWeight2*w140)), (T)(w133 +(-kWeight2*w140 + -kWeight2*w138))};
  const T w149  = w125 +w144;
  const T w150  = w125 +-w144;
  const T w151[2]  =  {(T)(w127[0] +( kWeight3*w146[0] - -kWeight4*w146[1])), (T)(w127[1] +( kWeight3*w146[1] + -kWeight4*w146[0]))};
  const T w152[2]  =  {(T)(w116 +( kWeight2*w135 -  kWeight2*w142)), (T)(-w123 +-( kWeight2*w142 +  kWeight2*w135))};
  const T w153[2]  =  {(T)(w129[0] +( kWeight4*w148[0] - -kWeight3*w148[1])), (T)(w129[1] +( kWeight4*w148[1] + -kWeight3*w148[0]))};
  const T w155[2]  =  {(T)(w129[0] +(-kWeight4*w148[0] -  kWeight3*w148[1])), (T)(-w129[1] +-(-kWeight4*w148[1] +  kWeight3*w148[0]))};
  const T w156[2]  =  {(T)(w116 +(-kWeight2*w135 - -kWeight2*w142)), (T)(w123 +(-kWeight2*w142 + -kWeight2*w135))};
  const T w157[2]  =  {(T)(w127[0] +(-kWeight3*w146[0] -  kWeight4*w146[1])), (T)(-w127[1] +-(-kWeight3*w146[1] +  kWeight4*w146[0]))};
  const T w158  = i3 +i35;
  const T w159  = i3 +-i35;
  const T w160  = i19 +i51;
  const T w161  = i19 +-i51;
  const T w162  = w158 +w160;
  const T w163  = w158 +-w160;
  const T w165  = i11 +i43;
  const T w166  = i11 +-i43;
  const T w167  = i27 +i59;
  const T w168  = i27 +-i59;
  const T w169  = w165 +w167;
  const T w170  = w165 +-w167;
  const T w172  = w162 +w169;
  const T w173  = w162 +-w169;
  const T w174[2]  =  {(T)(w159 +( kWeight2*w166 -  kWeight2*w168)), (T)(-w161 +-( kWeight2*w168 +  kWeight2*w166))};
  const T w176[2]  =  {(T)(w159 +(-kWeight2*w166 - -kWeight2*w168)), (T)(w161 +(-kWeight2*w168 + -kWeight2*w166))};
  const T w177  = i7 +i39;
  const T w178  = i7 +-i39;
  const T w179  = i23 +i55;
  const T w180  = i23 +-i55;
  const T w181  = w177 +w179;
  const T w182  = w177 +-w179;
  const T w184  = i15 +i47;
  const T w185  = i15 +-i47;
  const T w186  = i31 +i63;
  const T w187  = i31 +-i63;
  const T w188  = w184 +w186;
  const T w189  = w184 +-w186;
  const T w191  = w181 +w188;
  const T w192  = w181 +-w188;
  const T w193[2]  =  {(T)(w178 +( kWeight2*w185 -  kWeight2*w187)), (T)(-w180 +-( kWeight2*w187 +  kWeight2*w185))};
  const T w195[2]  =  {(T)(w178 +(-kWeight2*w185 - -kWeight2*w187)), (T)(w180 +(-kWeight2*w187 + -kWeight2*w185))};
  const T w196  = w172 +w191;
  const T w197  = w172 +-w191;
  const T w198[2]  =  {(T)(w174[0] +( kWeight3*w193[0] - -kWeight4*w193[1])), (T)(w174[1] +( kWeight3*w193[1] + -kWeight4*w193[0]))};
  const T w199[2]  =  {(T)(w163 +( kWeight2*w182 -  kWeight2*w189)), (T)(-w170 +-( kWeight2*w189 +  kWeight2*w182))};
  const T w200[2]  =  {(T)(w176[0] +( kWeight4*w195[0] - -kWeight3*w195[1])), (T)(w176[1] +( kWeight4*w195[1] + -kWeight3*w195[0]))};
  const T w202[2]  =  {(T)(w176[0] +(-kWeight4*w195[0] -  kWeight3*w195[1])), (T)(-w176[1] +-(-kWeight4*w195[1] +  kWeight3*w195[0]))};
  const T w203[2]  =  {(T)(w163 +(-kWeight2*w182 - -kWeight2*w189)), (T)(w170 +(-kWeight2*w189 + -kWeight2*w182))};
  const T w204[2]  =  {(T)(w174[0] +(-kWeight3*w193[0] -  kWeight4*w193[1])), (T)(-w174[1] +-(-kWeight3*w193[1] +  kWeight4*w193[0]))};
  const T w205  = w149 +w196;
  const T w206  = w149 +-w196;
  const T w207[2]  =  {(T)(w151[0] +( kWeight5*w198[0] - -kWeight6*w198[1])), (T)(w151[1] +( kWeight5*w198[1] + -kWeight6*w198[0]))};
  const T w208[2]  =  {(T)(w152[0] +( kWeight3*w199[0] - -kWeight4*w199[1])), (T)(w152[1] +( kWeight3*w199[1] + -kWeight4*w199[0]))};
  const T w209[2]  =  {(T)(w153[0] +( kWeight7*w200[0] - -kWeight8*w200[1])), (T)(w153[1] +( kWeight7*w200[1] + -kWeight8*w200[0]))};
  const T w210[2]  =  {(T)(w126 +( kWeight2*w173 -  kWeight2*w192)), (T)(-w145 +-( kWeight2*w192 +  kWeight2*w173))};
  const T w211[2]  =  {(T)(w155[0] +( kWeight8*w202[0] - -kWeight7*w202[1])), (T)(w155[1] +( kWeight8*w202[1] + -kWeight7*w202[0]))};
  const T w212[2]  =  {(T)(w156[0] +( kWeight4*w203[0] - -kWeight3*w203[1])), (T)(w156[1] +( kWeight4*w203[1] + -kWeight3*w203[0]))};
  const T w213[2]  =  {(T)(w157[0] +( kWeight6*w204[0] - -kWeight5*w204[1])), (T)(w157[1] +( kWeight6*w204[1] + -kWeight5*w204[0]))};
  const T w215[2]  =  {(T)(w157[0] +(-kWeight6*w204[0] -  kWeight5*w204[1])), (T)(-w157[1] +-(-kWeight6*w204[1] +  kWeight5*w204[0]))};
  const T w216[2]  =  {(T)(w156[0] +(-kWeight4*w203[0] -  kWeight3*w203[1])), (T)(-w156[1] +-(-kWeight4*w203[1] +  kWeight3*w203[0]))};
  const T w217[2]  =  {(T)(w155[0] +(-kWeight8*w202[0] -  kWeight7*w202[1])), (T)(-w155[1] +-(-kWeight8*w202[1] +  kWeight7*w202[0]))};
  const T w218[2]  =  {(T)(w126 +(-kWeight2*w173 - -kWeight2*w192)), (T)(w145 +(-kWeight2*w192 + -kWeight2*w173))};
  const T w219[2]  =  {(T)(w153[0] +(-kWeight7*w200[0] -  kWeight8*w200[1])), (T)(-w153[1] +-(-kWeight7*w200[1] +  kWeight8*w200[0]))};
  const T w220[2]  =  {(T)(w152[0] +(-kWeight3*w199[0] -  kWeight4*w199[1])), (T)(-w152[1] +-(-kWeight3*w199[1] +  kWeight4*w199[0]))};
  const T w221[2]  =  {(T)(w151[0] +(-kWeight5*w198[0] -  kWeight6*w198[1])), (T)(-w151[1] +-(-kWeight5*w198[1] +  kWeight6*w198[0]))};
  store(output + 0 * stride, w94 +w205);
  store(output + 1 * stride, w96[0] +( kWeight9*w207[0] - -kWeight10*w207[1]));
  store(output + 2 * stride, w97[0] +( kWeight5*w208[0] - -kWeight6*w208[1]));
  store(output + 3 * stride, w98[0] +( kWeight11*w209[0] - -kWeight12*w209[1]));
  store(output + 4 * stride, w99[0] +( kWeight3*w210[0] - -kWeight4*w210[1]));
  store(output + 5 * stride, w100[0] +( kWeight13*w211[0] - -kWeight14*w211[1]));
  store(output + 6 * stride, w101[0] +( kWeight7*w212[0] - -kWeight8*w212[1]));
  store(output + 7 * stride, w102[0] +( kWeight15*w213[0] - -kWeight16*w213[1]));
  store(output + 8 * stride, w39 +( kWeight2*w150 -  kWeight2*w197));
  store(output + 9 * stride, w104[0] +( kWeight16*w215[0] - -kWeight15*w215[1]));
  store(output + 10 * stride, w105[0] +( kWeight8*w216[0] - -kWeight7*w216[1]));
  store(output + 11 * stride, w106[0] +( kWeight14*w217[0] - -kWeight13*w217[1]));
  store(output + 12 * stride, w107[0] +( kWeight4*w218[0] - -kWeight3*w218[1]));
  store(output + 13 * stride, w108[0] +( kWeight12*w219[0] - -kWeight11*w219[1]));
  store(output + 14 * stride, w109[0] +( kWeight6*w220[0] - -kWeight5*w220[1]));
  store(output + 15 * stride, w110[0] +( kWeight10*w221[0] - -kWeight9*w221[1]));
  store(output + 16 * stride, w95);
  store(output + 17 * stride, w110[0] +(-kWeight10*w221[0] -  kWeight9*w221[1]));
  store(output + 18 * stride, w109[0] +(-kWeight6*w220[0] -  kWeight5*w220[1]));
  store(output + 19 * stride, w108[0] +(-kWeight12*w219[0] -  kWeight11*w219[1]));
  store(output + 20 * stride, w107[0] +(-kWeight4*w218[0] -  kWeight3*w218[1]));
  store(output + 21 * stride, w106[0] +(-kWeight14*w217[0] -  kWeight13*w217[1]));
  store(output + 22 * stride, w105[0] +(-kWeight8*w216[0] -  kWeight7*w216[1]));
  store(output + 23 * stride, w104[0] +(-kWeight16*w215[0] -  kWeight15*w215[1]));
  store(output + 24 * stride, w39 +(-kWeight2*w150 - -kWeight2*w197));
  store(output + 25 * stride, w102[0] +(-kWeight15*w213[0] -  kWeight16*w213[1]));
  store(output + 26 * stride, w101[0] +(-kWeight7*w212[0] -  kWeight8*w212[1]));
  store(output + 27 * stride, w100[0] +(-kWeight13*w211[0] -  kWeight14*w211[1]));
  store(output + 28 * stride, w99[0] +(-kWeight3*w210[0] -  kWeight4*w210[1]));
  store(output + 29 * stride, w98[0] +(-kWeight11*w209[0] -  kWeight12*w209[1]));
  store(output + 30 * stride, w97[0] +(-kWeight5*w208[0] -  kWeight6*w208[1]));
  store(output + 31 * stride, w96[0] +(-kWeight9*w207[0] -  kWeight10*w207[1]));
  store(output + 32 * stride, w94 +-w205);
  store(output + 33 * stride, w96[1] +( kWeight9*w207[1] + -kWeight10*w207[0]));
  store(output + 34 * stride, w97[1] +( kWeight5*w208[1] + -kWeight6*w208[0]));
  store(output + 35 * stride, w98[1] +( kWeight11*w209[1] + -kWeight12*w209[0]));
  store(output + 36 * stride, w99[1] +( kWeight3*w210[1] + -kWeight4*w210[0]));
  store(output + 37 * stride, w100[1] +( kWeight13*w211[1] + -kWeight14*w211[0]));
  store(output + 38 * stride, w101[1] +( kWeight7*w212[1] + -kWeight8*w212[0]));
  store(output + 39 * stride, w102[1] +( kWeight15*w213[1] + -kWeight16*w213[0]));
  store(output + 40 * stride, -w86 +-( kWeight2*w197 +  kWeight2*w150));
  store(output + 41 * stride, w104[1] +( kWeight16*w215[1] + -kWeight15*w215[0]));
  store(output + 42 * stride, w105[1] +( kWeight8*w216[1] + -kWeight7*w216[0]));
  store(output + 43 * stride, w106[1] +( kWeight14*w217[1] + -kWeight13*w217[0]));
  store(output + 44 * stride, w107[1] +( kWeight4*w218[1] + -kWeight3*w218[0]));
  store(output + 45 * stride, w108[1] +( kWeight12*w219[1] + -kWeight11*w219[0]));
  store(output + 46 * stride, w109[1] +( kWeight6*w220[1] + -kWeight5*w220[0]));
  store(output + 47 * stride, w110[1] +( kWeight10*w221[1] + -kWeight9*w221[0]));
  store(output + 48 * stride, -w206);
  store(output + 49 * stride, -w110[1] +-(-kWeight10*w221[1] +  kWeight9*w221[0]));
  store(output + 50 * stride, -w109[1] +-(-kWeight6*w220[1] +  kWeight5*w220[0]));
  store(output + 51 * stride, -w108[1] +-(-kWeight12*w219[1] +  kWeight11*w219[0]));
  store(output + 52 * stride, -w107[1] +-(-kWeight4*w218[1] +  kWeight3*w218[0]));
  store(output + 53 * stride, -w106[1] +-(-kWeight14*w217[1] +  kWeight13*w217[0]));
  store(output + 54 * stride, -w105[1] +-(-kWeight8*w216[1] +  kWeight7*w216[0]));
  store(output + 55 * stride, -w104[1] +-(-kWeight16*w215[1] +  kWeight15*w215[0]));
  store(output + 56 * stride, w86 +(-kWeight2*w197 + -kWeight2*w150));
  store(output + 57 * stride, -w102[1] +-(-kWeight15*w213[1] +  kWeight16*w213[0]));
  store(output + 58 * stride, -w101[1] +-(-kWeight7*w212[1] +  kWeight8*w212[0]));
  store(output + 59 * stride, -w100[1] +-(-kWeight13*w211[1] +  kWeight14*w211[0]));
  store(output + 60 * stride, -w99[1] +-(-kWeight3*w210[1] +  kWeight4*w210[0]));
  store(output + 61 * stride, -w98[1] +-(-kWeight11*w209[1] +  kWeight12*w209[0]));
  store(output + 62 * stride, -w97[1] +-(-kWeight5*w208[1] +  kWeight6*w208[0]));
  store(output + 63 * stride, -w96[1] +-(-kWeight9*w207[1] +  kWeight10*w207[0]));
}
