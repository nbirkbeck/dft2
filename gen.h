#include <complex>



void dft_1(float* input, std::complex<float>* output) {
  const float kWeight0 = 0;
  const float kWeight1 = 1;
  output[0] = input[0];
}

void dft_2(float* input, std::complex<float>* output) {
  const float kWeight0 = 0;
  const float kWeight1 = 1;
  output[0] =  input[0]  +  input[1] ;
  output[1] =  input[0]  +  -input[1] ;
}

// Num expressions: 11
// Weights: 2
void dft_4(float* input, std::complex<float>* output) {
  const float kWeight0 = 0;
  const float kWeight1 = 1;
  const std::complex<float> w0 = input[0]  + input[2] ;
  const std::complex<float> w1 = input[0]  + -input[2] ;
  const std::complex<float> w2 = input[1]  + input[3] ;
  const std::complex<float> w3 = input[1]  + -input[3] ;
  output[0] = w0  + w2 ;
  output[2] = w0  + -w2 ;
  output[1] = w1  +  std::complex<float>( kWeight0, -kWeight1) * w3;
}

void dft_8(float* input, std::complex<float>* output) {
  const float kWeight0 = 0;
  const float kWeight1 = 1;
  const float kWeight2 = 0.707107;
  const std::complex<float> w0 = input[0]  + input[4] ;
  const std::complex<float> w1 = input[0]  + -input[4] ;
  const std::complex<float> w2 = input[2]  + input[6] ;
  const std::complex<float> w3 = input[2]  + -input[6] ;
  const std::complex<float> w4 = w0  + w2 ;
  const std::complex<float> w5 = w0  + -w2 ;
  const std::complex<float> w6 = w1  +  std::complex<float>( kWeight0, -kWeight1) * w3;
  const std::complex<float> w7 = input[1]  + input[5] ;
  const std::complex<float> w8 = input[1]  + -input[5] ;
  const std::complex<float> w9 = input[3]  + input[7] ;
  const std::complex<float> w10 = input[3]  + -input[7] ;
  const std::complex<float> w11 = w7  + w9 ;
  const std::complex<float> w12 = w7  + -w9 ;
  const std::complex<float> w13 = w8  +  std::complex<float>( kWeight0, -kWeight1) * w10;
  output[0] = w4  + w11 ;
  output[4] = w4  + -w11 ;
  output[1] = w6  +  std::complex<float>( kWeight2, -kWeight2) * w13;
  output[2] = w5  +  std::complex<float>( kWeight0, -kWeight1) * w12;
  output[3] = std::conj(w6)  +  std::complex<float>(-kWeight2, -kWeight2) * std::conj(w13);
}


void dft_16(float* input, std::complex<float>* output) {
  const float kWeight0 = 0;
  const float kWeight1 = 1;
  const float kWeight2 = 0.707107;
  const float kWeight3 = 0.92388;
  const float kWeight4 = 0.382683;
  const std::complex<float> w0 = input[0]  + input[8] ;
  const std::complex<float> w1 = input[0]  + -input[8] ;
  const std::complex<float> w2 = input[4]  + input[12] ;
  const std::complex<float> w3 = input[4]  + -input[12] ;
  const std::complex<float> w4 = w0  + w2 ;
  const std::complex<float> w5 = w0  + -w2 ;
  const std::complex<float> w6 = w1  +  std::complex<float>( kWeight0, -kWeight1) * w3;
  const std::complex<float> w7 = input[2]  + input[10] ;
  const std::complex<float> w8 = input[2]  + -input[10] ;
  const std::complex<float> w9 = input[6]  + input[14] ;
  const std::complex<float> w10 = input[6]  + -input[14] ;
  const std::complex<float> w11 = w7  + w9 ;
  const std::complex<float> w12 = w7  + -w9 ;
  const std::complex<float> w13 = w8  +  std::complex<float>( kWeight0, -kWeight1) * w10;
  const std::complex<float> w14 = w4  + w11 ;
  const std::complex<float> w15 = w4  + -w11 ;
  const std::complex<float> w16 = w6  +  std::complex<float>( kWeight2, -kWeight2) * w13;
  const std::complex<float> w17 = w5  +  std::complex<float>( kWeight0, -kWeight1) * w12;
  const std::complex<float> w18 = std::conj(w6)  +  std::complex<float>(-kWeight2, -kWeight2) * std::conj(w13);
  const std::complex<float> w19 = input[1]  + input[9] ;
  const std::complex<float> w20 = input[1]  + -input[9] ;
  const std::complex<float> w21 = input[5]  + input[13] ;
  const std::complex<float> w22 = input[5]  + -input[13] ;
  const std::complex<float> w23 = w19  + w21 ;
  const std::complex<float> w24 = w19  + -w21 ;
  const std::complex<float> w25 = w20  +  std::complex<float>( kWeight0, -kWeight1) * w22;
  const std::complex<float> w26 = input[3]  + input[11] ;
  const std::complex<float> w27 = input[3]  + -input[11] ;
  const std::complex<float> w28 = input[7]  + input[15] ;
  const std::complex<float> w29 = input[7]  + -input[15] ;
  const std::complex<float> w30 = w26  + w28 ;
  const std::complex<float> w31 = w26  + -w28 ;
  const std::complex<float> w32 = w27  +  std::complex<float>( kWeight0, -kWeight1) * w29;
  const std::complex<float> w33 = w23  + w30 ;
  const std::complex<float> w34 = w23  + -w30 ;
  const std::complex<float> w35 = w25  +  std::complex<float>( kWeight2, -kWeight2) * w32;
  const std::complex<float> w36 = w24  +  std::complex<float>( kWeight0, -kWeight1) * w31;
  const std::complex<float> w37 = std::conj(w25)  +  std::complex<float>(-kWeight2, -kWeight2) * std::conj(w32);
  output[0] = w14  + w33 ;
  output[8] = w14  + -w33 ;
  output[1] = w16  +  std::complex<float>( kWeight3, -kWeight4) * w35;
  output[2] = w17  +  std::complex<float>( kWeight2, -kWeight2) * w36;
  output[3] = w18  +  std::complex<float>( kWeight4, -kWeight3) * w37;
  output[4] = w15  +  std::complex<float>( kWeight0, -kWeight1) * w34;
  output[5] = std::conj(w18)  +  std::complex<float>(-kWeight4, -kWeight3) * std::conj(w37);
  output[6] = std::conj(w17)  +  std::complex<float>(-kWeight2, -kWeight2) * std::conj(w36);
  output[7] = std::conj(w16)  +  std::complex<float>(-kWeight3, -kWeight4) * std::conj(w35);
}


void dft_32(float* input, std::complex<float>* output) {
  const float kWeight0 = 0;
  const float kWeight1 = 1;
  const float kWeight2 = 0.707107;
  const float kWeight3 = 0.92388;
  const float kWeight4 = 0.382683;
  const float kWeight5 = 0.980785;
  const float kWeight6 = 0.19509;
  const float kWeight7 = 0.83147;
  const float kWeight8 = 0.55557;
  const std::complex<float> w0 = input[0]  + input[16] ;
  const std::complex<float> w1 = input[0]  + -input[16] ;
  const std::complex<float> w2 = input[8]  + input[24] ;
  const std::complex<float> w3 = input[8]  + -input[24] ;
  const std::complex<float> w4 = w0  + w2 ;
  const std::complex<float> w5 = w0  + -w2 ;
  const std::complex<float> w6 = w1  +  std::complex<float>( kWeight0, -kWeight1) * w3;
  const std::complex<float> w7 = input[4]  + input[20] ;
  const std::complex<float> w8 = input[4]  + -input[20] ;
  const std::complex<float> w9 = input[12]  + input[28] ;
  const std::complex<float> w10 = input[12]  + -input[28] ;
  const std::complex<float> w11 = w7  + w9 ;
  const std::complex<float> w12 = w7  + -w9 ;
  const std::complex<float> w13 = w8  +  std::complex<float>( kWeight0, -kWeight1) * w10;
  const std::complex<float> w14 = w4  + w11 ;
  const std::complex<float> w15 = w4  + -w11 ;
  const std::complex<float> w16 = w6  +  std::complex<float>( kWeight2, -kWeight2) * w13;
  const std::complex<float> w17 = w5  +  std::complex<float>( kWeight0, -kWeight1) * w12;
  const std::complex<float> w18 = std::conj(w6)  +  std::complex<float>(-kWeight2, -kWeight2) * std::conj(w13);
  const std::complex<float> w19 = input[2]  + input[18] ;
  const std::complex<float> w20 = input[2]  + -input[18] ;
  const std::complex<float> w21 = input[10]  + input[26] ;
  const std::complex<float> w22 = input[10]  + -input[26] ;
  const std::complex<float> w23 = w19  + w21 ;
  const std::complex<float> w24 = w19  + -w21 ;
  const std::complex<float> w25 = w20  +  std::complex<float>( kWeight0, -kWeight1) * w22;
  const std::complex<float> w26 = input[6]  + input[22] ;
  const std::complex<float> w27 = input[6]  + -input[22] ;
  const std::complex<float> w28 = input[14]  + input[30] ;
  const std::complex<float> w29 = input[14]  + -input[30] ;
  const std::complex<float> w30 = w26  + w28 ;
  const std::complex<float> w31 = w26  + -w28 ;
  const std::complex<float> w32 = w27  +  std::complex<float>( kWeight0, -kWeight1) * w29;
  const std::complex<float> w33 = w23  + w30 ;
  const std::complex<float> w34 = w23  + -w30 ;
  const std::complex<float> w35 = w25  +  std::complex<float>( kWeight2, -kWeight2) * w32;
  const std::complex<float> w36 = w24  +  std::complex<float>( kWeight0, -kWeight1) * w31;
  const std::complex<float> w37 = std::conj(w25)  +  std::complex<float>(-kWeight2, -kWeight2) * std::conj(w32);
  const std::complex<float> w38 = w14  + w33 ;
  const std::complex<float> w39 = w14  + -w33 ;
  const std::complex<float> w40 = w16  +  std::complex<float>( kWeight3, -kWeight4) * w35;
  const std::complex<float> w41 = w17  +  std::complex<float>( kWeight2, -kWeight2) * w36;
  const std::complex<float> w42 = w18  +  std::complex<float>( kWeight4, -kWeight3) * w37;
  const std::complex<float> w43 = w15  +  std::complex<float>( kWeight0, -kWeight1) * w34;
  const std::complex<float> w44 = std::conj(w18)  +  std::complex<float>(-kWeight4, -kWeight3) * std::conj(w37);
  const std::complex<float> w45 = std::conj(w17)  +  std::complex<float>(-kWeight2, -kWeight2) * std::conj(w36);
  const std::complex<float> w46 = std::conj(w16)  +  std::complex<float>(-kWeight3, -kWeight4) * std::conj(w35);
  const std::complex<float> w47 = input[1]  + input[17] ;
  const std::complex<float> w48 = input[1]  + -input[17] ;
  const std::complex<float> w49 = input[9]  + input[25] ;
  const std::complex<float> w50 = input[9]  + -input[25] ;
  const std::complex<float> w51 = w47  + w49 ;
  const std::complex<float> w52 = w47  + -w49 ;
  const std::complex<float> w53 = w48  +  std::complex<float>( kWeight0, -kWeight1) * w50;
  const std::complex<float> w54 = input[5]  + input[21] ;
  const std::complex<float> w55 = input[5]  + -input[21] ;
  const std::complex<float> w56 = input[13]  + input[29] ;
  const std::complex<float> w57 = input[13]  + -input[29] ;
  const std::complex<float> w58 = w54  + w56 ;
  const std::complex<float> w59 = w54  + -w56 ;
  const std::complex<float> w60 = w55  +  std::complex<float>( kWeight0, -kWeight1) * w57;
  const std::complex<float> w61 = w51  + w58 ;
  const std::complex<float> w62 = w51  + -w58 ;
  const std::complex<float> w63 = w53  +  std::complex<float>( kWeight2, -kWeight2) * w60;
  const std::complex<float> w64 = w52  +  std::complex<float>( kWeight0, -kWeight1) * w59;
  const std::complex<float> w65 = std::conj(w53)  +  std::complex<float>(-kWeight2, -kWeight2) * std::conj(w60);
  const std::complex<float> w66 = input[3]  + input[19] ;
  const std::complex<float> w67 = input[3]  + -input[19] ;
  const std::complex<float> w68 = input[11]  + input[27] ;
  const std::complex<float> w69 = input[11]  + -input[27] ;
  const std::complex<float> w70 = w66  + w68 ;
  const std::complex<float> w71 = w66  + -w68 ;
  const std::complex<float> w72 = w67  +  std::complex<float>( kWeight0, -kWeight1) * w69;
  const std::complex<float> w73 = input[7]  + input[23] ;
  const std::complex<float> w74 = input[7]  + -input[23] ;
  const std::complex<float> w75 = input[15]  + input[31] ;
  const std::complex<float> w76 = input[15]  + -input[31] ;
  const std::complex<float> w77 = w73  + w75 ;
  const std::complex<float> w78 = w73  + -w75 ;
  const std::complex<float> w79 = w74  +  std::complex<float>( kWeight0, -kWeight1) * w76;
  const std::complex<float> w80 = w70  + w77 ;
  const std::complex<float> w81 = w70  + -w77 ;
  const std::complex<float> w82 = w72  +  std::complex<float>( kWeight2, -kWeight2) * w79;
  const std::complex<float> w83 = w71  +  std::complex<float>( kWeight0, -kWeight1) * w78;
  const std::complex<float> w84 = std::conj(w72)  +  std::complex<float>(-kWeight2, -kWeight2) * std::conj(w79);
  const std::complex<float> w85 = w61  + w80 ;
  const std::complex<float> w86 = w61  + -w80 ;
  const std::complex<float> w87 = w63  +  std::complex<float>( kWeight3, -kWeight4) * w82;
  const std::complex<float> w88 = w64  +  std::complex<float>( kWeight2, -kWeight2) * w83;
  const std::complex<float> w89 = w65  +  std::complex<float>( kWeight4, -kWeight3) * w84;
  const std::complex<float> w90 = w62  +  std::complex<float>( kWeight0, -kWeight1) * w81;
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
  output[8] = w39  +  std::complex<float>( kWeight0, -kWeight1) * w86;
  output[9] = std::conj(w46)  +  std::complex<float>(-kWeight6, -kWeight5) * std::conj(w93);
  output[10] = std::conj(w45)  +  std::complex<float>(-kWeight4, -kWeight3) * std::conj(w92);
  output[11] = std::conj(w44)  +  std::complex<float>(-kWeight8, -kWeight7) * std::conj(w91);
  output[12] = std::conj(w43)  +  std::complex<float>(-kWeight2, -kWeight2) * std::conj(w90);
  output[13] = std::conj(w42)  +  std::complex<float>(-kWeight7, -kWeight8) * std::conj(w89);
  output[14] = std::conj(w41)  +  std::complex<float>(-kWeight3, -kWeight4) * std::conj(w88);
  output[15] = std::conj(w40)  +  std::complex<float>(-kWeight5, -kWeight6) * std::conj(w87);
}
