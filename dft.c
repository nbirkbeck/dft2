#include <complex>
#include <vector>
#include <math.h>
#include <stdio.h>
#include <memory>
#include <map>
#include <unordered_map>
#include <string.h>
#include <fftw3.h>
#include <xmmintrin.h>
#include <immintrin.h>
#include "simd.c"

template <typename T, typename InputType=float>
struct SimdHelper {
  static const T constant(const InputType& f);
  static const T load(const InputType* f);
  static void store(InputType* f, const T&);
};

template <>
const float SimdHelper<float, float>::constant(const float& f) {
  return f;
}
template <>
const float SimdHelper<float, float>::load(const float* f) {
  return *f;
}
template <>
void SimdHelper<float, float>::store(float* output, const float& f) {
  *output = f;
}

template <>
const double SimdHelper<double, double>::constant(const double& f) {
  return f;
}

template <>
const double SimdHelper<double, double>::load(const double* f) {
  return *f;
}

template <>
void SimdHelper<double, double>::store(double* output, const double& f) {
  *output = f;
}

template <>
const __m256 SimdHelper<__m256, float>::constant(const float& f) {
  return _mm256_set1_ps(f);
}
template <>
const __m256 SimdHelper<__m256, float>::load(const float* f) {
  return _mm256_load_ps(f);
}
template <>
void SimdHelper<__m256, float>::store(float* output, const __m256& f) {
  _mm256_store_ps(output, f);
}

template <>
const __m256d SimdHelper<__m256d, double>::constant(const double& f) {
  return _mm256_set1_pd(f);
}
template <>
const __m256d SimdHelper<__m256d, double>::load(const double* f) {
  return _mm256_load_pd(f);
}
template <>
void SimdHelper<__m256d, double>::store(double* output, const __m256d& f) {
  _mm256_store_pd(output, f);
}

template <>
const __m128 SimdHelper<__m128, float>::constant(const float& f) {
  return _mm_set1_ps(f);
}

template <>
const __m128 SimdHelper<__m128, float>::load(const float* f) {
  return _mm_load_ps(f);
}

template <>
void SimdHelper<__m128, float>::store(float* output, const __m128& f) {
  _mm_store_ps(output, f);
}

template <>
const __m128d SimdHelper<__m128d, double>::constant(const double& f) {
  return _mm_set1_pd(f);
}

template <>
const __m128d SimdHelper<__m128d, double>::load(const double* f) {
  return _mm_load_pd(f);
}

template <>
void SimdHelper<__m128d, double>::store(double* output, const __m128d& f) {
  _mm_store_pd(output, f);
}

template <typename T>
T load(float f);

template <>
float load(float f) {
  return f;
}
/*
template <>
__m128 load(float f) {
  return _mm_set1_ps(f);
}
*/
template <>
__m256 load(float f) {
  return _mm256_set1_ps(f);
}

#include "gen.h"

struct Timer {
  void start() {
    start_ = ((double)clock()) / CLOCKS_PER_SEC;
  }
  double stop() {
    stop_ = ((double)clock()) / CLOCKS_PER_SEC;
    return stop_ - start_;
  }
  double start_;
  double stop_;
};

template <typename T>
T* aligned_floats(int n) {
  T* f = (T*)malloc(sizeof(T)*(n + 32));
  while (((uint64_t)f) % 128 != 0) {
    f++;
  }
  return f;
}

template <typename InputType>
struct dft_funs {
  typedef void (*dft_1d_t)(const InputType*, InputType*, int);
  typedef void (*dft_2d_t)(
      const InputType* input, std::complex<InputType>* output, int n,
      dft_1d_t tform);
};

template <typename InputType=float>
void dft(InputType* data, std::complex<InputType>* result, int n) {
  if (n == 1) {
    result[0] = data[0];
    return;
  }
  std::vector<InputType> temp(n);
  for (int k = 0; k < n/2; ++k) {
    temp[k] = data[2 * k];
    temp[n/2 + k] = data[2 * k + 1];
  }
  dft(&temp[0], result, n / 2);
  dft(&temp[0] + n / 2, result + n / 2, n / 2);
  for (int k = 0; k < n / 2; ++k) {
    std::complex<InputType> w = std::complex<InputType>(
        cos(2. * M_PI * k / n), -sin(2. * M_PI * k / n));
    std::complex<InputType> a = result[k];
    std::complex<InputType> b = result[n / 2 + k];
    result[k] = a + w * b;
    result[n / 2 + k] = a - w * b;
  }
}

template <typename InputType=float>
std::vector<std::complex<InputType> > dft(std::vector<InputType> &data) {
  std::vector<std::complex<InputType> > result(data.size());
  dft(&data[0], &result[0], data.size());
  return result;
  /*
  const int n = data.size();
  std::vector<std::complex<double> > result(n);

  for (int i = 0; i < n; ++i) {
    std::complex<double> c;
    for (int k = 0; k < n; ++k) {
      c.real() += cos(i * k * 2. * M_PI / n) * data[k];
      c.imag() += sin(i * k * 2. * M_PI / n) * data[k];
    }
    result[i] = c;
  }
  return result;
  */
}

template <class InputType=float>
void validate_1d(int n,
                 void (*func)(const InputType*, InputType*, int)) {
  std::vector<InputType> d(n);
  std::vector<std::complex<InputType> > act(n);
  std::vector<InputType> compact(n);
  int num_different = 0;
  double total_diff;

  for (int i = 0; i < n; ++i) {
    d[i] = 1;
    std::vector<std::complex<InputType> > exp = dft<InputType>(d);
    func(&d[0], &compact[0], 1);
    for (int k = 0; k <= n / 2; k++) {
      act[k] = std::complex<InputType>(
          compact[k],
          k > 0 && k < n / 2 ? compact[n/2 + k] : 0);
    }

    for (int k = 0; k <= n / 2; ++k) {
      //printf("%f+%fi ", act[k].real(), act[k].imag());
      if (std::abs(act[k].imag() - exp[k].imag()) > 1e-4 ||
          std::abs(act[k].real() - exp[k].real()) > 1e-4) {
        num_different++;
        printf("%f+%fi  %f+%fi on %d,%d\n",
               exp[k].real(), exp[k].imag(),
               act[k].real(), act[k].imag(), i, k);
      }
      total_diff += std::abs(act[k] - exp[k]);
      //printf("%f+%fi ", res[k].real(), res[k].imag());
    }
    //printf("\n");
    d[i] = 0;
  }
  printf("%d  %s (%f)\n", n, num_different == 0 ? "OK" : "FAILED", total_diff / n);
}

template <typename T>
void transpose4x4_simd(const T *A, T *B, const int lda, const int ldb);

template <>
void transpose4x4_simd(const float *A, float *B, const int lda, const int ldb) {
  __m128 row1 = _mm_load_ps(&A[0*lda]);
  __m128 row2 = _mm_load_ps(&A[1*lda]);
  __m128 row3 = _mm_load_ps(&A[2*lda]);
  __m128 row4 = _mm_load_ps(&A[3*lda]);
  _MM_TRANSPOSE4_PS(row1, row2, row3, row4);
  _mm_store_ps(&B[0*ldb], row1);
  _mm_store_ps(&B[1*ldb], row2);
  _mm_store_ps(&B[2*ldb], row3);
  _mm_store_ps(&B[3*ldb], row4);
}

template <>
void transpose4x4_simd(const double *A, double *B, const int lda, const int ldb) {
  __m256i row0 = _mm256_load_si256((const __m256i*)&A[0 * lda]);
  __m256i row1 = _mm256_load_si256((const __m256i*)&A[1 * lda]);
  __m256i row2 = _mm256_load_si256((const __m256i*)&A[2 * lda]);
  __m256i row3 = _mm256_load_si256((const __m256i*)&A[3 * lda]);
  __m256i tmp3, tmp2, tmp1, tmp0;
  tmp0 = _mm256_unpacklo_epi64(row0, row1);
  tmp1 = _mm256_unpackhi_epi64(row0, row1);
  tmp2 = _mm256_unpacklo_epi64(row2, row3);
  tmp3 = _mm256_unpackhi_epi64(row2, row3);
  _mm256_store_si256((__m256i*)&B[0 * ldb], _mm256_permute2x128_si256(tmp0, tmp2, 0x20));
  _mm256_store_si256((__m256i*)&B[1 * ldb], _mm256_permute2x128_si256(tmp1, tmp3, 0x20));
  _mm256_store_si256((__m256i*)&B[2 * ldb], _mm256_permute2x128_si256(tmp0, tmp2, 0x31));
  _mm256_store_si256((__m256i*)&B[3 * ldb], _mm256_permute2x128_si256(tmp1, tmp3, 0x31));
}

inline void transpose8x8_simd(const float *A, float *B) {
  transpose4x4_simd(A, B, 8, 8);
  transpose4x4_simd(A + 4 * 8, B + 4, 8, 8);
  transpose4x4_simd(A + 4, B + 4 * 8, 8, 8);
  transpose4x4_simd(A + 4 * 8 + 4, B + 4 * 8 + 4, 8, 8);
}

template <typename InputType>
inline void simple_transpose(const InputType* A, InputType* B, int n) {
  for (int y = 0; y < n; y++) {
    for (int x = 0; x < n; x++) {
      B[y * n + x] = A[x * n + y];
    }
  }
}

template <typename InputType>
inline void transpose(const InputType* A, InputType* B, int n) {
  if (n <= 2) {
    simple_transpose(A, B, n);
  } else {
    for (int y = 0; y < n; y += 4) {
      for (int x = 0; x < n; x += 4) {
        transpose4x4_simd(A + y * n + x,
                          B + x * n + y, n, n);
      }
    }
  }
}

template <typename T>
inline void unpack_2d_output(const T* col_fft,
                      std::complex<T>* output, int n) {
  for (int y = 0; y <= n/2; ++y) {
    const int y2 = y + n/2;
    const bool y_extra = y2 > n/2 && y2 < n;

    for (int x = 0; x <= n/2; ++x) {
      const int x2 = x + n/2;
      const bool x_extra = x2 > n/2 && x2 < n;
      output[y * n + x] = std::complex<float>(
          col_fft[y * n + x] -
          (x_extra && y_extra ? col_fft[y2 * n + x2] : 0),
          (y_extra ? col_fft[y2 * n + x] : 0) +
          (x_extra ? col_fft[y * n + x2] : 0 ));

      if (y_extra) {
        output[(n - y) * n + x] = std::complex<float>(
            col_fft[y * n + x] +
            (x_extra && y_extra ? col_fft[y2 * n + x2] : 0),
            -(y_extra ? col_fft[y2 * n + x] : 0) +
            (x_extra ? col_fft[y * n + x2] : 0 ));
      }
    }
  }
}

template <>
void unpack_2d_output(const float* packed,
                      std::complex<float>* output, int n) {
  const int n2 = n / 2;
  output[0] = packed[0];
  output[n2 * n] =  packed[n2 * n];

  output[n2] =  packed[n2];
  output[n2 * n + n2] =  packed[n2 * n + n2];

  for (int c = 1; c < n2; ++c) {
    output[0 * n + c] = std::complex<float>(packed[c],
                                            packed[c + n2]);
    output[n2 * n + c] = std::complex<float>(packed[n2 * n + c],
                                            packed[n2 * n + c + n2]);
  }
  for (int r = 1; r < n2; ++r) {
    output[r * n + 0] = std::complex<float>(
        packed[r * n], packed[(r + n2) * n]);
    output[r * n + n2] = std::complex<float>(
        packed[r * n + n2], packed[(r + n2) * n + n2]);

    for (int c = 1; c < std::min(n2, 4); ++c) {
      output[r * n + c] = std::complex<float>(
          packed[r * n + c] - packed[(r + n2) * n + c + n2],
          packed[(r + n2) * n + c] +
          packed[r * n + c + n2]);
    }

    for (int c = 4; c < n2; c += 4) {
      __m128 real1 = _mm_load_ps(packed + r * n + c);
      __m128 real2 = _mm_load_ps(packed + (r + n2) * n + c + n2);
      __m128 imag1 = _mm_load_ps(packed + (r + n2) * n + c);
      __m128 imag2 = _mm_load_ps(packed + r * n + c + n2);
      real1 = real1 - real2;
      imag1 = imag1 + imag2;
      _mm_store_ps((float*)(output + r * n + c),
                   _mm_unpacklo_ps(real1, imag1));
      _mm_store_ps((float*)(output + r * n + c + 2),
                   _mm_unpackhi_ps(real1, imag1));
      /*output[r * n + c] = std::complex<float>(
          packed[r * n + c] - packed[(r + n2) * n + c + n2],
          packed[(r + n2) * n + c] +
          packed[r * n + c + n2]);*/
    }

    int r2 = r + n2;
    int r3 = n - r2;
    output[r2 * n + 0] = std::complex<float>(
        packed[r3 * n], -packed[(r3 + n2) * n]);
    output[r2 * n + n2] = std::complex<float>(
        packed[r3 * n + n2], -packed[(r3 + n2) * n + n2]);
    for (int c = 1; c < std::min(4, n2); ++c) {
      output[r2 * n + c] = std::complex<float>(
          packed[r3 * n + c] + packed[(r3 + n2) * n + c + n2],
          -packed[(r3 + n2) * n + c] +
          packed[r3 * n + c + n2]);
    }
    for (int c = 4; c < n2; c += 4) {
      __m128 real1 = _mm_load_ps(packed + r3 * n + c);
      __m128 real2 = _mm_load_ps(packed + (r3 + n2) * n + c + n2);
      __m128 imag1 = _mm_load_ps(packed + (r3 + n2) * n + c);
      __m128 imag2 = _mm_load_ps(packed + r3 * n + c + n2);
      real1 = real1 + real2;
      imag1 = imag2 -imag1;
      _mm_store_ps((float*)(output + r2 * n + c),
                   _mm_unpacklo_ps(real1, imag1));
      _mm_store_ps((float*)(output + r2 * n + c + 2),
                   _mm_unpackhi_ps(real1, imag1));
    }
  }
}

template <>
void unpack_2d_output(const double* packed,
                      std::complex<double>* output, int n) {
  const int n2 = n / 2;
  output[0] = packed[0];
  output[n2 * n] =  packed[n2 * n];

  output[n2] =  packed[n2];
  output[n2 * n + n2] =  packed[n2 * n + n2];

  for (int c = 1; c < n2; ++c) {
    output[0 * n + c] = std::complex<double>(packed[c],
                                            packed[c + n2]);
    output[n2 * n + c] = std::complex<double>(packed[n2 * n + c],
                                            packed[n2 * n + c + n2]);
  }
  for (int r = 1; r < n2; ++r) {
    output[r * n + 0] = std::complex<double>(
        packed[r * n], packed[(r + n2) * n]);
    output[r * n + n2] = std::complex<double>(
        packed[r * n + n2], packed[(r + n2) * n + n2]);

    for (int c = 1; c < std::min(n2, n2); ++c) {
      output[r * n + c] = std::complex<double>(
          packed[r * n + c] - packed[(r + n2) * n + c + n2],
          packed[(r + n2) * n + c] +
          packed[r * n + c + n2]);
    }
    for (int c = 4; c < n2; c += 4) {
      __m256d real1 = _mm256_load_pd(packed + r * n + c);
      __m256d real2 = _mm256_load_pd(packed + (r + n2) * n + c + n2);
      __m256d imag1 = _mm256_load_pd(packed + (r + n2) * n + c);
      __m256d imag2 = _mm256_load_pd(packed + r * n + c + n2);
      real1 = _mm256_permute4x64_pd(real1 - real2, _MM_SHUFFLE(3, 1, 2, 0));
      imag1 = _mm256_permute4x64_pd(imag1 + imag2, _MM_SHUFFLE(3, 1, 2, 0));
      _mm256_storeu_pd((double*)(output + r * n + c),
                      _mm256_unpacklo_pd(real1, imag1));
      _mm256_storeu_pd((double*)(output + r * n + c + 2),
                      _mm256_unpackhi_pd(real1, imag1));
    }
    int r2 = r + n2;
    int r3 = n - r2;
    output[r2 * n + 0] = std::complex<double>(
        packed[r3 * n], -packed[(r3 + n2) * n]);
    output[r2 * n + n2] = std::complex<double>(
        packed[r3 * n + n2], -packed[(r3 + n2) * n + n2]);
    for (int c = 1; c < std::min(4, n2); ++c) {
      output[r2 * n + c] = std::complex<double>(
          packed[r3 * n + c] + packed[(r3 + n2) * n + c + n2],
          -packed[(r3 + n2) * n + c] +
          packed[r3 * n + c + n2]);
    }

    for (int c = 4; c < n2; c += 4) {
      __m256d real1 = _mm256_load_pd(packed + r3 * n + c);
      __m256d real2 = _mm256_load_pd(packed + (r3 + n2) * n + c + n2);
      __m256d imag1 = _mm256_load_pd(packed + (r3 + n2) * n + c);
      __m256d imag2 = _mm256_load_pd(packed + r3 * n + c + n2);
      real1 = _mm256_permute4x64_pd(real1 + real2, _MM_SHUFFLE(3, 1, 2, 0));
      imag1 = _mm256_permute4x64_pd(imag2 - imag1, _MM_SHUFFLE(3, 1, 2, 0));

      _mm256_storeu_pd((double*)(output + r2 * n + c),
                       _mm256_unpacklo_pd(real1, imag1));
      _mm256_storeu_pd((double*)(output + r2 * n + c + 2),
                       _mm256_unpackhi_pd(real1, imag1));
    }
  }
}

template <int vec_size=4, typename InputType=float>
void dft_2d_simd2(const InputType* input, std::complex<InputType>* output, int n,
                  typename dft_funs<InputType>::dft_1d_t tform) {
  static InputType* out_real = aligned_floats<InputType>(64 * 64 + 64);
  static InputType* out_real2 = aligned_floats<InputType>(64 * 64 + 64);

  for (int x = 0; x < n; x += vec_size) {
    tform(input + x, out_real + x, n);
  }

  transpose(out_real, out_real2, n);

  for (int x = 0; x < n; x += vec_size) {
    tform(out_real2 + x, out_real + x, n);
  }

  transpose(out_real, out_real2, n);

  unpack_2d_output(out_real2, output, n);
}

void dft_2d_simd(const float* input, std::complex<float>* output, int n,
                 typename dft_funs<float>::dft_1d_t tform) {
  static float* out_real_8x8 = (float*)aligned_floats<float>(64 * 64);
  static float* out_real2_8x8 = (float*)aligned_floats<float>(64 * 64);

  dft_8_simd(input, out_real_8x8, 8);

  transpose(out_real_8x8, out_real2_8x8, 8);

  dft_8_simd(out_real2_8x8, out_real_8x8, 8);

  transpose(out_real_8x8, out_real2_8x8, 8);
  //unpack_2d_output(out_real2_8x8, output, n);
  //return;
  output[0] = out_real2_8x8[0];
  output[4 * n] =  out_real2_8x8[4 * n];

  output[4] =  out_real2_8x8[4];
  output[4 * n + 4] =  out_real2_8x8[4 * n + 4];

  for (int c = 1; c < 4; ++c) {
    output[0 * n + c] = std::complex<float>(out_real2_8x8[c],
                                            out_real2_8x8[c + 4]);
    output[4 * n + c] = std::complex<float>(out_real2_8x8[4 * n + c],
                                            out_real2_8x8[4 * n + c + 4]);
  }
  for (int r = 1; r < 4; ++r) {
    output[r * n + 0] = std::complex<float>(
        out_real2_8x8[r * n], out_real2_8x8[(r + 4) * n]);
    output[r * n + 4] = std::complex<float>(
        out_real2_8x8[r * n + 4], out_real2_8x8[(r + 4) * n + 4]);

    for (int c = 1; c < 4; ++c) {
      output[r * n + c] = std::complex<float>(
          out_real2_8x8[r * n + c] - out_real2_8x8[(r + 4) * n + c + 4],
          out_real2_8x8[(r + 4) * n + c] +
          out_real2_8x8[r * n + c + 4]);
    }

    int r2 = r + 4;
    int r3 = n - r2;
    output[r2 * n + 0] = std::complex<float>(
        out_real2_8x8[r3 * n], -out_real2_8x8[(r3 + 4) * n]);
    output[r2 * n + 4] = std::complex<float>(
        out_real2_8x8[r3 * n + 4], -out_real2_8x8[(r3 + 4) * n + 4]);
    for (int c = 1; c < 4; ++c) {
      output[r2 * n + c] = std::complex<float>(
          out_real2_8x8[r3 * n + c] + out_real2_8x8[(r3 + 4) * n + c + 4],
          -out_real2_8x8[(r3 + 4) * n + c] +
          out_real2_8x8[r3 * n + c + 4]);
    }
  }
  return;
}

template <typename InputType=float>
void dft_2d(const InputType* input, std::complex<InputType>* output, int n,
            typename dft_funs<InputType>::dft_1d_t tform) {
  static InputType* row_fft = aligned_floats<InputType>(64 * 64);
  static InputType* col_fft = aligned_floats<InputType>(64 * 64);

  InputType* fft_ptr = row_fft;
  for (int y = 0; y < n; y++) {
    tform(input, fft_ptr, 1);
    fft_ptr += n;
    input += n;
  }
  transpose(row_fft, col_fft, n);

  for (int x = 0; x < n; ++x) {
    tform(col_fft + x * n, row_fft + x * n, 1);
  }
  transpose(row_fft, col_fft, n);
  unpack_2d_output(col_fft, output, n);
}

void benchmark() {
  const int n = 8;
  //std::vector<float> d(n*2);
  std::vector<std::complex<float> > act(n*2);
  //std::vector<float> out(n*2);
  float* d = (float*)aligned_alloc(n * 2 * sizeof(float), 32);
  float* out = (float*)aligned_alloc(n * 2 * sizeof(float), 32);

  double sum = 0;
  for (int i = 0; i < 1000000; ++i) {
    for (int j = 0; j < n; ++j) {
      d[j % n] = 1 + i + j;
      //dft_8_sse(d, out);
      dft_8_compact<float>(&d[0], &out[0]);
      //dft_32_compact<float>(&d[0], &out[0]);
      //dft(&d[0], &act[0], n);
      //d[j % n] = 0;
      sum += out[0] + out[1] + out[2] + out[3] +
          out[4] + out[5] + out[6] + out[7];// act[0].imag() + out[0];
    }
  }
  printf("sum: %f\n", sum);
}

template <typename InputType=float>
void validate_2d(int n,
                 typename dft_funs<InputType>::dft_2d_t fun2d,
                 typename dft_funs<InputType>::dft_1d_t func) {
  std::vector<double> d(n * n);
  std::vector<std::complex<InputType> > act(n * n);
  int num_different = 0;

  fftw_complex* fft = (fftw_complex*)malloc(sizeof(fftw_complex) * n * n);
  fftw_plan plan = fftw_plan_dft_r2c_2d(n, n, &d[0], fft, 0);
  InputType* f = aligned_floats<InputType>(n * n + 64);
  double total_diff = 0;

  memset(f, 0, sizeof(InputType) * n * n);
  memset(&d[0], 0, sizeof(InputType) * n * n);

  for (int di = 0; di < n * n; ++di) {
    f[di] = 1;
    d[di] = 1;

    fun2d(&f[0], &act[0], n, func);
    fftw_execute(plan);

    f[di] = 0;
    d[di] = 0;

    int w = n/2 + 1;
    const double kThreshold = 1e-4;
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j <= n/2; ++j) {
        double real_diff = std::abs(act[n * i + j].real() - fft[w * i + j][0]);
        double imag_diff = std::abs(act[n * i + j].imag() - fft[w * i + j][1]);
        if (real_diff > kThreshold || imag_diff > kThreshold) {
          printf("r[%d,%d] %f %f %s\n", i, j,
                 act[n * i + j].real(),
                 fft[w * i + j][0],
                 std::abs(act[n * i + j].real() - fft[w * i + j][0]) < kThreshold ? "ok" : "bad");
          printf("i[%d,%d] %f %f %s\n", i, j,
                 act[n * i + j].imag(),
                 fft[w * i + j][1],
                 std::abs(act[n * i + j].imag() - fft[w * i + j][1]) < kThreshold ? "ok" : "bad");
          printf("\n");
          num_different++;
        }
        total_diff += sqrt(real_diff*real_diff + imag_diff*imag_diff);
      }
    }
  }
  printf("2d %dx%d  %s (%lf)\n", n, n, num_different == 0 ? "OK" : "FAILED", total_diff / (n*n));

  free(fft);
  fftw_free(plan);
}

template <typename InputType=float>
void benchmark_2d(const int n,
                  typename dft_funs<InputType>::dft_2d_t fun2d,
                  typename dft_funs<InputType>::dft_1d_t func) {
  const int n2 = n * n;
  //std::vector<float> d(n * n);
  std::vector<std::complex<InputType> > act(n * n);

  std::vector<double> b(n * n); //  = (double*)malloc(n * n * sizeof(double));
  fftw_complex* fft = (fftw_complex*)malloc(sizeof(fftw_complex) * n * n);
  fftw_plan plan = fftw_plan_dft_r2c_2d(n, n, &b[0], fft, 0);

  std::vector<InputType> r(n * n);
  InputType* d = aligned_floats<InputType>((n * n + 16));
  /*
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      double v = 2.0 * ((double)rand() / RAND_MAX) - 1.0;
      printf("%f ", v);
      b[i * n + j] = v;
      d[i * n + j] = v;
    }
    printf("\n");
  }
  printf("\n\n");

  fftw_execute(plan);

  fun2d(d, &act[0], n, func);

  int w = n/2 + 1;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j <= n/2; ++j) {
      printf("r[%d,%d] %f %f %s\n", i, j,
             act[n * i + j].real(),
             fft[w * i + j][0],
             std::abs(act[n * i + j].real() - fft[w * i + j][0]) < 1e-4 ? "ok" : "bad");
      printf("i[%d,%d] %f %f %s\n", i, j,
             act[n * i + j].imag(),
             fft[w * i + j][1],
             std::abs(act[n * i + j].imag() - fft[w * i + j][1]) < 1e-4 ? "ok" : "bad");
    }
    printf("\n");
  }
  printf("\n\n");
  */

  const int num_trials = 10000000 / n;
  //fprintf(stderr, "\nBenchmarking %d\n", n);
  //fprintf(stderr, "running dft_2d %d\n", n);
  Timer timer;
  timer.start();
  double sum = 0;
  for (int i = 0; i < num_trials; ++i) {
    d[i % n2] += 1.0 / (i + 1);
    fun2d(d, &act[0], n, func);
    //dft_2d_simd(d, &act[0], n, dft);
    //dft_2d(d, &act[0], n, dft);
    //dft_8(&d[0], &act[0]);
    sum += act[1].real();
  }
  const double dft_time = timer.stop();

  //fprintf(stderr, "running fftw %d\n", n);
  timer.start();
  for (int i = 0; i < num_trials; ++i) {
    b[i % n2] += 1.0 / (i + 1);
    fftw_execute(plan);
    sum += fft[1][0];
  }
  const double fftw_time = timer.stop();
  fprintf(stderr, "%d dft: %f  fftw: %f  speedup: %f  (%d)\n",
          n, dft_time, fftw_time, dft_time / fftw_time, sum > 1000);
}

int main(int ac, char* av[]) {
  if (ac <= 1 || av[1][0] == 'b') {
    benchmark();
    return 0;
  }
  if (av[1][0] == 'v') {
    fprintf(stderr, "Testing 1d:\n");
    validate_1d(2, dft_2_compact<float>);
    validate_1d(4, dft_4_compact<float>);
    validate_1d(8, dft_8_compact<float>);
    validate_1d(16, dft_16_compact<float>);
    validate_1d(32, dft_32_compact<float>);
    validate_1d(64, dft_64_compact<float>);

    fprintf(stderr, "\nValidating 2d:\n");
    validate_2d(2, dft_2d<float>, dft_2_compact<float, float>);
    validate_2d(4, dft_2d<float>, dft_4_compact<float, float>);
    validate_2d(8, dft_2d<float>, dft_8_compact<float, float>);
    validate_2d(16, dft_2d<float>, dft_16_compact<float, float>);
    validate_2d(32, dft_2d<float>, dft_32_compact<float, float>);
    validate_2d(64, dft_2d<float>, dft_64_compact<float, float>);

    fprintf(stderr, "\nTesting simd2:\n");
    //validate_2d(2, dft_2d_simd2<4>, dft_2_compact<__m128>);
    validate_2d(4, dft_2d_simd2<4>, dft_4_compact<__m128>);
    validate_2d(8, dft_2d_simd2<4>, dft_8_compact<__m128, float>);
    validate_2d(16, dft_2d_simd2<4>, dft_16_compact<__m128, float>);
    validate_2d(32, dft_2d_simd2<4>, dft_32_compact<__m128, float>);
    validate_2d(64, dft_2d_simd2<4>, dft_64_compact<__m128, float>);

    fprintf(stderr, "\nTest simd (256)\n");
    validate_2d(8, dft_2d_simd, dft_8_compact<__m256, float>);

    fprintf(stderr, "\nTest simd2 (256)\n");
    validate_2d(8, dft_2d_simd2<8>, dft_8_compact<__m256, float>);
    validate_2d(16, dft_2d_simd2<8>, dft_16_compact<__m256, float>);
    validate_2d(32, dft_2d_simd2<8>, dft_32_compact<__m256, float>);
    validate_2d(64, dft_2d_simd2<8>, dft_64_compact<__m256>);

    fprintf(stderr, "Validating double\n");
    validate_1d(2, dft_2_compact<double, double>);
    validate_1d(4, dft_4_compact<double, double>);
    validate_1d(8, dft_8_compact<double, double>);
    validate_1d(16, dft_16_compact<double, double>);
    validate_1d(32, dft_32_compact<double, double>);
    validate_1d(64, dft_64_compact<double, double>);

    fprintf(stderr, "\nTesting 2d (double):\n");
    validate_2d<double>(2, dft_2d<double>, dft_2_compact<double, double>);
    validate_2d<double>(4, dft_2d<double>, dft_4_compact<double, double>);
    validate_2d<double>(8, dft_2d<double>, dft_8_compact<double, double>);
    validate_2d<double>(16, dft_2d<double>, dft_16_compact<double, double>);
    validate_2d<double>(32, dft_2d<double>, dft_32_compact<double, double>);
    validate_2d<double>(64, dft_2d<double>, dft_64_compact<double, double>);

    fprintf(stderr, "\nTesting simd2(128)[double]\n");
    //validate_2d<double>(2, dft_2d_simd2<2, double>, dft_2_compact<__m128d, double>);
    //validate_2d<double>(4, dft_2d_simd2<2, double>, dft_4_compact<__m128d, double>);
    validate_2d<double>(8, dft_2d_simd2<2, double>, dft_8_compact<__m128d, double>);
    validate_2d<double>(16, dft_2d_simd2<2, double>, dft_16_compact<__m128d, double>);
    validate_2d<double>(32, dft_2d_simd2<2, double>, dft_32_compact<__m128d, double>);
    validate_2d<double>(64, dft_2d_simd2<2, double>, dft_64_compact<__m128d, double>);

    fprintf(stderr, "\nTesting simd2(256)[double]\n");
    //validate_2d<double>(4, dft_2d_simd2<4, double>, dft_8_compact<__m256d, double>);
    validate_2d<double>(8, dft_2d_simd2<4, double>, dft_8_compact<__m256d, double>);
    validate_2d<double>(16, dft_2d_simd2<4, double>, dft_16_compact<__m256d, double>);
    validate_2d<double>(32, dft_2d_simd2<4, double>, dft_32_compact<__m256d, double>);
    validate_2d<double>(64, dft_2d_simd2<4, double>, dft_64_compact<__m256d, double>);
  }
  if (av[1][0] == '2') {
    fprintf(stderr, "\nTesting simd(256)\n");
    benchmark_2d(8, dft_2d_simd, dft_8_compact<__m256>);

    fprintf(stderr, "\nTesting simd2(256)[float]\n");
    benchmark_2d(8, dft_2d_simd2<8>, dft_8_compact<__m256, float>);
    benchmark_2d(16, dft_2d_simd2<8>, dft_16_compact<__m256, float>);
    benchmark_2d(32, dft_2d_simd2<8>, dft_32_compact<__m256, float>);
    benchmark_2d(64, dft_2d_simd2<8>, dft_64_compact<__m256, float>);

    fprintf(stderr, "\nTesting simd2(256)[double]\n");
    benchmark_2d<double>(8, dft_2d_simd2<4, double>, dft_8_compact<__m256d, double>);
    benchmark_2d<double>(16, dft_2d_simd2<4, double>, dft_16_compact<__m256d, double>);
    benchmark_2d<double>(32, dft_2d_simd2<4, double>, dft_32_compact<__m256d, double>);
    benchmark_2d<double>(64, dft_2d_simd2<4, double>, dft_64_compact<__m256d, double>);

    fprintf(stderr, "\nTesting simd2(128)\n");
    benchmark_2d(4, dft_2d_simd2<4>, dft_4_compact<__m128>);
    benchmark_2d(8, dft_2d_simd2<4>, dft_8_compact<__m128>);
    benchmark_2d(16, dft_2d_simd2<4>, dft_16_compact<__m128>);
    benchmark_2d(32, dft_2d_simd2<4>, dft_32_compact<__m128>);
    benchmark_2d(64, dft_2d_simd2<4>, dft_64_compact<__m128>);

    fprintf(stderr, "\nTesting simd2(128)[double]\n");
    benchmark_2d<double>(8, dft_2d_simd2<2, double>, dft_8_compact<__m128d, double>);
    benchmark_2d<double>(16, dft_2d_simd2<2, double>, dft_16_compact<__m128d, double>);
    benchmark_2d<double>(32, dft_2d_simd2<2, double>, dft_32_compact<__m128d, double>);
    benchmark_2d<double>(64, dft_2d_simd2<2, double>, dft_64_compact<__m128d, double>);


    fprintf(stderr, "\nTesting dft\n");
    benchmark_2d(4, dft_2d<float>, dft_4_compact<float,float>);
    benchmark_2d(8, dft_2d<float>, dft_8_compact<float,float>);
    benchmark_2d(16, dft_2d<float>, dft_16_compact<float,float>);
    benchmark_2d(32, dft_2d<float>, dft_32_compact<float,float>);
    benchmark_2d(64, dft_2d<float>, dft_64_compact<float,float>);

    fprintf(stderr, "\nTesting dft(double)\n");
    benchmark_2d<double>(4, dft_2d<double>, dft_4_compact<double,double>);
    benchmark_2d<double>(8, dft_2d<double>, dft_8_compact<double,double>);
    benchmark_2d<double>(16, dft_2d<double>, dft_16_compact<double,double>);
    benchmark_2d<double>(32, dft_2d<double>, dft_32_compact<double,double>);
    benchmark_2d<double>(64, dft_2d<double>, dft_64_compact<double,double>);
  }
  return 0;
}
