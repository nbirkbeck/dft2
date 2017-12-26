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

template <typename T>
struct SimdHelper {
  static const T constant(const float& f);
  static const T load(const float* f);
  static void store(float* f, const T&);
};

template <>
const float SimdHelper<float>::constant(const float& f) {
  return f;
}

template <>
const float SimdHelper<float>::load(const float* f) {
  return *f;
}

template <>
void SimdHelper<float>::store(float* output, const float& f) {
  *output = f;
}

template <>
const __m256 SimdHelper<__m256>::constant(const float& f) {
  return _mm256_set1_ps(f);
}
template <>
const __m256 SimdHelper<__m256>::load(const float* f) {
  return _mm256_load_ps(f);
}
template <>
void SimdHelper<__m256>::store(float* output, const __m256& f) {
  _mm256_store_ps(output, f);
}

template <>
const __m128 SimdHelper<__m128>::constant(const float& f) {
  return _mm_set1_ps(f);
}

template <>
const __m128 SimdHelper<__m128>::load(const float* f) {
  return _mm_load_ps(f);
}
template <>
void SimdHelper<__m128>::store(float* output, const __m128& f) {
  _mm_store_ps(output, f);
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

typedef void (*dft_fun_t)(const float*, float*, int);


#include "gen.h"

struct Timer {
  void start() {
    start_ = ((double)clock()) / CLOCKS_PER_SEC;
  }
  double stop() {
    stop_ = ((double)clock()) / CLOCKS_PER_SEC;
    double elapsed = stop_ - start_;
    fprintf(stderr, "elapsed time: %lfs\n", elapsed);
    return elapsed;
  }
  double start_;
  double stop_;
};

float* aligned_floats(int n) {
  float* f = (float*)malloc(sizeof(float)*(n + 16));
  while (((uint64_t)f) % 128 != 0) {
    f++;
  }
  return f;
}

void dft(float* data, std::complex<float>* result, int n) {
  if (n == 1) {
    result[0] = data[0];
    return;
  }
  std::vector<float> temp(n);
  for (int k = 0; k < n/2; ++k) {
    temp[k] = data[2 * k];
    temp[n/2 + k] = data[2 * k + 1];
  }
  dft(&temp[0], result, n / 2);
  dft(&temp[0] + n / 2, result + n / 2, n / 2);
  for (int k = 0; k < n / 2; ++k) {
    std::complex<float> w = std::complex<float>(
        cos(2. * M_PI * k / n), -sin(2. * M_PI * k / n));
    std::complex<float> a = result[k];
    std::complex<float> b = result[n / 2 + k];
    result[k] = a + w * b;
    result[n / 2 + k] = a - w * b;
  }
}

std::vector<std::complex<float> > dft(std::vector<float> &data) {
  std::vector<std::complex<float> > result(data.size());
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

void validate_1d(int n, dft_fun_t func) {
  std::vector<float> d(n);
  std::vector<std::complex<float> > act(n);
  std::vector<float> compact(n);
  int num_different = 0;

  for (int i = 0; i < n; ++i) {
    d[i] = 1;
    std::vector<std::complex<float> > exp = dft(d);
    func(&d[0], &compact[0], 1);
    for (int k = 0; k <= n / 2; k++) {
      act[k] = std::complex<float>(
          compact[k],
          k > 0 && k < n / 2 ? compact[n/2 + k] : 0);
    }

    for (int k = 0; k <= n / 2; ++k) {
      //printf("%f+%fi ", act[k].real(), act[k].imag());
      if (std::abs(act[k].imag() - exp[k].imag()) > 1e-5 ||
          std::abs(act[k].real() - exp[k].real()) > 1e-5) {
        num_different++;
        printf("%f+%fi  %f+%fi on %d,%d\n",
               exp[k].real(), exp[k].imag(),
               act[k].real(), act[k].imag(), i, k);
      }
      //printf("%f+%fi ", res[k].real(), res[k].imag());
    }
    //printf("\n");
    d[i] = 0;
  }
  printf("%d  %s\n", n, num_different == 0 ? "OK" : "FAILED");
}

inline void transpose4x4_SSE(const float *A, float *B, const int lda, const int ldb) {
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

inline void transpose8x8_SSE(const float *A, float *B) {
  transpose4x4_SSE(A, B, 8, 8);
  transpose4x4_SSE(A + 4 * 8, B + 4, 8, 8);
  transpose4x4_SSE(A + 4, B + 4 * 8, 8, 8);
  transpose4x4_SSE(A + 4 * 8 + 4, B + 4 * 8 + 4, 8, 8);
}

void unpack_2d_output(const float* col_fft,
                      std::complex<float>* output, int n) {
  for (int y = 0; y <= n/2; ++y) {
    const int y2 = y + n/2;
    const bool y_extra = y2 > n/2 && y2 < n;

    for (int x = 0; x <= n/2; ++x) {
      const int x2 = x + n/2;
      const bool x_extra = x2 > n/2 && x2 < n;
      output[y * n + x] = std::complex<float>(
          col_fft[x * n + y] -
          (x_extra && y_extra ? col_fft[x2 * n + y2] : 0),
          (y_extra ? col_fft[x * n + y2] : 0) +
          (x_extra ? col_fft[x2 * n + y] : 0 ));

      if (y_extra) {
        output[(n - y) * n + x] = std::complex<float>(
            col_fft[x * n + y] +
            (x_extra && y_extra ? col_fft[x2 * n + y2] : 0),
            -(y_extra ? col_fft[x * n + y2] : 0) +
            (x_extra ? col_fft[x2 * n + y] : 0 ));
      }
    }
  }
}

template <int vec_size=4>
void dft_2d_simd2(const float* input, std::complex<float>* output, int n,
                  dft_fun_t tform) {
  static float* out_real = (float*)aligned_floats(32 * 32 + 10);
  static float* out_real2 = (float*)aligned_floats(32 * 32 + 10);

  for (int x = 0; x < n; x += vec_size) {
    tform(input + x, out_real + x, n);
  }

  for (int y = 0; y < n; y += 4) {
    for (int x = 0; x < n; x += 4) {
      transpose4x4_SSE(out_real + y * n + x,
                       out_real2 + x * n + y, n, n);
    }
  }

  for (int x = 0; x < n; x += vec_size) {
    tform(out_real2 + x, out_real + x, n);
  }

  for (int y = 0; y < n; y += 4) {
    for (int x = 0; x < n; x += 4) {
      transpose4x4_SSE(out_real + y * n + x,
                       out_real2 + x * n + y, n, n);
    }
  }
  unpack_2d_output(out_real, output, n);
}

void dft_2d_simd(const float* input, std::complex<float>* output, int n,
                 dft_fun_t tform) {
  static float* out_real_8x8 = (float*)aligned_floats(32 * 32);
  static float* out_real2_8x8 = (float*)aligned_floats(32 * 32);

  dft_8_simd(input, out_real_8x8, 8);

  transpose8x8_SSE(out_real_8x8, out_real2_8x8);

  dft_8_simd(out_real2_8x8, out_real_8x8, 8);

  transpose8x8_SSE(out_real_8x8, out_real2_8x8);

  //unpack_2d_output(out_real_8x8, output, n);
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

typedef void (*dft_2d_fun_t)(const float*, std::complex<float>*, int n, dft_fun_t);

void dft_2d(const float* input, std::complex<float>* output, int n,
            dft_fun_t tform) {
  static float row_fft[64 * 64];
  static float col_fft[64 * 64];

  float* fft_ptr = row_fft;
  for (int y = 0; y < n; y++) {
    tform(input, fft_ptr, 1);
    fft_ptr += n;
    input += n;
  }
  float col[n];
  for (int x = 0; x < n; ++x) {
    for (int y = 0; y < n; ++y) {
      col[y] = row_fft[y * n + x];
    }
    tform(col, col_fft + x * n, 1);
  }
  unpack_2d_output(col_fft, output, n);
  /*
  for (int y = 0; y <= n/2; ++y) {
    const int y2 = y + n/2;
    const bool y_extra = y2 > n/2 && y2 < n;

    for (int x = 0; x <= n/2; ++x) {
      const int x2 = x + n/2;
      const bool x_extra = x2 > n/2 && x2 < n;
      output[y * n + x] = std::complex<float>(
          col_fft[x * n + y] -
          (x_extra && y_extra ? col_fft[x2 * n + y2] : 0),
          (y_extra ? col_fft[x * n + y2] : 0) +
          (x_extra ? col_fft[x2 * n + y] : 0 ));

      if (y_extra) {
        output[(n - y) * n + x] = std::complex<float>(
            col_fft[x * n + y] +
            (x_extra && y_extra ? col_fft[x2 * n + y2] : 0),
            -(y_extra ? col_fft[x * n + y2] : 0) +
            (x_extra ? col_fft[x2 * n + y] : 0 ));
      }
    }
    }*/
}

dft_fun_t get_dft(int n) {
  if (n == 2)
    return dft_2_compact<float>;
  if (n == 4)
    return dft_4_compact<float>;
  if (n == 8)
    return dft_8_compact<float>;
  if (n == 16)
    return dft_16_compact<float>;
  if (n == 32)
    return dft_32_compact<float>;
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

void validate_2d(int n, dft_2d_fun_t fun2d, dft_fun_t func) {
  std::vector<double> d(n * n);
  std::vector<std::complex<float> > act(n * n);
  int num_different = 0;

  fftw_complex* fft = (fftw_complex*)malloc(sizeof(fftw_complex) * n * n);
  fftw_plan plan = fftw_plan_dft_r2c_2d(n, n, &d[0], fft, 0);

  //std::vector<float> f(n * n);
  float* f = (float*)malloc((n * n + 64) * sizeof(float));
  while (((uint64_t)f) % 128 != 0) {
    f++;
  }
  memset(f, 0, sizeof(float) * n * n);

  for (int di = 0; di < n * n; ++di) {
    f[di] = 1;
    d[di] = 1;

    fun2d(&f[0], &act[0], n, func);
    fftw_execute(plan);

    f[di] = 0;
    d[di] = 0;

    int w = n/2 + 1;
    const double kThreshold = 1e-5;
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
      }
    }
  }
  printf("2d %dx%d  %s\n", n, n, num_different == 0 ? "OK" : "FAILED");

  free(fft);
  fftw_free(plan);
}

void benchmark_2d(const int n, dft_2d_fun_t fun2d, dft_fun_t func) {
  const int n2 = n * n;
  //std::vector<float> d(n * n);
  std::vector<std::complex<float> > act(n * n);

  std::vector<double> b(n * n); //  = (double*)malloc(n * n * sizeof(double));
  fftw_complex* fft = (fftw_complex*)malloc(sizeof(fftw_complex) * n * n);
  fftw_plan plan = fftw_plan_dft_r2c_2d(n, n, &b[0], fft, 0);

  std::vector<float> r(n * n);
  float* d = (float*)malloc((n * n + 16) * sizeof(float));
  while (((uint64_t)d) % 64 != 0) {
    d++;
  }

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

  const int num_trials = 1000000;
  fprintf(stderr, "\nBenchmarking %d\n", n);
  fprintf(stderr, "running dft_2d %d\n", n);
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

  fprintf(stderr, "running fftw %d\n", n);
  timer.start();
  for (int i = 0; i < num_trials; ++i) {
    b[i % n2] += 1.0 / (i + 1);
    fftw_execute(plan);
    sum += fft[1][0];
  }
  const double fftw_time = timer.stop();
  fprintf(stderr, "Speedup: %f  (%d)\n", dft_time / fftw_time, sum > 1000);
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

    fprintf(stderr, "\nTesting 2d:\n");
    validate_2d(2, dft_2d, dft_2_compact<float>);
    validate_2d(4, dft_2d, dft_4_compact<float>);
    validate_2d(8, dft_2d, dft_8_compact<float>);
    validate_2d(16, dft_2d, dft_16_compact<float>);
    validate_2d(32, dft_2d, dft_32_compact<float>);

    fprintf(stderr, "\nTesting simd2:\n");
    //validate_2d(2, dft_2d_simd2<4>, dft_2_compact<__m128>);
    validate_2d(4, dft_2d_simd2<4>, dft_4_compact<__m128>);
    validate_2d(8, dft_2d_simd2<4>, dft_8_compact<__m128>);
    validate_2d(16, dft_2d_simd2<4>, dft_16_compact<__m128>);
    validate_2d(32, dft_2d_simd2<4>, dft_32_compact<__m128>);

    fprintf(stderr, "\nTest simd (256)\n");
    validate_2d(8, dft_2d_simd, dft_8_compact<__m256>);

    fprintf(stderr, "\nTest simd2 (256)\n");
    validate_2d(8, dft_2d_simd2<8>, dft_8_compact<__m256>);
    validate_2d(16, dft_2d_simd2<8>, dft_16_compact<__m256>);
    validate_2d(32, dft_2d_simd2<8>, dft_32_compact<__m256>);
  }
  if (av[1][0] == '2') {
    fprintf(stderr, "\nTesting simd(256)\n");
    benchmark_2d(8, dft_2d_simd, dft_8_compact<__m256>);

    fprintf(stderr, "\nTesting simd2(256)\n");
    benchmark_2d(8, dft_2d_simd2<8>, dft_8_compact<__m256>);
    benchmark_2d(16, dft_2d_simd2<8>, dft_16_compact<__m256>);
    benchmark_2d(32, dft_2d_simd2<8>, dft_32_compact<__m256>);

    benchmark_2d(4, dft_2d, dft_4_compact<float>);
    benchmark_2d(8, dft_2d, dft_8_compact<float>);
    benchmark_2d(16, dft_2d, dft_16_compact<float>);
    benchmark_2d(32, dft_2d, dft_32_compact<float>);

    benchmark_2d(4, dft_2d_simd2<4>, dft_4_compact<__m128>);
    benchmark_2d(8, dft_2d_simd2<4>, dft_8_compact<__m128>);
    benchmark_2d(16, dft_2d_simd2<4>, dft_16_compact<__m128>);
    benchmark_2d(32, dft_2d_simd2<4>, dft_32_compact<__m128>);

  }
  return 0;
}
