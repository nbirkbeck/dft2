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
    double elapsed = stop_ - start_;
    fprintf(stderr, "elapsed time: %lfs\n", elapsed);
    return elapsed;
  }
  double start_;
  double stop_;
};

struct Expression {
  int num = 0;
  int out_var = 0;
  bool is_real = 0;
  int vars[2];
  std::complex<float> weights[2];
  bool conj[2];
  int weight_ind[2][2];
};

std::vector<std::unique_ptr<Expression> > expressions;

void dft_builder(Expression** input,
                 Expression** output, int n,
                 std::vector<std::unique_ptr<Expression> >& exprs) {
  if (n == 1) {
    output[0] = input[0];
    return;
  }
  std::vector<Expression*> temp(n);
  std::vector<Expression*> out1(n / 2);
  std::vector<Expression*> out2(n / 2);
  for (int k = 0; k < n/2; ++k) {
    temp[k] = input[2 * k];
    temp[n/2 + k] = input[2 * k + 1];
  }
  dft_builder(&temp[0], &out1[0], n / 2, exprs);
  dft_builder(&temp[0] + n / 2, &out2[0], n / 2, exprs);

  for (int k = 0; k < n / 2; ++k) {
    std::complex<float> w = std::complex<float>(
        cos(2. * M_PI * k / n), -sin(2. * M_PI * k / n));
    int k1 = k;
    bool conj_v1 = false;
    bool conj_v2 = false;

    // Values in output > n / 4 don't exist; use conjugate symmetry
    if (k1 > n / 4) {
      conj_v1 = true;
      conj_v2 = true;
      k1 = n / 2 - k1;
    }
    //fprintf(stderr, "%d/%d %d/%d %d\n", k1, conj_v1, k2, conj_v2, n / 2);
    std::unique_ptr<Expression> new_expr(new Expression);
    const int v1 = out1[k1]->out_var;
    const int v2 = out2[k1]->out_var;

    new_expr->num = 2;
    new_expr->is_real = out1[k1]->is_real && out2[k1]->is_real && (w.imag() == 0);
    new_expr->vars[0] = v1;
    new_expr->vars[1] = v2;
    new_expr->weights[0] = 1;
    new_expr->weights[1] = w;
    new_expr->conj[0] = conj_v1;
    new_expr->conj[1] = conj_v2;
    new_expr->out_var = exprs.size();
    output[k] = new_expr.get();
    exprs.push_back(std::move(new_expr));

    if (k1 == 0) { //  <= n / 2) {
      // Add the one extra output
      new_expr.reset(new Expression);
      new_expr->num = 2;
      new_expr->is_real = out1[k1]->is_real && out2[k1]->is_real && (w.imag() == 0);
      new_expr->vars[0] = v1;
      new_expr->vars[1] = v2;
      new_expr->weights[0] = 1;
      new_expr->weights[1] = -w;
      new_expr->conj[0] = conj_v1;
      new_expr->conj[1] = conj_v2;
      new_expr->out_var = exprs.size();
      output[k1 + n / 2] = new_expr.get();
      exprs.push_back(std::move(new_expr));
    }
  }
  for (int k = n / 2 + 1; k < n; ++k) {
    output[k] = 0;
  }
}

void run_dft_builder(int n) {
  std::vector<std::unique_ptr<Expression> > exprs(n);
  std::vector<Expression*> input(n);
  std::vector<Expression*> output(n);
  for (int i = 0; i < n; ++i) {
    exprs[i] = std::unique_ptr<Expression>(new Expression);
    exprs[i]->out_var = i;
    exprs[i]->num = 1;
    exprs[i]->is_real = 1;
    exprs[i]->vars[0] = i;
    exprs[i]->vars[1] = -1;
    exprs[i]->weights[0] = 1;
    exprs[i]->weights[1] = 0;
    input[i] = exprs[i].get();
  }
  dft_builder(&input[0], &output[0], n, exprs);
  fprintf(stderr, "// Num expressions: %d\n", (int)exprs.size());
  std::vector<float> weights;
  weights.push_back(0);
  weights.push_back(1);

  for (int k = 0; k < exprs.size(); ++k) {
    for (int h = 0; h < 2; ++h) {
      for (int i = 0; i < 2; ++i) {
        int found = -1;
        float w = i == 0 ? exprs[k]->weights[h].real() :
            exprs[k]->weights[h].imag();
        for (int j = 0; j < weights.size(); ++j) {
          if (std::abs(weights[j] - std::abs(w)) < 1e-7) {
            found = j;
            break;
          }
        }
        if (found < 0) {
          found = weights.size();
          weights.push_back(std::abs(w));
        }
        exprs[k]->weight_ind[h][i] = w < 0 ? -found : found;
      }
    }
  }
  fprintf(stderr, "// Weights: %d\n", (int)weights.size());
  std::unordered_map<int, int> output_vars;
  for (int h = 0; h < output.size() / 2 + 1; ++h) {
    output_vars[output[h]->out_var] = h;
  }

  fprintf(stderr, "void dft_%d(float* input, std::complex<float>* output) {\n", n);
  for (int k = 2; k < weights.size(); ++k) {
    fprintf(stderr, "  const float kWeight%d = %g;\n", k, weights[k]);
  }
  for (int h = 0; h < exprs.size(); ++h) {
    const Expression& expr = *exprs[h];
    if (expr.num == 1) {
    } else {
      if (output_vars.find(h) == output_vars.end()) {
        if (expr.is_real) {
          fprintf(stderr, "  const float w%d = ",
                  expr.out_var - n);
        } else {
          fprintf(stderr, "  const std::complex<float> w%d = ",
                  expr.out_var - n);
        }
      } else {
        fprintf(stderr, "  output[%d] = ", output_vars[h]);
      }
      std::vector<std::string> ab(2);
      std::vector<std::pair<std::string, std::string> > parts(2);
      std::vector<bool> is_real(2);
      for (int k = 0; k < 2; ++k) {
        int v = expr.vars[k];
        char var_name[64];
        is_real[k] = exprs[v]->is_real;

        if (exprs[v]->num == 1) {
          snprintf(var_name, sizeof(var_name), "input[%d]", exprs[v]->out_var);
        } else if (output_vars.find(exprs[v]->out_var) != output_vars.end()) {
          snprintf(var_name, sizeof(var_name), "output[%d]",
                   output_vars[exprs[v]->out_var]);
        } else {
          snprintf(var_name, sizeof(var_name), "w%d", exprs[v]->out_var - n);
        }
        char mod_var[64];
        if (expr.conj[k] && !is_real[k]) {
          snprintf(mod_var, sizeof(mod_var), "std::conj(%s)", var_name);
        } else {
          snprintf(mod_var, sizeof(mod_var), "%s", var_name);
        }
        char result[64];
        if (expr.weight_ind[k][0] == 1 &&
            expr.weight_ind[k][1] == 0) {
          snprintf(result, sizeof(result), "%s ", mod_var);
          parts[k] = std::make_pair(std::string(result), "");
        } else if (expr.weight_ind[k][0] == -1 &&
                   expr.weight_ind[k][1] == 0) {
          snprintf(result, sizeof(result), "-%s ", mod_var);
          parts[k] = std::make_pair(std::string(result), "");
        } else {
          char real_weight[64] = {0};
          char imag_weight[64] = {0};
          if (expr.weight_ind[k][0] == -1) {
            snprintf(real_weight, sizeof(real_weight), "-");
          } else if (expr.weight_ind[k][0] != 1) {
            snprintf(real_weight, sizeof(real_weight), "%ckWeight%d",
                     expr.weight_ind[k][0] < 0 ? '-' : ' ',
                     std::abs(expr.weight_ind[k][0]));
          }
          if (expr.weight_ind[k][1] == -1) {
            snprintf(imag_weight, sizeof(imag_weight), "-");
          } else if (expr.weight_ind[k][1] != 1) {
            snprintf(imag_weight, sizeof(imag_weight), "%ckWeight%d",
                     expr.weight_ind[k][1] < 0 ? '-' : ' ',
                     std::abs(expr.weight_ind[k][1]));
          }
          snprintf(result, sizeof(result),
                  " std::complex<float>(%s, %s) * %s",
                   real_weight, imag_weight, mod_var);
          // This only makes sense for the case when variable is real
          parts[k] = std::make_pair(
              expr.weight_ind[k][0] == 0 ? "" : std::string(real_weight) + " * " + mod_var,
              strlen(imag_weight) <= 2 ? std::string(imag_weight) + mod_var :
              std::string(imag_weight) + " * " + mod_var);
        }
        ab[k] = result;
      }
      if (!expr.is_real && is_real[0] && is_real[1]) {
        fprintf(stderr, " {%s %s %s, %s %s %s};\n",
                parts[0].first.c_str(),
                parts[0].first.empty() || parts[1].first.empty() ? "" : "+",
                parts[1].first.c_str(),
                parts[0].second.c_str(),
                parts[0].second.empty() || parts[1].second.empty() ? "" : "+",
                parts[1].second.c_str());
      } else {
        fprintf(stderr, "%s + %s;\n", ab[0].c_str(), ab[1].c_str());
      }
    }
  }
  fprintf(stderr, "}\n");
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

void validate(int n, void (*func)(float*, std::complex<float>*)) {
  std::vector<float> d(n);
  std::vector<std::complex<float> > act(n);
  int num_different = 0;

  for (int i = 0; i < n; ++i) {
    d[i] = 1;
    std::vector<std::complex<float> > exp = dft(d);
    func(&d[0], &act[0]);
    for (int k = 0; k <= n / 2; ++k) {
      printf("%f+%fi ", act[k].real(), act[k].imag());
      if (std::abs(act[k].imag() - exp[k].imag()) > 1e-5 ||
          std::abs(act[k].real() - exp[k].real()) > 1e-5) {
        num_different++;
        printf("%f+%fi  %f+%fi on %d,%d\n",
               exp[k].real(), exp[k].imag(),
               act[k].real(), act[k].imag(), i, k);
      }

      //printf("%f+%fi ", res[k].real(), res[k].imag());
    }
    printf("\n");
    d[i] = 0;
  }
  printf("%d  %s\n", n, num_different == 0 ? "OK" : "FAILED");
}

inline void transpose4x4_SSE(float *A, float *B, const int lda, const int ldb) {
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

inline void transpose8x8_SSE(float *A, float *B) {
  transpose4x4_SSE(A, B, 8, 8);
  transpose4x4_SSE(A + 4 * 8, B + 4, 8, 8);
  transpose4x4_SSE(A + 4, B + 4 * 8, 8, 8);
  transpose4x4_SSE(A + 4 * 8 + 4, B + 4 * 8 + 4, 8, 8);
}


void dft_2d_simd(float* input, std::complex<float>* output, int n,
            void (*tform)(float*, std::complex<float>*)) {
  static float* out_real_8x8 = (float*)aligned_alloc(8 * 8 * sizeof(float), 64);
  static float* out_real2_8x8 = (float*)aligned_alloc(8 * 8 * sizeof(float), 64);

  dft_8_simd(input, out_real_8x8);
  /*
  for (int r = n / 2 + 1; r < n; ++r) {
    memcpy(out_real_8x8 + r * n,
           out_real_8x8 + (n - r) * n,
           sizeof(float) * n);

    for (int x = 0; x < n; ++x) {
      out_imag_8x8[r * n + x] = -out_imag_8x8[(n - r) * n + n - x];
    }
    }
    /*
  for (int y = 0; y < n; ++y) {
    for (int x = 0; x < n; ++x) {
      printf("%f ", out_imag_8x8[y * n + x]);
    }
    printf("\n");
    }*/
  //  exit(1);
  transpose8x8_SSE(out_real_8x8, out_real2_8x8);

  dft_8_simd(out_real2_8x8, out_real_8x8);

  transpose8x8_SSE(out_real_8x8, out_real2_8x8);


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
  /*
  for (int y = 0; y < n; ++y) {
    for (int x = 0; x < n; ++x) {
      printf("%f ", output[y * n + x].real());
    }
    printf("\n");
  }
  exit(1);
  */
  return;

}

void dft_2d(float* input, std::complex<float>* output, int n,
            void (*tform)(float*, std::complex<float>*)) {
  static std::complex<float> row_fft[64 * 64];
  std::complex<float>* row_fft_ptr = row_fft;
  for (int y = 0; y < n; y++) {
    tform(input, row_fft_ptr);
    for (int x = n / 2 + 1; x < n; ++x) {
      row_fft_ptr[x] = std::conj(row_fft_ptr[n - x]);
    }
    row_fft_ptr += n;
    input += n;
  }

  float real[n], imag[n];
  std::complex<float> real_fft[n], imag_fft[n];
  for (int x = 0; x < n; ++x) {
    bool all_imag_zero = true;
    bool all_real_zero = true;
    for (int y = 0; y < n; ++y) {
      real[y] = row_fft[y * n + x].real();
      imag[y] = row_fft[y * n + x].imag();
      all_imag_zero &= (imag[y] == 0);
      all_real_zero &= (real[y] == 0);
    }
    if (all_real_zero) {
      memset(real_fft, 0, sizeof(real_fft));
    } else {
      tform(real, real_fft);
    }
    if (all_imag_zero) {
      memset(imag_fft, 0, sizeof(imag_fft));
    } else {
      tform(imag, imag_fft);
    }
    for (int y = 0; y <= n/2; ++y) {
      output[y * n + x] = real_fft[y] +
          std::complex<float>(-imag_fft[y].imag(), imag_fft[y].real());
    }
    for (int y = n/2+1; y < n; ++y) {
      output[y * n + x] =
          std::conj(real_fft[n - y]) +
          std::complex<float>(imag_fft[n - y].imag(),
                              imag_fft[n - y].real());
    }
  }
}

void dft32_2d(float* input, std::complex<float>* output) {
  int w = 32;
  std::complex<float> row_fft[32 * 32];
  for (int y = 0; y < 32; ++y) {
    dft_32(input + y * 32, row_fft + y * 32);
    for (int x = 17; x < 32; ++x) {
      row_fft[y * 32 + x] = std::conj(row_fft[y * 32 + 32 - x]);
    }
  }
  float real[32], imag[32];
  std::complex<float> real_fft[32], imag_fft[32];
  for (int x = 0; x <= 16; ++x) {
    for (int y = 0; y < 32; ++y) {
      real[y] = row_fft[y * 32 + x].real();
      imag[y] = row_fft[y * 32 + x].imag();
    }
    dft_32(real, real_fft);
    dft_32(imag, imag_fft);
    for (int y = 0; y <= 16; ++y) {
      output[y * w + x] = real[y] + std::complex<float>(0, 1) * imag[y];
    }
    for (int y = 17; y < 32; ++y) {
      output[y * w + x] = std::conj(output[(32 - y) * w + x]);
    }
  }
  // Fill in remaining columns using conjugate symmetry
  for (int x = 17; x < 32; ++x) {
    for (int y = 0; y < 32; ++y) {
      output[y * w + x] = output[y * w + (32 - x)];
    }
  }
}

struct MyType {
  float a, b, c, d;

  MyType operator+(const MyType& m2) const {
    const MyType& m1 = *this;
    return {m1.a + m2.a, m1.b + m2.b, m1.c + m2.c, m1.d + m2.d};
  }
  MyType operator-(const MyType& m2) const {
    const MyType& m1 = *this;
    return {m1.a - m2.a, m1.b - m2.b, m1.c - m2.c, m1.d - m2.d};
  }
  MyType operator*(const MyType& m2) const {
    const MyType& m1 = *this;
    return {m1.a * m2.a, m1.b * m2.b, m1.c * m2.c, m1.d * m2.d};
  }
  const MyType& operator*=(const MyType& m2) const {
    return *this;
  }
};

typedef void (*dft_fun_t)(float*, std::complex<float>*);

dft_fun_t get_dft(int n) {
  if (n == 2)
    return dft_2;
  if (n == 4)
    return dft_4;
  if (n == 8)
    return dft_8;
  if (n == 16)
    return dft_16;
  if (n == 32)
    return dft_32;
}

int main(int ac, char* av[]) {
  if (ac == 1) {
    for (int x = 2; x <= 32; x *= 2) {
      run_dft_builder(x);
    }
    return 0;
  }
  if (av[1][0] == 'v') {
    validate(2, dft_2);
    validate(4, dft_4);
    validate(8, dft_8);
    validate(16, dft_16);
    validate(32, dft_32);
  }
  if (av[1][0] == 'b') {
    std::vector<float> d(32);
    std::vector<std::complex<float> > act(32);
    double sum = 0;
    for (int i = 0; i < 100000; ++i) {
      for (int j = 0; j < 32; ++j) {
        d[j % 32] = 1;
        dft_32(&d[0], &act[0]);
        //dft(&d[0], &act[0], 32);
        d[j % 32] = 0;
      }
      sum += act[0].imag();
    }
  }
  if (av[1][0] == '2') {
    const int n = 8;
    int n2 = n  * n;
    //std::vector<float> d(n * n);
    float* d = (float*)aligned_alloc(n * n * sizeof(float), 32);
    std::vector<std::complex<float> > act(n * n);

    dft_fun_t dft = get_dft(n);

    std::vector<double> b(n * n);
    fftw_complex* fft = (fftw_complex*)malloc(sizeof(fftw_complex) * n * n);
    fftw_plan plan = fftw_plan_dft_r2c_2d(n, n, &b[0], fft, 0);

    std::vector<float> r(n * n);

    float data[8][8];
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        data[i][j] = 2.0 * ((double)rand() / RAND_MAX) - 1.0;
        printf("%f ", data[i][j]);
        b[i * n + j] = data[i][j];
        d[i * n + j] = data[i][j];
      }
      printf("\n");
    }
    printf("\n\n");
    dft_2d(d, &act[0], n, dft);
    fftw_execute(plan);

    int w = n/2 + 1;
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j <= n/2; ++j) {
        printf("r[%d,%d] %f %f %s\n", i, j,
               act[n * i + j].real(),
               fft[w * i + j][0],
               std::abs(act[n * i + j].real() - fft[w * i + j][0]) < 1e-5 ? "ok" : "bad");
        printf("i[%d,%d] %f %f %s\n", i, j,
               act[n * i + j].imag(),
               fft[w * i + j][1],
               std::abs(act[n * i + j].imag() - fft[w * i + j][1]) < 1e-5 ? "ok" : "bad");
      }
      printf("\n");
    }
    printf("\n\n");
    //return 0;

    const int num_trials = 1000000;
    fprintf(stderr, "running dft_2d\n");
    Timer timer;
    timer.start();
    for (int i = 0; i < num_trials; ++i) {
      d[i % n2] += 1.0 / (i + 1);
      dft_2d(d, &act[0], n, dft);
      //dft_8(&d[0], &act[0]);
    }
    double dft_time = timer.stop();

    fprintf(stderr, "running fftw\n");
    timer.start();
    for (int i = 0; i < num_trials; ++i) {
      b[i % n2] += 1.0 / (i + 1);
      fftw_execute(plan);
    }
    double fftw_time = timer.stop();
    fprintf(stderr, "Speedup: %f\n", dft_time / fftw_time);
  }
  return 0;
}
