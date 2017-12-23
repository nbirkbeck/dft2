#include <complex>
#include <vector>
#include <math.h>
#include <stdio.h>
#include <memory>
#include <map>
#include <unordered_map>
#include "gen.h"

struct Expression {
  int num = 0;
  int out_var = 0;
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
  for (int k = 0; k < weights.size(); ++k) {
    fprintf(stderr, "  const float kWeight%d = %g;\n", k, weights[k]);
  }
  for (int h = 0; h < exprs.size(); ++h) {
    const Expression& expr = *exprs[h];
    if (expr.num == 1) {
    } else {
      if (output_vars.find(h) == output_vars.end()) {
        fprintf(stderr, "  const std::complex<float> w%d = ",
                expr.out_var - n);
      } else {
        fprintf(stderr, "  output[%d] = ", output_vars[h]);
      }
      for (int k = 0; k < 2; ++k) {
        int v = expr.vars[k];
        char var_name[64];
        if (exprs[v]->num == 1) {
          snprintf(var_name, sizeof(var_name), "input[%d]", exprs[v]->out_var);
        } else if (output_vars.find(exprs[v]->out_var) != output_vars.end()) {
          snprintf(var_name, sizeof(var_name), "output[%d]",
                   output_vars[exprs[v]->out_var]);
        } else {
          snprintf(var_name, sizeof(var_name), "w%d", exprs[v]->out_var - n);
        }
        /*
          if (expr.weight_ind[k][0] == 1 &&
              expr.weight_ind[k][1] == 0) {
            fprintf(stderr, "%s input[%d] %s",
                    expr.conj[k] ? "std::conj(" : "",
                    exprs[v]->out_var,
                    expr.conj[k] ? ")" : "");
          } else if (expr.weight_ind[k][0] == -1 &&
                     expr.weight_ind[k][1] == 0) {
            fprintf(stderr, "%s -input[%d] %s",
                    expr.conj[k] ? "std::conj(" : "",
                    exprs[v]->out_var,
                    expr.conj[k] ? ")" : "");
          } else {
            fprintf(stderr,
                    " std::complex<float>(%ckWeight%d, %ckWeight%d) * input[%d]",
                    expr.weight_ind[k][0] < 0 ? '-' : ' ',
                    std::abs(expr.weight_ind[k][0]),
                    expr.weight_ind[k][1] < 0 ? '-' : ' ',
                    std::abs(expr.weight_ind[k][1]),
                    exprs[v]->out_var);
          }
          } */
        char mod_var[64];
        if (expr.conj[k]) {
          snprintf(mod_var, sizeof(mod_var), "std::conj(%s)", var_name);
        } else {
          snprintf(mod_var, sizeof(mod_var), "%s", var_name);
        }
        if (expr.weight_ind[k][0] == 1 &&
            expr.weight_ind[k][1] == 0) {
          fprintf(stderr, "%s ", mod_var);
        } else if (expr.weight_ind[k][0] == -1 &&
                   expr.weight_ind[k][1] == 0) {
          fprintf(stderr, "-%s ", mod_var);
        } else {
          fprintf(stderr,
                  " std::complex<float>(%ckWeight%d, %ckWeight%d) * %s",
                  expr.weight_ind[k][0] < 0 ? '-' : ' ',
                  std::abs(expr.weight_ind[k][0]),
                  expr.weight_ind[k][1] < 0 ? '-' : ' ',
                  std::abs(expr.weight_ind[k][1]),
                  mod_var);
        }
        if (k == 0) fprintf(stderr, " + ");
      }
      fprintf(stderr, ";\n");
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
      if (std::abs(act[k] - exp[k]) > 1e-5) {
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

int main(int ac, char* av[]) {
  if (ac == 1) {
    for (int x = 1; x <= 32; x *= 2) {
      run_dft_builder(x);
    }
    return 0;
  }
  if (av[1][0] == 'v') {
    validate(1, dft_1);
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
  return 0;
}
