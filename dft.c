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
  for (int k = 0; k < n/2; ++k) {
    temp[k] = input[2 * k];
    temp[n/2 + k] = input[2 * k + 1];
  }
  dft_builder(&temp[0], output, n / 2, exprs);
  dft_builder(&temp[0] + n / 2, output + n / 2, n / 2, exprs);

  for (int k = 0; k < n / 2; ++k) {
    std::complex<float> w = std::complex<float>(
        cos(2. * M_PI * k / n), -sin(2. * M_PI * k / n));
    const int k2 = n / 2 + k;
    std::unique_ptr<Expression> new_expr(new Expression);
    const int v1 = output[k]->out_var;
    const int v2 = output[k2]->out_var;
    new_expr->num = 2;
    new_expr->vars[0] = v1;
    new_expr->vars[1] = v2;
    new_expr->weights[0] = 1;
    new_expr->weights[1] = w;
    new_expr->out_var = exprs.size();
    output[k] = new_expr.get();
    exprs.push_back(std::move(new_expr));

    new_expr.reset(new Expression);
    new_expr->num = 2;
    new_expr->vars[0] = v1;
    new_expr->vars[1] = v2;
    new_expr->weights[0] = 1;
    new_expr->weights[1] = -w;
    new_expr->out_var = exprs.size();
    output[k2] = new_expr.get();
    exprs.push_back(std::move(new_expr));
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
  fprintf(stderr, "Num expressions: %d\n", (int)exprs.size());
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
  fprintf(stderr, "Weights: %d\n", (int)weights.size());
  for (int k = 0; k < weights.size(); ++k) {
    fprintf(stderr, "const float kWeight%d = %e;\n", k, weights[k]);
  }
  std::unordered_map<int, int> output_vars;
  for (int h = 0; h < output.size(); ++h) {
    output_vars[output[h]->out_var] = h;
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
        if (exprs[v]->num == 1) {
          if (expr.weight_ind[k][0] == 1 &&
              expr.weight_ind[k][1] == 0) {
            fprintf(stderr, "input[%d] ", exprs[v]->out_var);
          } else if (expr.weight_ind[k][0] == -1 &&
                     expr.weight_ind[k][1] == 0) {
            fprintf(stderr, "-input[%d] ", exprs[v]->out_var);
          } else {
            fprintf(stderr,
                    " std::complex<float>(%ckWeight%d, %ckWeight%d) * input[%d]",
                    expr.weight_ind[k][0] < 0 ? '-' : ' ',
                    std::abs(expr.weight_ind[k][0]),
                    expr.weight_ind[k][1] < 0 ? '-' : ' ',
                    std::abs(expr.weight_ind[k][1]),
                    exprs[v]->out_var);
          }
        } else {
          if (expr.weight_ind[k][0] == 1 &&
              expr.weight_ind[k][1] == 0) {
            fprintf(stderr, "w%d ", exprs[v]->out_var - n);
          } else if (expr.weight_ind[k][0] == -1 &&
                     expr.weight_ind[k][1] == 0) {
            fprintf(stderr, "-w%d ", exprs[v]->out_var - n);
          } else {
            fprintf(stderr,
                    " std::complex<float>(%ckWeight%d, %ckWeight%d) * w%d",
                    expr.weight_ind[k][0] < 0 ? '-' : ' ',
                    std::abs(expr.weight_ind[k][0]),
                    expr.weight_ind[k][1] < 0 ? '-' : ' ',
                    std::abs(expr.weight_ind[k][1]),
                    exprs[v]->out_var - n);
          }
        }
        if (k == 0) fprintf(stderr, " + ");
      }
      fprintf(stderr, ";\n");
    }
  }
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

int main(int ac, char* av[]) {
  //run_dft_builder(16);

  std::vector<float> d(16);
  double r = 0;
  for (int k = 0; k < 100000; ++k) {
  for (int i = 0; i < d.size(); ++i) {
    d[i] = k;
    std::vector<std::complex<float> > res = dft(d); // &d[0]);
    for (int k = 0; k < res.size(); ++k) {
      //printf("%f+%fi ", res[k].real(), res[k].imag());
    }
    //printf("\n");
    d[i] = 0;
    r += res[0].real();
  }
  }
  return 0;
}
