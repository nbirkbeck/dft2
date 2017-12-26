#include <math.h>
#include <stdlib.h>
#include <complex.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <map>
#include <memory>
#include <string.h>


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

std::string sign_string(double w) {
  return w < 0 ? "-" : "";
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
  std::unordered_map<int, int> output_vars;
  for (int h = 0; h < output.size() / 2 + 1; ++h) {
    output_vars[output[h]->out_var] = h;
  }

  struct OutputVar {
    std::string output_var_name;
    std::string real_part;
    std::string imag_part;
    int output_index;
  };
  std::vector<OutputVar> declarations;

  for (int h = 0; h < exprs.size(); ++h) {
    const Expression& expr = *exprs[h];
    if (expr.num == 1 && expr.out_var < n) {
    } else {
      char output_var_name[64];
      if (output_vars.find(h) == output_vars.end()) {
        if (expr.is_real) {
          snprintf(output_var_name, 64, "  const T w%d ",
                   expr.out_var - n);
        } else {
          snprintf(output_var_name, 64, "  const T w%d[2] ",
                   expr.out_var - n);
        }
      }
      std::vector<std::pair<std::string, std::string> > parts(2);
      std::vector<bool> is_real(2);
      for (int k = 0; k < 2; ++k) {
        int v = expr.vars[k];
        char var_name[64];
        is_real[k] = exprs[v]->is_real;

        double imag_sign = 1;
        if (exprs[v]->num == 1) {
          snprintf(var_name, sizeof(var_name), "i%d", exprs[v]->out_var);
        } else if (output_vars.find(exprs[v]->out_var) != output_vars.end()) {
          snprintf(var_name, sizeof(var_name), "output[%d]",
                   output_vars[exprs[v]->out_var]);
        } else {
          snprintf(var_name, sizeof(var_name), "w%d", exprs[v]->out_var - n);
        }
        if (expr.conj[k] && !is_real[k]) {
          imag_sign = -1;
        }
        if (expr.weight_ind[k][0] == 1 &&
            expr.weight_ind[k][1] == 0) {
          if (is_real[k]) {
            parts[k] = std::make_pair(var_name, "");
          } else {
            parts[k] = std::make_pair(std::string(var_name) + "[0]",
                                      sign_string(imag_sign) + var_name + "[1]");
          }
        } else if (expr.weight_ind[k][0] == -1 &&
                   expr.weight_ind[k][1] == 0) {
          if (is_real[k]) {
            parts[k] = std::make_pair(sign_string(-1) + var_name, "");
          } else {
            imag_sign *= -1;
            parts[k] = std::make_pair(sign_string(-1) + var_name + "[0]",
                                      sign_string(imag_sign) + var_name + "[1]");
          }
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
                     (imag_sign * expr.weight_ind[k][1]) < 0 ? '-' : ' ',
                     std::abs(expr.weight_ind[k][1]));
          }
          if (is_real[k]) {
            // This only makes sense for the case when variable is real
            parts[k] = std::make_pair(
                expr.weight_ind[k][0] == 0 ? "" : std::string(real_weight) + " * " + var_name,
                sign_string(imag_sign) +
                (strlen(imag_weight) <= 2 ? std::string(imag_weight) + var_name :
                 std::string(imag_weight) + " * " + var_name));
          } else {
            // This only makes sense for the case when variable is real
            parts[k] = std::make_pair(
                std::string("(") + std::string(real_weight) + "*" + var_name + "[0] - " +
                std::string(imag_weight) + "*" + var_name + "[1])",
                sign_string(imag_sign) +
                std::string("(") + std::string(real_weight) + "*" + var_name + "[1] + " +
                std::string(imag_weight) + "*" + var_name + "[0])");
          }
        }
      }
      std::string real_part, imag_part;
        real_part =
            parts[0].first +
            (parts[0].first.empty() || parts[1].first.empty() ? "" : " +") +
            parts[1].first;
        imag_part =
            parts[0].second +
            (parts[0].second.empty() || parts[1].second.empty() ? "" : " +") +
            parts[1].second;

      OutputVar output_var;
      output_var.output_var_name = output_var_name;
      output_var.real_part = real_part;
      output_var.imag_part = imag_part;
      output_var.output_index = -1;
      if (output_vars.find(h) != output_vars.end()) {
        output_var.output_index = output_vars[h];
      }
      declarations.push_back(output_var);
    }
  }

  fprintf(stderr, "template <typename T>\n");
  fprintf(stderr, "void dft_%d_compact(const float* input, float* output, int stride=1) {\n", n);
  for (int k = 2; k < weights.size(); ++k) {
    fprintf(stderr, "  const T kWeight%d = SimdHelper<T>::constant(%g);\n",
            k, weights[k]);
  }
  for (int h = 0; h < n; ++h) {
    const Expression& expr = *exprs[h];
    if (expr.num == 1 && expr.out_var < n) {
      fprintf(stderr, "  const T i%d = SimdHelper<T>::load(input + %d * stride);\n",
              expr.out_var, expr.out_var);
    }
  }
  std::map<int, std::string> assignments;
  for (int i = 0; i < declarations.size(); ++i) {
    const OutputVar& v = declarations[i];
    if (v.output_index >= 0) {
      if (v.real_part.size()) {
        assignments[v.output_index] = v.real_part;
      }
      if (v.imag_part.size()) {
        int output_index = v.output_index + n / 2;
        assignments[output_index] = v.imag_part;
      }

    } else {
      fprintf(stderr, "%s = ", v.output_var_name.c_str());
      if (v.real_part.size() &&
          v.imag_part.size()) {
        fprintf(stderr, " {%s, %s};\n",
                v.real_part.c_str(),
                v.imag_part.c_str());
      } else if (v.real_part.size()) {
        fprintf(stderr, "%s;\n",
                v.real_part.c_str());
      } else if (v.imag_part.size()) {
        fprintf(stderr, "%s;\n",
                v.imag_part.c_str());
      }
    }
  }
  for (const auto& assignment : assignments) {
    fprintf(stderr, "  SimdHelper<T>::store(output + %d * stride, %s);\n",
            assignment.first,
            assignment.second.c_str());
  }
  fprintf(stderr, "}\n");
}

int main(int ac, char* av[]) {
  for (int x = 2; x <= 32; x *= 2) {
    run_dft_builder(x);
  }
  return 0;
}
