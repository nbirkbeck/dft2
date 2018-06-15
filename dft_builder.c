#include <math.h>
#include <stdlib.h>
#include <complex.h>
#include <assert.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <map>
#include <memory>
#include <string.h>
#include <typeinfo>

#define PRINT_CONCISE 0
#define SIMPLIFY_DEBUG 0

struct Expression {
  int num = 0;
  int out_var = 0;
  bool is_real = 0;
  int vars[2];
  std::complex<float> weights[2];
  bool conj[2];
  int weight_ind[2][2];
  int num_refs = 0;
};

std::vector<std::unique_ptr<Expression> > expressions;

void dft_builder(Expression** input,
                 Expression** output, int n, bool inverse,
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
  dft_builder(&temp[0], &out1[0], n / 2, inverse, exprs);
  dft_builder(&temp[0] + n / 2, &out2[0], n / 2, inverse, exprs);

  for (int k = 0; k < n / 2; ++k) {
    std::complex<float> w = std::complex<float>(
        cos(2. * M_PI * k / n), -sin(2. * M_PI * k / n));
    int k1 = k;
    bool conj_v1 = false;
    bool conj_v2 = false;

    if (!inverse) {
      // Values in output > n / 4 don't exist; use conjugate symmetry
      if (k1 > n / 4) {
        conj_v1 = true;
        conj_v2 = true;
        k1 = n / 2 - k1;
      }
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

    if (k1 == 0 || inverse) {
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
  if (!inverse) {
    for (int k = n / 2 + 1; k < n; ++k) {
      output[k] = 0;
    }
  }
}

std::string sign_string(double w) {
  return w < 0 ? "-" : "";
}

class SimpleExpr {
 public:
  SimpleExpr() {}
  virtual ~SimpleExpr() {}
  virtual bool is_null() {return false; }
  virtual std::string to_string() = 0;
  virtual SimpleExpr* copy() const = 0;

};

class NullExpr: public SimpleExpr {
 public:
  virtual bool is_null() {return true; }
  virtual std::string to_string() { return ""; }
  virtual SimpleExpr* copy() const { return new NullExpr(); }
};

class VarExpr: public SimpleExpr {
 public:
  VarExpr(std::string var_name) : var_name_(var_name) {}
  std::string to_string() {
    return var_name_;
  }
  virtual SimpleExpr* copy() const {
    return new VarExpr(var_name_);
  }
  std::string var_name_;
};

class NegateExpr: public SimpleExpr {
 public:
  NegateExpr(std::unique_ptr<SimpleExpr> a): op_(std::move(a)) {}
  std::string to_string() {
#if PRINT_CONCISE
    return std::string("-" + op_->to_string());
#else
    return std::string("sub(kWeight0, ") + op_->to_string() + ")";
#endif
  }
  SimpleExpr* copy() const {
    return new NegateExpr(std::unique_ptr<SimpleExpr>(op_->copy()));
  }
  std::unique_ptr<SimpleExpr> op_;
};

class AddExpr: public SimpleExpr {
 public:
  AddExpr(std::unique_ptr<SimpleExpr> a,
          std::unique_ptr<SimpleExpr> b) :
      op1_(std::move(a)), op2_(std::move(b)) {}
  std::string to_string() {
#if PRINT_CONCISE
    return "(" + op1_->to_string() + " + " + op2_->to_string() + ")";
#else
    return std::string("add( " + op1_->to_string()
                     + "," + op2_->to_string() + ")");
#endif

  }
  SimpleExpr* copy() const {
    return new AddExpr(std::unique_ptr<SimpleExpr>(op1_->copy()),
                       std::unique_ptr<SimpleExpr>(op2_->copy()));
  }
  std::unique_ptr<SimpleExpr> op1_;
  std::unique_ptr<SimpleExpr> op2_;
};

class SubExpr: public SimpleExpr {
 public:
  SubExpr(std::unique_ptr<SimpleExpr> a,
          std::unique_ptr<SimpleExpr> b) :
      op1_(std::move(a)), op2_(std::move(b)) {}
  std::string to_string() {
#if PRINT_CONCISE
      return std::string("( " + op1_->to_string()
                         + "-" + op2_->to_string() + ")");
#else
      return std::string("sub( " + op1_->to_string()
                         + "," + op2_->to_string() + ")");
#endif
  }
  SimpleExpr* copy() const {
    return new SubExpr(std::unique_ptr<SimpleExpr>(op1_->copy()),
                       std::unique_ptr<SimpleExpr>(op2_->copy()));
  }
  std::unique_ptr<SimpleExpr> op1_;
  std::unique_ptr<SimpleExpr> op2_;
};

class MulExpr: public SimpleExpr {
 public:
  MulExpr(std::unique_ptr<SimpleExpr> a,
          std::unique_ptr<SimpleExpr> b) :
      op1_(std::move(a)), op2_(std::move(b)) { }
  std::string to_string() {
#if PRINT_CONCISE
    return std::string(op1_->to_string() + "*" + op2_->to_string());
#else
    return std::string("mul( " + op1_->to_string()
                     + "," + op2_->to_string() + ")");
#endif
  }
  SimpleExpr* copy() const {
    return new MulExpr(std::unique_ptr<SimpleExpr>(op1_->copy()),
                       std::unique_ptr<SimpleExpr>(op2_->copy()));
  }
  std::unique_ptr<SimpleExpr> op1_;
  std::unique_ptr<SimpleExpr> op2_;
};


std::unique_ptr<SimpleExpr> null() {
  return std::unique_ptr<SimpleExpr>(new NullExpr);
}


std::unique_ptr<SimpleExpr>
add(std::unique_ptr<SimpleExpr> a,
    std::unique_ptr<SimpleExpr> b) {
  if (!a->is_null() && !b->is_null())
    return std::unique_ptr<SimpleExpr>(new AddExpr(std::move(a),
                                                 std::move(b)));

  if (a->is_null() && b->is_null()) return null();
  if (!a->is_null()) return std::move(a);
  if (!b->is_null()) return std::move(b);
}

std::unique_ptr<SimpleExpr> sign(float exp_sign, std::unique_ptr<SimpleExpr> e) {
  if (exp_sign < 0) {
    return std::unique_ptr<SimpleExpr>(new NegateExpr(std::move(e)));
  } else return e;
}

std::unique_ptr<SimpleExpr>
sub(std::unique_ptr<SimpleExpr> a,
    std::unique_ptr<SimpleExpr> b) {
  if (!a->is_null() && !b->is_null())
    return std::unique_ptr<SimpleExpr>(new SubExpr(std::move(a),                                                 std::move(b)));
  if (!a->is_null()) return std::move(a);
  if (!b->is_null()) return sign(-1, std::move(b));
  return null();

  /*
std::string sub(std::string a, std::string b, bool paren) {
  if (paren) {
    //return "( " + a + " - " + b + ")";
    return "sub(" + a + "," + b + ")";
  } else {
    return "sub(" + a + "," + b + ")";
    }*/
}

std::unique_ptr<SimpleExpr> var(std::string e) {
  return std::unique_ptr<SimpleExpr>(new VarExpr(e));
}

std::unique_ptr<SimpleExpr>
mul(std::unique_ptr<SimpleExpr> a,
    std::unique_ptr<SimpleExpr> b) {
  if (!a->is_null() && !b->is_null())
    return std::unique_ptr<SimpleExpr>(new MulExpr(std::move(a),
                                                   std::move(b)));
  if (!a->is_null()) return std::move(a);
  if (!b->is_null()) return std::move(b);
  return null();
}

std::unique_ptr<SimpleExpr> simplify(std::unique_ptr<SimpleExpr> expr,
                                     bool move_neg_up = false) {
  AddExpr* add_expr = dynamic_cast<AddExpr*>(expr.get());
  SubExpr* sub_expr = dynamic_cast<SubExpr*>(expr.get());
  std::string input = expr->to_string();

  if (add_expr) {
    std::unique_ptr<SimpleExpr> a = simplify(std::move(add_expr->op1_), move_neg_up);
    std::unique_ptr<SimpleExpr> b = simplify(std::move(add_expr->op2_), move_neg_up);
    NegateExpr* a_neg = dynamic_cast<NegateExpr*>(a.get());
    NegateExpr* b_neg = dynamic_cast<NegateExpr*>(b.get());
    if (b_neg) { // && !a_neg) {
      auto out = sub(std::move(a), std::move(b_neg->op_));
#if SIMPLIFY_DEBUG
      fprintf(stderr, "changing add to sub: %s  to   %s\n", input.c_str(), out->to_string().c_str());
#endif
      return std::move(out);
    }
    if (a_neg && !b_neg) {
      auto out = sub(std::move(b), std::move(a_neg->op_));
#if SIMPLIFY_DEBUG
      fprintf(stderr, "changing add to sub: %s   to   %s\n", input.c_str(), out->to_string().c_str());
#endif
      return std::move(out);
    }
    MulExpr* a_mul = dynamic_cast<MulExpr*>(a.get());
    MulExpr* b_mul = dynamic_cast<MulExpr*>(b.get());
    if (a_mul && b_mul &&
        a_mul->op1_->to_string() == b_mul->op1_->to_string()) {
#if SIMPLIFY_DEBUG
      fprintf(stderr, "distributing k * a + k * b -> k *(a + b) with %s  to   %s\n",
              a_mul->op1_->to_string().c_str(),
              input.c_str());
#endif
      return mul(std::move(a_mul->op1_),
                 add(std::move(a_mul->op2_), std::move(b_mul->op2_)));
    }
    return add(std::move(a), std::move(b));
  }
  if (sub_expr) {
    std::unique_ptr<SimpleExpr> a = simplify(std::move(sub_expr->op1_), move_neg_up);
    std::unique_ptr<SimpleExpr> b = simplify(std::move(sub_expr->op2_), move_neg_up);
    NegateExpr* b_neg = dynamic_cast<NegateExpr*>(b.get());
    NegateExpr* a_neg = dynamic_cast<NegateExpr*>(a.get());
    if (a_neg && b_neg) {
#if SIMPLIFY_DEBUG
      fprintf(stderr, "Double neg\n");
      // -1 * ((a) - (b))
#endif
      return sign(-1, sub(std::move(a_neg->op_), std::move(b_neg->op_)));
    }
    if (b_neg) {
#if SIMPLIFY_DEBUG
      fprintf(stderr, "sub, b neg to add\n");
#endif
      return add(std::move(a), std::move(b_neg->op_));
    }
    MulExpr* a_mul = dynamic_cast<MulExpr*>(a.get());
    MulExpr* b_mul = dynamic_cast<MulExpr*>(b.get());
    if (a_mul && b_mul &&
        a_mul->op1_->to_string() == b_mul->op1_->to_string()) {
#if SIMPLIFY_DEBUG
      fprintf(stderr, "distributing k * a - k * b -> k *(a - b) with k = %s, a = %s, b = %s to   %s\n",
              a_mul->op1_->to_string().c_str(),
              a_mul->op2_->to_string().c_str(),
              b_mul->op2_->to_string().c_str(),
              input.c_str());
#endif
      return mul(std::move(a_mul->op1_),
                 sub(std::move(a_mul->op2_), std::move(b_mul->op2_)));
    }
    return sub(std::move(a), std::move(b));
  }
  MulExpr* mul_expr = dynamic_cast<MulExpr*>(expr.get());
  if (mul_expr) {
    std::unique_ptr<SimpleExpr> a = simplify(std::move(mul_expr->op1_), move_neg_up);
    std::unique_ptr<SimpleExpr> b = simplify(std::move(mul_expr->op2_), move_neg_up);
    NegateExpr* a_neg = dynamic_cast<NegateExpr*>(a.get());
    NegateExpr* b_neg = dynamic_cast<NegateExpr*>(b.get());
    if (a_neg && b_neg) {
#if SIMPLIFY_DEBUG
      fprintf(stderr, "Removing neg\n");
#endif
      return mul(std::move(a_neg->op_), std::move(b_neg->op_));
    }

    if (move_neg_up && a_neg) {
      auto out = sign(-1, mul(std::move(a_neg->op_), std::move(b)));
#if SIMPLIFY_DEBUG
      fprintf(stderr, "Moving neg up, a: %s    %s\n", input.c_str(), out->to_string().c_str());
#endif
      return std::move(out);
    }
    if (move_neg_up && b_neg) {
      auto out = sign(-1, mul(std::move(a), std::move(b_neg->op_)));
#if SIMPLIFY_DEBUG
      fprintf(stderr, "Moving neg up, b: %s    %s\n", input.c_str(), out->to_string().c_str());
#endif
      return std::move(out);
    }
    return mul(std::move(a), std::move(b));
  }
  return std::move(expr);
}

std::unique_ptr<SimpleExpr> copy(const std::unique_ptr<SimpleExpr>& in) {
  return std::unique_ptr<SimpleExpr>(in->copy());
}

void run_dft_builder(int n, bool inverse) {
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
  if (inverse) {
    exprs.resize(n + n - 2);
    for (int i = 1; i < n / 2; ++i) {
      int i1 = i + n - 1;
      exprs[i1] = std::unique_ptr<Expression>(new Expression);
      exprs[i1]->out_var = i1;
      exprs[i1]->num = 2;
      exprs[i1]->is_real = 0;
      exprs[i1]->vars[0] = i;
      exprs[i1]->vars[1] = n / 2 + i;
      exprs[i1]->weights[0] = 1;
      exprs[i1]->weights[1] = std::complex<float>(0, 1);

      int i2 = 2 * n - 2 - i;
      exprs[i2] = std::unique_ptr<Expression>(new Expression);
      exprs[i2]->out_var = i2;
      exprs[i2]->num = 2;
      exprs[i2]->is_real = 0;
      exprs[i2]->vars[0] = i;
      exprs[i2]->vars[1] = n / 2 + i;
      exprs[i2]->weights[0] = 1;
      exprs[i2]->weights[1] = std::complex<float>(0, -1);

      // This doesn't make sense!
      input[i] = exprs[i2].get();
      input[n - i] = exprs[i1].get();
    }
  }
  dft_builder(&input[0], &output[0], n, inverse, exprs);
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
  if (inverse) {
    for (int h = 0; h < output.size(); ++h) {
      if (output[h]) {
        output_vars[output[h]->out_var] = h;
      } else {
        fprintf(stderr, "Output %d doesnt exist\n", h);
      }
    }
  } else {
    for (int h = 0; h < output.size() / 2 + 1; ++h) {
      output_vars[output[h]->out_var] = h;
    }
  }

  struct OutputVar {
    std::string output_var_name;
    std::string real_part;
    std::string imag_part;
    int output_index;
  };
  std::vector<OutputVar> declarations;

  for (int h = 0; h < exprs.size(); ++h) {
    Expression& expr = *exprs[h];
    if (expr.num == 1 && expr.out_var < n) {
    } else {
      char output_var_name[64];
      if (output_vars.find(h) == output_vars.end()) {
        if (expr.is_real) {
          snprintf(output_var_name, 64, "  const T_VEC w%d ",
                   expr.out_var - n);
        } else {
          snprintf(output_var_name, 64, "  const T_VEC w%d[2] ",
                   expr.out_var - n);
        }
      }
      std::vector<std::pair<std::unique_ptr<SimpleExpr>,
                            std::unique_ptr<SimpleExpr> > > parts(2);
      std::vector<bool> is_real(2);
      if (!expr.is_real && h < output[0]->out_var &&
          expr.num == 2 &&
          exprs[expr.vars[0]]->is_real &&
          exprs[expr.vars[1]]->is_real &&
              //expr.vars[0] < n && (expr.num == 1 || expr.vars[1] < n) &&
              //!expr.is_real && exprs[expr.vars[0]]->is_real && exprs[expr.vars[1]]->is_real &&
          ((fabs(expr.weights[0].real()) - fabs(expr.weights[1].imag())) < 1e-4 &&
           (fabs(expr.weights[0].imag()) - fabs(expr.weights[1].real())) < 1e-4 &&
           (fabs(expr.weights[0].real()) - fabs(expr.weights[0].imag())) > 1e-4)) {
        /*
        fprintf(stderr, "expr h = %d/%d %d is a ref %d(%d) %d(%d) num = %d (%f,%f)  (%f,%f)\n",
                h, n, expr.out_var,expr.vars[0], expr.conj[0],
                expr.num == 1 ? 0 : expr.vars[1],
                expr.conj[1], expr.num,
                expr.weights[0].real(), expr.weights[0].imag(),
                expr.weights[1].real(), expr.weights[1].imag());
        */
        /*
        for (int h2 = 0; h2 < exprs.size(); ++h2) {
          if (exprs[h2]->vars[0] == h) {
            fprintf(stderr, "  replacing %d\n", h2);
            exprs[h2]->vars[0] = expr.vars[0];
          } else if (exprs[h2]->vars[1] == h) {
            fprintf(stderr, "  replacing %d\n", h2);
            exprs[h2]->vars[1] = expr.vars[0];
          }
        }
        */
        expr.num = 0; //
        continue;
      }
      for (int k = 0; k < expr.num; ++k) {
        int v = expr.vars[k];
        char var_name[2][64];
        is_real[k] = exprs[v]->is_real;

        double imag_sign = 1;
        if (exprs[v]->num == 0) {
          if (exprs[exprs[v]->vars[0]]->num == 1) {
            snprintf(var_name[0], sizeof(var_name[0]), "i%d", exprs[v]->vars[0]);
          } else {
            snprintf(var_name[0], sizeof(var_name[0]), "w%d", exprs[v]->vars[0] - n);
          }
          if (exprs[exprs[v]->vars[1]]->num == 1) {
            snprintf(var_name[1], sizeof(var_name[1]), "i%d", exprs[v]->vars[1]);
          } else {
            snprintf(var_name[1], sizeof(var_name[1]), "w%d", exprs[v]->vars[1] - n);
          }
          if (exprs[v]->weight_ind[1][0] == 0 &&
              exprs[v]->weight_ind[1][1] == -1) {
            imag_sign *= -1;
          }
          //          fprintf(stderr, "vars: %s %s %f\n", var_name[0], var_name[1], imag_sign);
        } else if (exprs[v]->num == 1) {
          snprintf(var_name[0], sizeof(var_name[0]), "i%d", exprs[v]->out_var);
          strcpy(var_name[1], var_name[0]);
        } else if (output_vars.find(exprs[v]->out_var) != output_vars.end()) {
          snprintf(var_name[0], sizeof(var_name[0]), "output[%d]",
                   output_vars[exprs[v]->out_var]);
          strcpy(var_name[1], var_name[0]);
        } else {
          if (is_real[k]) {
            snprintf(var_name[0], sizeof(var_name[0]), "w%d", exprs[v]->out_var - n);
            strcpy(var_name[1], var_name[0]);
          } else {
            snprintf(var_name[0], sizeof(var_name[0]), "w%d[0]", exprs[v]->out_var - n);
            snprintf(var_name[1], sizeof(var_name[1]), "w%d[1]", exprs[v]->out_var - n);
          }
        }
        if (expr.conj[k] && !is_real[k]) {
          imag_sign *= -1;
        }
        if (expr.weight_ind[k][0] == 0 &&
            expr.weight_ind[k][1] == 1) {
          if (is_real[k]) {
            parts[k] = std::make_pair(null(),
                                      sign(imag_sign, var(var_name[0])));
          } else {
            parts[k] = std::make_pair(sign(-1, var(var_name[1])),
                                      sign(imag_sign, var(var_name[0])));
          }
        } else if (expr.weight_ind[k][0] == 0 &&
                   expr.weight_ind[k][1] == -1) {
          imag_sign *= -1;
          if (is_real[k]) {
            parts[k] = std::make_pair(null(),
                                      sign(imag_sign, var(var_name[0])));
          } else {
            parts[k] = std::make_pair(var(var_name[1]),
                                      sign(imag_sign, var(var_name[0])));
          }
        } else if (expr.weight_ind[k][0] == 1 &&
            expr.weight_ind[k][1] == 0) {
          if (is_real[k]) {
            parts[k] = std::make_pair(var(var_name[0]), null());
          } else {
            parts[k] = std::make_pair(var(var_name[0]),
                                      sign(imag_sign, var(var_name[1])));
          }
        } else if (expr.weight_ind[k][0] == -1 &&
                   expr.weight_ind[k][1] == 0) {
          if (is_real[k]) {
            parts[k] = std::make_pair(sign(-1, var(var_name[0])), null());
          } else {
            imag_sign *= -1;
            parts[k] = std::make_pair(sign(-1, var(var_name[0])),
                                      sign(imag_sign, var(var_name[1])));
          }
        } else {
          std::unique_ptr<SimpleExpr> real_weight;
          std::unique_ptr<SimpleExpr> imag_weight;
          bool real_neg = false;
          bool imag_neg = false;
          if (expr.weight_ind[k][0] == -1) {
            //snprintf(real_weight_name, sizeof(real_weight_name), "-");
            real_neg = true;
          } else if (expr.weight_ind[k][0] != 1) {
            char real_weight_name[64] = {0};
            snprintf(real_weight_name, sizeof(real_weight_name), "kWeight%d",
                     std::abs(expr.weight_ind[k][0]));
            real_weight = sign(expr.weight_ind[k][0], var(real_weight_name));
          }
          if (expr.weight_ind[k][1] == -1) {
            //snprintf(imag_weight_name, sizeof(imag_weight_name), "-");
            imag_neg = true;
          } else if (expr.weight_ind[k][1] != 1) {
            char imag_weight_name[64] = {0};
            snprintf(imag_weight_name, sizeof(imag_weight_name), "kWeight%d",
                     std::abs(expr.weight_ind[k][1]));
            imag_weight = sign(imag_sign * expr.weight_ind[k][1], var(imag_weight_name));
          }
          if (is_real[k]) {
            // This only makes sense for the case when variable is real
            parts[k] = std::make_pair(
                    expr.weight_ind[k][0] == 0 ? std::unique_ptr<SimpleExpr>(new NullExpr) :
                    !real_weight ? sign(real_neg ? -1 : 1, var(var_name[0])) :
                    mul(std::move(real_weight), var(var_name[0])),
                sign(imag_sign,
                     (!imag_weight ? sign(imag_neg ? -1 : 1,  var(var_name[0])) :
                      mul(std::move(imag_weight), var(var_name[0])))));
          } else {
            assert(imag_weight);
            assert(real_weight);
            // This only makes sense for the case when variable is real
            parts[k] = std::make_pair(
                sub(mul(copy(real_weight), var(var_name[0])),
                    mul(copy(imag_weight), var(var_name[1]))),
                sign(imag_sign,
                     add(mul(copy(real_weight), var(var_name[1])),
                         mul(copy(imag_weight), var(var_name[0])))));
          }
        }
      }
      auto real = add(std::move(parts[0].first),
                      std::move(parts[1].first));
      auto imag = add(std::move(parts[0].second),
                      std::move(parts[1].second));
      for (int i = 0; i < 10; ++i) {
        real = simplify(std::move(real), i && (i % 3 == 0));
        imag = simplify(std::move(imag), i && (i % 3 == 0));
      }
      std::string real_part = real->to_string();
      std::string imag_part = imag->to_string();
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

  fprintf(stderr, "template <typename T, typename I=float>\n");
  fprintf(stderr, "void %sdft_%d_compact(const I* input, I* output, int stride=1) { \\ \n",
          inverse ? "i" : "", n);
  for (int k = 0; k < weights.size(); ++k) {
    if (k == 1) continue;
    fprintf(stderr, "  const T_VEC kWeight%d = constant(%gf); \\ \n",
            k, weights[k]);
  }
  for (int h = 0; h < n; ++h) {
    const Expression& expr = *exprs[h];
    if (expr.num == 1 && expr.out_var < n) {
      fprintf(stderr, "  const T_VEC i%d = load(input + %d * stride); \\ \n",
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
      if (!inverse) {
        if (v.imag_part.size()) {
          int output_index = v.output_index + n / 2;
          assignments[output_index] = v.imag_part;
        }
      }
    } else {
      fprintf(stderr, "%s = ", v.output_var_name.c_str());
      if (v.real_part.size() &&
          v.imag_part.size()) {
        fprintf(stderr, " {%s, %s}; \\ \n",
                v.real_part.c_str(),
                v.imag_part.c_str());
      } else if (v.real_part.size()) {
        fprintf(stderr, "%s; \\ \n",
                v.real_part.c_str());
      } else if (v.imag_part.size()) {
        fprintf(stderr, "%s; \\ \n",
                v.imag_part.c_str());
      }
    }
  }
  const char* store = 1 ? "store" : "SimdHelper<T, I>::store";
  for (const auto& assignment : assignments) {
    fprintf(stderr, "  %s(output + %d * stride, %s);  \\\n",
            store,
            assignment.first,
            assignment.second.c_str());
  }
  fprintf(stderr, "}\n");
}

int main(int ac, char* av[]) {
  for (int x = 2; x <= 32; x *= 2) {
    run_dft_builder(x, false);
  }
  for (int x = 2; x <= 32; x *= 2) {
    run_dft_builder(x, true);
  }
  return 0;
}
