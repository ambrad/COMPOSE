#ifndef INCLUDE_NPX_HPP
#define INCLUDE_NPX_HPP

#include "types.hpp"
#include "tables.hpp"
#include "interpmethod.hpp"

template <typename Scalar>
void eval_lagrange_poly (const Scalar* x_gll, const Int& np, const Scalar& x,
                         Scalar* const y) {
  for (int i = 0; i < np; ++i) {
    Scalar f = 1;
    for (int j = 0; j < np; ++j)
      f *= (i == j) ?
        1 :
        (x - x_gll[j]) / (x_gll[i] - x_gll[j]);
    y[i] = f;
  }
}

template <typename Scalar=Real>
struct npx {
  static void eval (const Int np, const Scalar& x, Scalar* const v) {
    eval_lagrange_poly(islet::get_x_gll(np), np, x, v);
  }
};

template <typename Scalar=Real>
struct npxstab : public npx<Scalar> {
  template <int np, int nreg>
  static void eval (const std::array<int, nreg>& order,
                    const std::array<int, nreg>& os,
                    const Scalar& x, Scalar* const v) {
    if (x > 0) {
      eval<np, nreg>(order, os, -x, v);
      for (int i = 0; i < np/2; ++i)
        std::swap(v[i], v[np-i-1]);
      return;
    }
    const auto x_gll = islet::get_x_gll(np);
    bool done = false;
    for (Int i = 0; i < nreg; ++i)
      if (x < x_gll[i+1]) {
        std::fill(v, v + np, 0);
        eval_lagrange_poly(x_gll + os[i], order[i], x, v + os[i]);
        done = true;
        break;
      }
    if ( ! done)
      eval_lagrange_poly(x_gll, np, x, v);
  }

  static void outer_eval (const Real& x, Real v[4]) {
    static const Real
      sqrt5 = std::sqrt(5.0),
      oosqrt5 = 1.0 / sqrt5,
      xbar = (2*oosqrt5) / (1 + oosqrt5),
      ooxbar = 1 / xbar,
      ybar = 1 / (xbar - 1);
    const Real xn = (x + oosqrt5) / (1 + oosqrt5);
    v[0] = 0;
    v[1] = 1 + ybar*xn*((1 - ooxbar)*xn + ooxbar - xbar);
    v[2] = ybar*ooxbar*xn*(xn - 1);
    v[3] = ybar*xn*(xbar - xn);
  }

  static void np4_subgrid_eval (const Real& x, Real y[4]) {
    static const Real
      sqrt5 = std::sqrt(5.0),
      oosqrt5 = 1.0 / sqrt5,
      alpha = 0.5527864045000416708 + 1e-4, // for a tiny bit of extra stability
      v = 0.427*(1 + alpha),
      x2 = 0.4472135954999579277, // 1/sqrt(5) if the compiler can't handle oosqrt5
      x3 = 1 - x2,
      det = x2*x3*(x2 - x3),
      y2 = alpha,
      y3 = v,
      c1 = (x3*y2 - x2*y3)/det,
      c2 = (-x3*x3*y2 + x2*x2*y3)/det;
    if (x < -oosqrt5 || x > oosqrt5) {
      if (x < -oosqrt5) {
        outer_eval(-x, y);
        std::swap(y[0], y[3]);
        std::swap(y[1], y[2]);
      } else
        outer_eval(x, y);
      Real y4[4];
      eval<4,0>({}, {}, x, y4);
      const Real x0 = 1 - std::abs(x);
      const Real a = (c1*x0 + c2)*x0;
      for (int i = 0; i < 4; ++i)
        y[i] = a*y[i] + (1 - a)*y4[i];
    } else
      eval<4,0>({}, {}, x, y);
  }

  static void eval(const Int& np, const Scalar& x, Scalar* const v);

  static int ooa_vs_np (const int np) {
    if (np == 5) return 2;
    return np - 1 - ((np-1)/3);
  }

  template <int np, int nreg, int alphanp>
  static void eval (const std::array<int, nreg>& subnp,
                    const std::array<int, nreg>& os,
                    const std::array<Scalar, nreg*alphanp>& alphac,
                    const Scalar& x, Scalar* const v) {
    if (x > 0) {
      eval<np, nreg, alphanp>(subnp, os, alphac, -x, v);
      for (int i = 0; i < np/2; ++i)
        std::swap(v[i], v[np-i-1]);
      return;
    }
    const auto x_gll = islet::get_x_gll(np);
    bool done = false;
    for (int i = 0; i < nreg; ++i)
      if (x < x_gll[i+1]) {
        eval_lagrange_poly(x_gll, np, x, v);
        if (subnp[i] < np) {
          Real w[12] = {0};
          eval_lagrange_poly(x_gll + os[i], subnp[i], x, w + os[i]);
          Real alpha = 0;
          if (alphanp == 1)
            alpha = alphac[i];
          else {
            assert(alphanp <= 3);
            const auto alpha_r_gll = islet::get_x_gll(alphanp);
            const auto r = (x - x_gll[i]) / (x_gll[i+1] - x_gll[i]);
            Real a[3];
            eval_lagrange_poly(alpha_r_gll, alphanp, r, a);
            for (int j = 0; j < alphanp; ++j)
              alpha += alphac[alphanp*i + j] * a[j];
          }
          for (int j = 0; j < np; ++j)
            v[j] = alpha*v[j] + (1 - alpha)*w[j];
        }
        done = true;
        break;
      }
    if ( ! done)
      eval_lagrange_poly(x_gll, np, x, v);
  }
};

struct InterpMethod {
  typedef std::shared_ptr<InterpMethod> Ptr;

  enum Type { notype, gll, npx, npxstab, user };
  Int np;
  Type type;
  std::shared_ptr<UserInterpMethod> uim;

  static Type convert (const std::string& s) {
    if (s == "gll") return gll;
    if (s == "npx") return npx;
    if (s == "npxstab") return npxstab;
    if (s == "user") return user;
    throw std::runtime_error(std::string("Not an InterpMethod::Type: ") + s);
  }

  static std::string convert (const Type& t) {
    if (t == gll) return "gll";
    if (t == npx) return "npx";
    if (t == npxstab) return "npxstab";
    if (t == user) return "user";
    throw std::runtime_error("Not an InterpMethod::Type.");
  }

  InterpMethod () : np(-1), type(notype) {}
  InterpMethod (Int inp, Type itype) : np(inp), type(itype) {}
  InterpMethod (const std::shared_ptr<UserInterpMethod>& iuim)
    : np(iuim->get_np()), type(user), uim(iuim) {}

  const Real* get_xnodes() const {
    return uim ? uim->get_xnodes() : islet::get_x_gll(np);
  };
};

void op_eval(const InterpMethod& im, const Real a_src, Real* v);

#endif
