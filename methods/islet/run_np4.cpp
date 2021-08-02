#include "islet_np4.hpp"

#include "islet_util.hpp"
#include "npx.hpp"
#include "searchxnodes.hpp"
#include "islet_maxeigcomp.hpp"
#include "islet_pum.hpp"

static const Real oosqrt5 = 0.44721359549995793928;

static Real eval_lagrange_poly (const Int& n, const Real* xsup, const Real* ysup,
                                const Real& x) {
  Real y = 0;
  for (int i = 0; i < n; ++i) {
    Real f = 1;
    for (int j = 0; j < n; ++j)
      f *= (i == j) ?
        1 :
        (x - xsup[j]) / (xsup[i] - xsup[j]);
    y += f*ysup[i];
  }
  return y;
}

struct Options {
  Int ne_max;
  Real meam1_tol, pum_tol;

  Options ()
    : ne_max(1001), meam1_tol(1e-14), pum_tol(1e-7)
  {}
};

// Convex combination parameter for np=3 and np=4 combination that gives exactly
// 1 for the interpolant at gll_x[2] - 1 for the antisymmetric function with GLL
// point values (0, 1, -1, 0).
static Real calc_alpha () {
  const auto x_gll = islet::get_x_gll(4);
  const Real x0 = x_gll[1] + 1;
  Real y[4];
  npxstab<>::outer_eval(x0, y);
  const Real y3sum = y[2] - y[1];
  eval_lagrange_poly(x_gll, 4, x0, y);
  const Real y4sum = y[2] - y[1];
  return (1 - y4sum) / (y3sum - y4sum);
}

// original method impl using alpha from calc_alpha
static Real normalize_x (const Real* gll_x, const Real& x) {
  const Real x0 = gll_x[1];
  return (x - x0) / (1 - x0);
}
static void outer_eval (const Real* gll_x, const Real& x, Real v[4]) {
  const Real
    xbar = normalize_x(gll_x, gll_x[2]),
    ooxbar = 1 / xbar,
    ybar = 1 / (xbar - 1),
    xn = normalize_x(gll_x, x);
  v[0] = 0;
  v[1] = 1 + ybar*xn*((1 - ooxbar)*xn + ooxbar - xbar);
  v[2] = ybar*ooxbar*xn*(xn - 1);
  v[3] = ybar*xn*(xbar - xn);
}
static void eval_np4 (const Real* const x_gll, const Real& x, Real y[4]) {
  static constexpr Real
    alpha = 0.5527864045000416708,
    v = 0.427*(1 + alpha),
    x2 = 0.4472135954999579277, // 1/sqrt(5)
    x3 = 1 - x2,
    det = x2*x3*(x2 - x3),
    y2 = alpha,
    y3 = v,
    c1 = (x3*y2 - x2*y3)/det,
    c2 = (-x3*x3*y2 + x2*x2*y3)/det;
  if (x < x_gll[1] || x > x_gll[2]) {
    Real y4[4];
    eval_lagrange_poly(x_gll, 4, x, y4);
    if (x < x_gll[1]) {
      outer_eval(x_gll, -x, y);
      std::swap(y[0], y[3]);
      std::swap(y[1], y[2]);
    } else
      outer_eval(x_gll, x, y);
    const Real x0 = 1 - std::abs(x);
    const Real a = (c1*x0 + c2)*x0;
    for (int i = 0; i < 4; ++i)
      y[i] = a*y[i] + (1 - a)*y4[i];
  }
  else
    eval_lagrange_poly(x_gll, 4, x, y);
}

// For c = (0, c1, 1), test that our simpler expression matches eval_a.
static void test_eval_a () {
  Real c1 = 0.306;
  Np4InterpMethod im(0, c1, 1);
  const Int n = 100;
  for (Int i = 0; i <= n; ++i) {
    const Real
      x = -1 + (1 - oosqrt5)*i/n,
      x0 = 2*(1 - std::abs(x))/(1 - oosqrt5) - 1,
      a_simple = (c1 + (0.5 - c1)*x0)*(x0 + 1),
      a_eval = im.eval_a(x),
      re = std::abs(a_simple - a_eval); // relative to 1
    if (re > 1e-15)
      printf("x %1.3e a_simple %1.3e a_eval %1.3e re %1.1e\n",
             x, a_simple, a_eval, re);
  }
}

static Real run_maxeigcomp (const Np4InterpMethod::Ptr& uim, const Options& o) {
  MaxEigComputer mec;
  const auto meam1 = mec.run(4, o.ne_max, o.ne_max, o.meam1_tol, true, uim);
  printf("mec %1.3e\n", meam1);
  return meam1;
}

static Real run_pum (const UserInterpMethod::Ptr& uim, const Options& o) {
  pum::Options po;
  po.threaded = true;
  po.ntrial = 31;
  po.mec_ne = o.ne_max/10;
  po.perturb = 0.01;
  pum::PerturbedUniformMeshMetric pum(uim);
  Real pum_max = 0;
  printf("pum:"); fflush(stdout);
  for (Int ne = 3; ne <= 15; ++ne) {
    po.ne = ne;
    pum.reset_opts(po);
    const auto pum_val = pum.run();
    printf(" %1.1e", pum_val); fflush(stdout);
    pum_max = std::max(pum_max, pum_val);
  }
  printf("\npum_max %1.4e\n", pum_max);
  return pum_max;
}

/*
  |f(x) - p_i(x)| <= e_i(x)
  sum_i a_i = 1
  |f(x) - sum_i a_i p_i(x)|
    = |sum_i a_i f(x) - sum_i a_i p_i(x)|
    = |sum_i a_i (f(x) - p_i(x))|
   <= sum_i |a_i| |f(x) - p_i(x)|
    = sum_i |a_i| e_i(x)
 */
static void calc_metrics (const Np4InterpMethod& uim, Real metrics[3]) {
  const Int nseg = 100;
  const auto* xnodes = islet::get_x_gll(4);
  Real npm1 = 0, npm2 = 0, npm_max = 0;
  for (Int ireg = 0; ireg < 2; ++ireg) {
    const bool center = ireg == 1;
    const auto xs = xnodes[ireg], xe = xnodes[ireg+1];
    Real npm1_reg = 0, npm2_reg = 0, npm_max_reg = 0;
    for (Int seg = 0; seg < nseg; ++seg) {
      const auto x = xs + (seg + 0.5)*(xe - xs)/nseg;
      Real f = 1;
      if (ireg == 0) {
        Real f3 = 1, f4 = 1;
        for (Int i = 0; i < 3; ++i) f3 *= x - xnodes[i];
        for (Int i = 0; i < 4; ++i) f4 *= x - xnodes[i];
        const Real a = uim.eval_a(x);
        // Divide by 3!, 4!.
        f = std::abs(a*f3)/6 + std::abs((1 - a)*f4)/24;
      } else {
        for (Int i = 0; i < 4; ++i) f *= x - xnodes[i];
        f /= 24;
      }
      npm1_reg += std::abs(f);
      npm2_reg += slmm::square(f);
      npm_max_reg = std::max(npm_max_reg, std::abs(f));
    }
    const auto f = (center ? 1 : 2)*(xe - xs)/nseg;
    npm1 += f*npm1_reg;
    npm2 += f*npm2_reg;
    npm_max = std::max(npm_max, npm_max_reg);
  }
  metrics[0] = npm1;
  metrics[1] = std::sqrt(npm2);
  metrics[2] = npm_max;
}

static void optimize (Real best_metrics[3], const Options& o,
                      const bool c0_zero, const bool c2_one) {
  const Real alpha = calc_alpha();
  auto uim = std::make_shared<Np4InterpMethod>(0, alpha, alpha);
  MaxEigComputer mec;
  pum::Options po;
  po.threaded = true;
  po.ntrial = 31;
  po.mec_ne = o.ne_max/10;
  po.perturb = 0.01;
  pum::PerturbedUniformMeshMetric pum(uim);
  Int iteration = 0, expensive = 0;
  for (;;) {
    const Real
      c0 = c0_zero ? 0 : islet::urand(),
      c1 = islet::urand(),
      c2 = c2_one ? 1 : islet::urand();
    uim->reset_c(c0, c1, c2);

    Real metrics[3], meam1 = 1, pum_max = 1;
    calc_metrics(*uim, metrics);

    const auto print = [&] (const char* s) {
      printf("%14s: %1.16e %1.16e %1.16e | "
             "pum %1.3e meam1 %1.3e npm %1.3e %1.3e %1.3e %d %d\n",
             s, c0, c1, c2, pum_max, meam1, metrics[0], metrics[1], metrics[2],
             iteration, expensive);
    };

    bool fnd = false;
    for (Int i = 0; i < 3; ++i)
      if (metrics[i] < best_metrics[i])
        fnd = true;
    ++iteration;
    if ( ! fnd) continue;
    ++expensive;

    meam1 = mec.run(4, o.ne_max, o.ne_max, o.meam1_tol, true, uim);
    if (meam1 > o.meam1_tol) {
      if (meam1 < 1e-12) print("reject meam1");
      continue;
    }

    pum_max = 0;
    for (Int ne = 3; ne <= 15; ++ne) {
      po.ne = ne;
      pum.reset_opts(po);
      const auto pum_val = pum.run(o.pum_tol);
      pum_max = std::max(pum_max, pum_val);
      if (pum_max > o.pum_tol) break;
    }
    if (pum_max > o.pum_tol) {
      print("reject pum");
      continue;
    }

    bool all = true;
    for (Int i = 0; i < 3; ++i)
      if (metrics[i] > best_metrics[i])
        all = false;
    if (all)
      for (Int i = 0; i < 3; ++i)
        best_metrics[i] = metrics[i];

    print("accept");
  }
}

static void current_np4_to_c (Real c[3], const Real perturb = 0) {
  const Real
    sqrt5 = std::sqrt(5.0),
    oosqrt5 = 1.0 / sqrt5,
    alpha = 0.5527864045000416708 + perturb,
    v = 0.427*(1 + alpha),
    x2 = 0.4472135954999579277, // 1/sqrt(5) if the compiler can't handle oosqrt5
    x3 = 1 - x2,
    det = x2*x3*(x2 - x3),
    y2 = alpha,
    y3 = v,
    c1 = (x3*y2 - x2*y3)/det,
    c2 = (-x3*x3*y2 + x2*x2*y3)/det;
  Real x0[] = {0, 0.5*(1 - oosqrt5), 1 - oosqrt5};
  for (Int i = 0; i < 3; ++i)
    c[i] = (c1*x0[i] + c2)*x0[i];
}

// Test that our simpler impl matches the original np4 method and impl.
static void test_original () {
  Real c[3];
  current_np4_to_c(c);
  Np4InterpMethod op(c[0], c[1], c[2]);
  const auto x_gll = islet::get_x_gll(4);
  const Int n = 100;
  Int ne = 0;
  for (Int i = 0; i <= n; ++i) {
    const Real x = 2*(Real(i)/n) - 1;
    Real yop[4], yorig[4];
    op.eval(x, yop);
    eval_np4(x_gll, x, yorig);
    for (Int j = 0; j < 4; ++j) {
      const Real err = std::abs(yop[j] - yorig[j]);
      if (err > 10*std::numeric_limits<Real>::epsilon()) {
        printf("%d %1.3e %1.3e %1.3e %1.1e\n", j, x, yop[j], yorig[j], err);
        ++ne;
      };
    }
  }
}

// Test that the metrics specialization above matches the one we use everywhere
// else when alpha = 0.
static void test_metrics () {
  Real m[3], m4[3];
  Nodes nodes;
  nodes.init("4 1 | 0 4: 0 1 2 3 | 1 4: 0 1 2 3");
  calc_xnodes_metrics(nodes, islet::get_x_gll(4), m);
  Np4InterpMethod op(0, 0, 0);
  calc_metrics(op, m4);
  for (Int i = 0; i < 3; ++i)
    if (std::abs(m4[i] - m[i]) > 10*std::numeric_limits<Real>::epsilon())
      printf("FAIL test_metrics %d %1.16e %1.16e\n", i, m[i], m4[i]);
}

int main (int argc, char** argv) {
  printf("MaxEigComputer::unittest() %d\n", MaxEigComputer::unittest());
  test_eval_a();
  test_original();
  test_metrics();
  Real best_metrics[3]; {
    // We have a very good value; use this to make 2- and 3-dimensional the
    // searches faster.
    const auto uim = std::make_shared<Np4InterpMethod>(0, 0.306, 1);
    calc_metrics(*uim, best_metrics);
    printf("start: npm l1 %1.4e l2 %1.4e li %1.4e\n",
           best_metrics[0], best_metrics[1], best_metrics[2]);
  }
  if (argc > 1 && std::string(argv[1]) == "opt1") {
    Options o;
    o.ne_max = 3333;
    optimize(best_metrics, o, true, true);
  } else if (argc > 1 && std::string(argv[1]) == "opt2") {
    Options o;
    o.ne_max = 3333;
    optimize(best_metrics, o, true, false);
  } else if (argc > 1 && std::string(argv[1]) == "opt3") {
    Options o;
    o.ne_max = 3333;
    optimize(best_metrics, o, false, false);
  } else {
    const Real alpha = calc_alpha();
    printf("calc_alpha: %23.16e\n", alpha);

    const auto eval = [&] (const Real c[3]) {
      printf("c %1.16e %1.16e %1.16e\n", c[0], c[1], c[2]);
      const auto uim = std::make_shared<Np4InterpMethod>(c[0], c[1], c[2]);
      printf("eval_a(1/sqrt(5) - 1) %1.16e\n", uim->eval_a(oosqrt5-1));
      Real metrics[3];
      calc_metrics(*uim, metrics);
      printf("npm l1 %1.4e l2 %1.4e li %1.4e\n", metrics[0], metrics[1], metrics[2]);
      Options o;
      o.ne_max = (argc > 1 && std::string(argv[1]) == "dense") ? 1024 : 111;
      const auto mec = run_maxeigcomp(uim, o);
      run_pum(uim, o);
    };

    {
      Real c[3];
      c[0] = c[1] = c[2] = 1;
      eval(c);
    }

    for (const Real perturb : {0.0}) {
      Real c_cur[3];
      current_np4_to_c(c_cur, perturb);
      printf("current with perturb %1.2e\n", perturb);
      eval(c_cur);
    }

    // (1, 1, 1)     1.332e-15 pum 1.2516e-08 npm l1 1.5758e-02 l2 1.2782e-02 li 1.5109e-02
    // c_cur         2.021e-14 pum 3.6395e-04 npm l1 1.1687e-02 l2 9.1356e-03 li 9.4606e-03
    // (0, 0.306, 1) 1.110e-15 pum 9.9197e-09 npm l1 1.1611e-02 l2 9.0249e-03 li 9.0817e-03

    {
      const Real c[] = {0,0.306,1};
      eval(c);
    }
  }
}
