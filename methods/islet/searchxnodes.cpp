#include "searchxnodes.hpp"
#include "tables.hpp"
#include "npx.hpp"
#include "maxeigcomp.hpp"
#include "pum.hpp"
#include "isl.hpp"
#include "util.hpp"

static Real factorial (const Int n) {
  Real f = 1;
  for (Int i = 2; i <= n; ++i) f *= i;
  return f;
}

void calc_xnodes_metrics (const Nodes& nodes, const Real* const xnodes, Real* metrics) {
  const Int np = nodes.get_np(), nph = np/2, nseg = 100;
  Real npm1 = 0, npm2 = 0, npm_max = 0;
  for (Int ireg = 0; ireg < nph; ++ireg) {
    const bool center = np % 2 == 0 && ireg == nph-1;
    const auto xs = xnodes[ireg], xe = xnodes[ireg+1];
    const auto subnp  = nodes.get_subnp()[ireg];
    const auto active = nodes.get_nodes()[ireg];
    Real npm1_reg = 0, npm2_reg = 0, npm_max_reg = 0;
    for (Int seg = 0; seg < nseg; ++seg) {
      const auto x = xs + (seg + 0.5)*(xe - xs)/nseg;
      Real f = 1;
      for (Int i = 0; i < subnp; ++i)
        f *= x - xnodes[active[i]];
      npm1_reg += std::abs(f);
      npm2_reg += slmm::square(f);
      npm_max_reg = std::max(npm_max_reg, std::abs(f));
    }
    const auto fac = factorial(subnp);
    const auto f = (center ? 1 : 2)*(xe - xs)/fac/nseg;
    npm1 += f*npm1_reg;
    npm2 += f*npm2_reg/fac; // need an extra fac b/c of square
    npm_max = std::max(npm_max, npm_max_reg/fac);
  }
  metrics[0] = npm1;
  metrics[1] = std::sqrt(npm2);
  metrics[2] = npm_max;
}

Real calc_xnodes_metric (const Nodes& nodes, const Real* const xnodes) {
  Real metrics[3];
  calc_xnodes_metrics(nodes, xnodes, metrics);
  return metrics[0];
}

void calc_lebesgue_consts (const Nodes& nodes, const Real* const xnodes, Real* metrics) {
  const Int np = nodes.get_np(), nph = np/2, nseg = 100;
  Real npm1 = 0, npm2 = 0, npm_max = 0;
  for (Int ireg = 0; ireg < nph; ++ireg) {
    const bool center = np % 2 == 0 && ireg == nph-1;
    const auto xs = xnodes[ireg], xe = xnodes[ireg+1];
    const auto subnp  = nodes.get_subnp()[ireg];
    const auto active = nodes.get_nodes()[ireg];
    Real npm1_reg = 0, npm2_reg = 0, npm_max_reg = 0;
    for (Int seg = 0; seg < nseg; ++seg) {
      const auto x = xs + (seg + 0.5)*(xe - xs)/nseg;
      Real f = 0;
      for (Int i = 0; i < subnp; ++i) {
        Real g = 1;
        for (Int j = 0; j < subnp; ++j) {
          if (j == i) continue;
          g *= (x - xnodes[active[j]])/(xnodes[active[i]] - xnodes[active[j]]);
        }
        f += std::abs(g);
      }
      npm1_reg += f;
      npm2_reg += slmm::square(f);
      npm_max_reg = std::max(npm_max_reg, f);
    }
    const auto f = (center ? 1 : 2)*(xe - xs)/nseg;
    npm1 += f*npm1_reg;
    npm2 += f*npm2_reg; // need an extra fac b/c of square
    npm_max = std::max(npm_max, npm_max_reg);
  }
  metrics[0] = npm1;
  metrics[1] = std::sqrt(npm2);
  metrics[2] = npm_max;
}

MetricsTracker::MetricsTracker (const Int np, bool very_strict) {
  pum_min = pum_max = 1;
  const Real fac = std::pow(std::numeric_limits<Real>::epsilon(), 1.0/nbin);
  pum_bins[0] = 1;
  for (Int i = 0; i < nbin; ++i) pum_bins[i+1] = pum_bins[i]*fac;
  //islet::prarr("pum_bins",pum_bins,nbin);
  // From findbasic.
  Real iv[3];
  if (false) {
    switch (np) {
    case  4: iv[0] = 1.575830e-02; iv[1] = 1.278167e-02; iv[2] = 1.510916e-02; break;
    case  5: iv[0] = 2.549179e-03; iv[1] = 2.582596e-03; iv[2] = 4.154765e-03; break;
    case  6: iv[0] = 2.393393e-04; iv[1] = 2.104595e-04; iv[2] = 2.816403e-04; break; // use early findcombo results too
    case  7: iv[0] = 5.557714e-05; iv[1] = 4.790768e-05; iv[2] = 6.934868e-05; break;
    case  8: iv[0] = 7.265137e-06; iv[1] = 7.988089e-06; iv[2] = 1.618560e-05; break;
    case  9: iv[0] = 7.860606e-07; iv[1] = 7.683143e-07; iv[2] = 1.179540e-06; break;
    case 10: iv[0] = 1.075794e-07; iv[1] = 9.532486e-08; iv[2] = 1.540700e-07; break;
    case 11: iv[0] = 1.589070e-08; iv[1] = 1.867321e-08; iv[2] = 3.345386e-08; break;
    case 12: iv[0] = 6.963036e-10; iv[1] = 8.920290e-10; iv[2] = 1.715838e-09; break;
    case 13: iv[0] = 4.127583e-11; iv[1] = 4.809655e-11; iv[2] = 9.223544e-11; break;
    default: iv[0] = iv[1] = iv[2] = 1;
    }
    // 12 is taking too long, and incomplete search strongly suggests we can
    // restrict our attention to pum < 1e-6.
    if (np == 12) set_pum_max(1e-6);
  } else {
    iv[0] = iv[1] = iv[2] = 1;
  }
  if (very_strict) {
    // Use GLL nodal search results to make this as small as possible.
    Real pum_max;
    switch (np) {
    case  4: pum_max = 2.6609e-15; break;
    case  5: pum_max = 1.0509e-07; break;
    case  6: pum_max = 1.0809e-09; break;
    case  7: pum_max = 4.7909e-09; break;
    case  8: pum_max = 8.8109e-09; break;
    case  9: pum_max = 3.6409e-09; break;
    case 10: pum_max = 1.4409e-08; break;
    case 11: pum_max = 3.5009e-07; break;
    case 12: pum_max = 1.4509e-07; break;
    default: assert(0);
    }
    set_pum_max(pum_max);
  }
  for (Int i = 0; i < nmet*nbin; ++i) best_metrics[i] = iv[i % nmet];
}

void MetricsTracker::set_pum_max (const Real pum_max_) {
  pum_max = pum_max_;
  assert(pum_max <= 1 && pum_max > 0);
}

bool MetricsTracker
::acceptable_metrics (const Nodes& nodes, const Real* xnodes,
                      const Real* metrics) const {
  for (Int i = 0; i < nmet*nbin; ++i)
    if (metrics[i % nmet] < best_metrics[i])
      return true;
  return false;
}

Real MetricsTracker
::pum_to_accept (const Nodes& nodes, const Real* xnodes,
                 const Real* metrics) const {
  for (Int i = 0; i < nmet*nbin; ++i)
    if (metrics[i % nmet] < best_metrics[i])
      return std::min(pum_max, pum_bins[i/nmet]);
  return 0;
}

bool MetricsTracker
::would_update (const Real* metrics, const Real& pum) const {
  if (pum > pum_max) return false;
  Int bin;
  for (bin = 0; bin < nbin; ++bin)
    if (bin == nbin-1 || pum >= pum_bins[bin+1])
      break;
  for (Int i = 0; i < nmet; ++i)
    if (metrics[i] < best_metrics[nmet*bin + i])
      return true;
  return false;
}

void MetricsTracker::update (const Real* metrics, const Real& pum) {
  bool updated = false;
  for (Int bin = 0; bin < nbin; ++bin) {
    if (pum > pum_bins[bin]) break;
    for (Int i = 0; i < nmet; ++i)
      if (metrics[i] < best_metrics[nmet*bin + i]) {
        best_metrics[nmet*bin + i] = metrics[i];
        updated = true;
      }
  }
  if (updated) pum_min = std::min(pum_min, pum);
}

void MetricsTracker::get_metrics (Real pum, Real* metrics) const {
  Int bin;
  for (bin = 0; bin < nbin; ++bin)
    if (pum > pum_bins[bin]) break;
  bin = std::max(0, bin-1);
  for (Int i = 0; i < nmet; ++i)
    metrics[i] = best_metrics[nmet*bin + i];
}

bool MetricsTracker::write (std::ofstream& os) {
  using islet::write;
  return (write(os, nmet) &&
          write(os, nbin) &&
          write(os, nmet*nbin, best_metrics) &&
          write(os, nbin+1, pum_bins) &&
          write(os, pum_max) &&
          write(os, pum_min));
}

bool MetricsTracker::read (std::ifstream& os) {
  using islet::read;
  Int lnmet, lnbin, n;
  const bool ok = (read(os, lnmet) && lnmet == nmet &&
                   read(os, lnbin) && lnbin == nbin &&
                   read(os, n, best_metrics) && n == nmet*nbin &&
                   read(os, n, pum_bins) && n == nbin+1 &&
                   read(os, pum_max) &&
                   read(os, pum_min));
  return ok;
}

static void symmetrize (const Int n, Real* x) {
  const Int nh = n/2;
  for (Int i = 0; i < nh; ++i) x[n-1-i] = -x[i];
}

void calc_weights (const Nodes& nodes, const Real* const xnode, Real* const wt) {
  // Quadrature coefficients.
  const Int qn = 7;
  const Real* const qx = islet::get_x_gll(qn);
  const Real* const qw = islet::get_w_gll(qn);
  const Int np = nodes.get_np();
  Real v[islet::np_max], integral[islet::np_max] = {0};
  for (Int ireg = 0; ireg < np-1; ++ireg) {
    Real reg_integral[islet::np_max] = {0};
    for (Int qi = 0; qi < qn; ++qi) {
      const auto alpha = 0.5*(qx[qi] + 1);
      const auto x = (1 - alpha)*xnode[ireg] + alpha*xnode[ireg+1];
      eval(nodes.get_np(), nodes.include_bdy(), xnode,
           nodes.get_subnp(), nodes.get_nodes(), x, v);
      for (Int i = 0; i < np; ++i)
        reg_integral[i] += qw[qi]*v[i];
    }
    const auto fac = 0.5*(xnode[ireg+1] - xnode[ireg]);
    for (Int i = 0; i < np; ++i)
      integral[i] += fac*reg_integral[i];
  }
  // Numerically symmetrize.
  for (Int ireg = 0; ireg < np/2; ++ireg) {
    // std::min is to prevent a spurious -Warray-bounds warning.
    const Int other = std::min(islet::np_max-1, np-ireg-1);
    integral[ireg] = integral[other] =
      0.5*(integral[ireg] + integral[other]);
  }
  for (Int i = 0; i < np; ++i) wt[i] = integral[i];
}

static bool has_all_positive_weights (const Nodes& nodes, const Real* const xnode) {
  Real wt[islet::np_max];
  calc_weights(nodes, xnode, wt);
  bool pve = true;
  const Int np = nodes.get_np();
  for (Int i = 0; i < np; ++i)
    if (wt[i] <= 0)
      pve = false;
  return pve;
}

struct MyInterpMethod : public UserInterpMethod {
  MyInterpMethod (const Nodes& nodes_) : nodes(nodes_), xnodes(nullptr) {}
  void set_xnodes_ptr (const Real* xnodes_) { xnodes = xnodes_; };
  void eval (const Real& x, Real* const v) override {
    assert(xnodes);
    assert( ! nodes.include_bdy() || xnodes[0] == -1);
    ::eval(nodes, xnodes, x, v);
  }
  const Real* get_xnodes () const override { return xnodes; }
  virtual Int get_np () const override { return nodes.get_np(); }
private:
  Nodes nodes;
  const Real* xnodes;
};

bool incr (Int* idx, const Int idxlim, const Int n, const bool forward) {
  bool fnd = false;
  if (forward) {
    for (Int i = 0; i < n; ++i)
      if (idx[i] < idxlim) {
        fnd = true;
        ++idx[i];
        for (Int j = 0; j < i; ++j) idx[j] = 0;
        break;
      }
  } else {
    for (Int i = n-1; i >= 0; --i)
      if (idx[i] < idxlim) {
        fnd = true;
        ++idx[i];
        for (Int j = i+1; j < n; ++j) idx[j] = 0;
        break;
      }
  }
  return fnd;
}

static void calc_wts_metrics (const Int np, const Real* wt,
                              bool& all_pve_wts, Real& ratio) {
  all_pve_wts = true;
  for (Int i = 0; i < np; ++i) if (wt[i] <= 0) all_pve_wts = false;
  Real wtmin = 10, wtmax = -1;
  for (Int i = 0; i < np; ++i) wtmin = std::min(wtmin, wt[i]);
  for (Int i = 0; i < np; ++i) wtmax = std::max(wtmax, wt[i]);
  ratio = wtmax/wtmin;
}

void search_xnodes (const Nodes& nodes, Real* const xnodes_best,
                    Real& meam1_best, Real& pum_metric_best, Real& xnodes_metric_best,
                    Real& wtr_best, const SearchXnodesOptions opts) {
  static const Int N = (islet::np_max + 1)/2;
  const Int i0 = nodes.include_bdy() ? 1 : 0;
  Real xspan[N], xnodes[islet::np_max], xnodes0[islet::np_max];
  Real metric_best = 1e5;
  meam1_best = pum_metric_best = xnodes_metric_best = wtr_best = 1e5;
  Int idx[N];
  bool fnd = false;
  const Int np = nodes.get_np(), nnode = np/2;
  const Real* const xgll = islet::get_x_gll(np);
  Real scale = 1;
  for (Int i = 0; i < np; ++i) xnodes_best[i] = -10;
  for (Int i = 0; i < np; ++i) xnodes0[i] = xgll[i];
  for (Int i = 0; i <  N; ++i) xspan[i] = 0;
  const auto im = std::make_shared<MyInterpMethod>(nodes);
  im->set_xnodes_ptr(xnodes);
  MaxEigComputer mec(opts.threaded);
  pum::PerturbedUniformMeshMetric::Ptr pum; {
    const auto wrapper = std::make_shared<InterpMethod>(im);
    pum::Options o;
    o.threaded = opts.threaded;
    o.ntrial = 31;
    o.mec_ne = 3;
    o.ne = 4;
    o.perturb = 0.01;
    pum = std::make_shared<pum::PerturbedUniformMeshMetric>(wrapper, o);
  }
  for (Int ilev = 0; ilev < opts.nlev; ++ilev) {
    const Int npt = ilev == 0 ? opts.npt0 : opts.npt;
    for (Int i = i0; i < nnode; ++i) xspan[i] = 0.5*scale*(xgll[i+1] - xgll[i]);
    for (Int i = i0; i < nnode; ++i) idx[i] = 0;
    do {
      for (Int i =  0; i <    np; ++i) xnodes[i] = xnodes0[i];
      for (Int i = i0; i < nnode; ++i) xnodes[i] += (2*(Real(idx[i])/(npt - 1)) - 1)*xspan[i];
      bool good = xnodes[0] >= -1;
      for (Int i = i0; i < nnode; ++i)
        if ((i > 0 && xnodes[i] <= xnodes[i-1]) || xnodes[i] >= xnodes[i+1] || xnodes[i] >= 0)
          good = false;
      if ( ! good) continue;
      symmetrize(np, xnodes);
      Real wt[islet::np_max], wtr;
      calc_weights(nodes, xnodes, wt);
      bool all_pve_wts;
      calc_wts_metrics(np, wt, all_pve_wts, wtr);
      if (opts.positive_weights) good = all_pve_wts;
      if ( ! good) continue;
      if (opts.nondecreasing_weights)
        for (Int i = 1; i < (np+1)/2; ++i)
          if (wt[i-1] > wt[i]) {
            good = false;
            break;
          }
      if ( ! good) continue;
      const Real xnodes_metric = calc_xnodes_metric(nodes, xnodes);
      if (xnodes_metric > opts.node_position_metric_thr) continue;
      auto mea = mec.run(np, 11, 11, opts.tol, true, im);
      if (mea > opts.tol) continue;
      const auto pum_metric = pum->run(metric_best);
      if (pum_metric > opts.pum_metric_thr) continue;
      Real metric;
      if (fnd) {
        metric = opts.metric == SearchXnodesOptions::node_position ?
          xnodes_metric : pum_metric;
        if (metric >= metric_best) continue;
      }
      const Int ns = opts.ns;
      mea = mec.run(np, ns, ns, opts.tol, true, im);
      if (mea <= opts.tol) {
        if ( ! fnd) {
          metric = opts.metric == SearchXnodesOptions::node_position ?
            xnodes_metric : pum_metric;
        }
        fnd = true;
        if (metric < metric_best) {
          if (opts.verbose)
            printf("  meam1 %9.2e wtr %10.3e npm %10.3e pum %10.3e\n",
                   mea, wtr, xnodes_metric, pum_metric);
          metric_best = metric;
          meam1_best = mea;
          pum_metric_best = pum_metric;
          xnodes_metric_best = xnodes_metric;
          wtr_best = wtr;
          for (Int i = 0; i < np; ++i) xnodes_best[i] = xnodes[i];
        }
      }
    } while (incr(idx+i0, npt-1, nnode-i0));
    scale *= 3.0/(npt - 1);
    if (fnd)
      for (Int i = 0; i < np; ++i)
        xnodes0[i] = xnodes_best[i];
  }
}

Int unittest_search_xnodes () {
  Int nerr = 0;
  for (const bool forward : {true, false}) {
    Int idx[3] = {0};
    Int cnt = 0;
    do { ++cnt; } while (incr(idx, 4, 3, forward));
    if (cnt != 125) ++nerr;
    if (idx[1] != 4) ++nerr;
  }
  return nerr;
}
