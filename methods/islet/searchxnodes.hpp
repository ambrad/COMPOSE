#ifndef INCLUDE_SEARCHXNODES_HPP
#define INCLUDE_SEARCHXNODES_HPP

#include "types.hpp"
#include "nodalbasis.hpp"

#include <vector>
#include <sstream>

struct SearchXnodesOptions {
  enum Metric { node_position, perturbed_uniform_mesh };
  Int npt0, npt, nlev, ns;
  Real tol, pum_metric_thr, node_position_metric_thr;
  bool threaded, verbose, positive_weights, nondecreasing_weights;
  Metric metric;
  SearchXnodesOptions ()
    : npt0(21), npt(11), nlev(4), ns(3999),
      tol(1e-14), pum_metric_thr(1e5), node_position_metric_thr(1e5),
      threaded(true), verbose(true),
      positive_weights(true), nondecreasing_weights(true),
      metric(node_position)
  {}
};

// l1 only
Real calc_xnodes_metric(const Nodes& nodes, const Real* const xnodes);
// l1, l2, linf
void calc_xnodes_metrics(const Nodes& nodes, const Real* const xnodes, Real* metrics);

void calc_lebesgue_consts(const Nodes& nodes, const Real* const xnodes, Real* metrics);

void calc_weights(const Nodes& nodes, const Real* const xnode, Real* const wt);

void search_xnodes(const Nodes& nodes, Real* const xnodes_best,
                   Real& meam1_best, Real& pum_metric_best, Real& xnodes_metric_best,
                   Real& wtr_best, const SearchXnodesOptions opts = SearchXnodesOptions());

// Increment each entry of an array idx of length n from 0 to idxlim
// inclusive. Return true if that is possible; false if
//     idx[i] = idxlim for i = 0:n-1.
bool incr(Int* idx, const Int idxlim, const Int n, const bool forward = true);

Int unittest_search_xnodes();

struct MetricsTracker {
  typedef std::shared_ptr<MetricsTracker> Ptr;

  MetricsTracker(Int np, bool very_strict = false);

  void set_pum_max(Real pum); // optional; default is 1
  Real get_pum_max () const { return pum_max; }

  // Min pum seen so far. If none, return 1.
  Real get_pum_min () const { return pum_min; }

  // Compute metrics. Return whether these are provisionally acceptable.
  bool acceptable_metrics(const Nodes& nodes, const Real* xnodes,
                          const Real* metrics) const;
  // pum needs to be <= this value to update.
  Real pum_to_accept(const Nodes& nodes, const Real* xnodes,
                     const Real* metrics) const;
  // Would update based on metrics and pum?
  bool would_update(const Real* metrics, const Real& pum) const;
  // Do the update.
  void update(const Real* metrics, const Real& pum);

  void get_metrics(Real pum, Real* metrics) const;

  bool write(std::ofstream& os);
  bool read(std::ifstream& os);

private:
  static const Int nmet = 3, nbin = 30;
  Real best_metrics[nmet*nbin]; // l1, l2, linf
  Real pum_bins[nbin+1];
  Real pum_max, pum_min;
};

#endif
