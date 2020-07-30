// COMPOSE version 1.0: Copyright 2018 NTESS. This software is released under
// the BSD license; see LICENSE in the top-level directory.

#include "cedr_bfb_tree_allreduce.hpp"

namespace cedr {
using mpi::Parallel;

template <typename ES>
BfbTreeAllReducer<ES>
::BfbTreeAllReducer (const Parallel::Ptr& p, const tree::Node::Ptr& tree,
                     const Int nleaf, const Int nlocal, const Int nfield) {
  init(p, tree, nleaf, nlocal, nfield);
}

template <typename ES>
void BfbTreeAllReducer<ES>
::init (const Parallel::Ptr& p, const tree::Node::Ptr& tree,
        const Int nleaf, const Int nlocal, const Int nfield) {
  p_ = p;
  ns_ = tree::analyze(p, nleaf, tree);
  nlocal_ = nlocal;
  nfield_ = nfield;
}

template <typename ES>
void BfbTreeAllReducer<ES>
::get_host_buffers_sizes (size_t& buf1, size_t& buf2) {
  buf1 = ((impl::OnGpu<ES>::value ? nlocal_ : 0) + ns_->nslots)*nfield_;
  buf2 = 0;
}

template <typename ES>
void BfbTreeAllReducer<ES>
::set_host_buffers (Real* buf1, Real* buf2) {
  if ( ! buf1) return;
  size_t s1, s2;
  get_host_buffers_sizes(s1, s2);
  bd_ = RealList(buf1, s1);
}

template <typename ES>
void BfbTreeAllReducer<ES>
::finish_setup () {
  size_t s1, s2;
  get_host_buffers_sizes(s1, s2);
  if (bd_.size() > 0) {
    cedr_assert(bd_.size() == s1);
    return;
  }
  bd_ = RealList("bd_", s1);
}

template <typename ES>
const Real* BfbTreeAllReducer<ES>
::get_send_host (const ConstRealList& send) {
  cedr_assert(send.extent_int(0) == nlocal_*nfield_);
  if ( ! impl::OnGpu<ES>::value) return send.data();
  RealListHost m(bd_.data() + ns_->nslots * nfield_, nlocal_*nfield_);
  Kokkos::deep_copy(m, send);
  return m.data();
}

template <typename ES>
void BfbTreeAllReducer<ES>
::allreduce (const ConstRealList& send, const RealList& recv) {
  const Real* send_host = get_send_host(send);
}

template <typename ES>
Int BfbTreeAllReducer<ES>::unittest (const Parallel::Ptr& p) {
  return 0;
}

} // namespace cedr

#ifdef KOKKOS_ENABLE_SERIAL
template class cedr::BfbTreeAllReducer<Kokkos::Serial>;
#endif
#ifdef KOKKOS_ENABLE_OPENMP
template class cedr::BfbTreeAllReducer<Kokkos::OpenMP>;
#endif
#ifdef KOKKOS_ENABLE_CUDA
template class cedr::BfbTreeAllReducer<Kokkos::Cuda>;
#endif
#ifdef KOKKOS_ENABLE_THREADS
template class cedr::BfbTreeAllReducer<Kokkos::Threads>;
#endif
