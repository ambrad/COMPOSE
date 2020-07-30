// COMPOSE version 1.0: Copyright 2018 NTESS. This software is released under
// the BSD license; see LICENSE in the top-level directory.

#ifndef INCLUDE_CEDR_BFB_TREE_ALLREDUCE
#define INCLUDE_CEDR_BFB_TREE_ALLREDUCE

#include "cedr_tree.hpp"

namespace cedr {

// Use a tree and point-to-point communication to implement all-reduce. If the
// tree is independent of process deomposition, then
// BfbTreeAllReducer::allreduce is BFB-invariant to process decomposition.
template <typename ExeSpace = Kokkos::DefaultExecutionSpace>
struct BfbTreeAllReducer {
  typedef typename cedr::impl::DeviceType<ExeSpace>::type Device;
  typedef BfbTreeAllReducer<ExeSpace> Me;
  typedef std::shared_ptr<Me> Ptr;
  typedef Kokkos::View<Real*, Device> RealList;
  typedef typename Kokkos::View<Real*, Device>::HostMirror RealListHost;
  typedef Kokkos::View<const Real*, Device> ConstRealList;

  BfbTreeAllReducer(const mpi::Parallel::Ptr& p, const tree::Node::Ptr& tree,
                    // A leaf is a leaf node in the reduction tree. The global
                    // tree has nleaf leaves. Each leaf has nfield scalars to
                    // reduce.
                    const Int nleaf,
                    // nlocal is number of values to reduce in this rank.  nfld
                    // is number of fields.
                    const Int nlocal, const Int nfield);

  void get_host_buffers_sizes(size_t& buf1, size_t& buf2);
  void set_host_buffers(Real* buf1, Real* buf2);

  void finish_setup();

  // In Fortran, these are formatted as send(nlocal, nfield), recv(nfield).
  void allreduce(const ConstRealList& send, const RealList& recv);

  static Int unittest(const mpi::Parallel::Ptr& p);

private:
  mpi::Parallel::Ptr p_;
  Int nlocal_, nfield_;
  std::shared_ptr<const tree::NodeSets> ns_;
  RealListHost bd_;

  void init(const mpi::Parallel::Ptr& p, const tree::Node::Ptr& tree,
            const Int nleaf, const Int nlocal, const Int nfield);
  const Real* get_send_host(const ConstRealList& send);
};

} // namespace cedr

#endif
