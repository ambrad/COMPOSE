// COMPOSE version 1.0: Copyright 2018 NTESS. This software is released under
// the BSD license; see LICENSE in the top-level directory.

#ifndef INCLUDE_CEDR_BFB_TREE_ALLREDUCE
#define INCLUDE_CEDR_BFB_TREE_ALLREDUCE

#include "cedr_tree.hpp"

namespace cedr {

struct BfbTreeAllReducer {
  BfbTreeAllReducer(const mpi::Parallel::Ptr& p, const tree::Node::Ptr& tree,
                    // A leaf is a leaf node in the reduction tree. Each leaf
                    // has nfield scalars to reduce.
                    const Int nleaf, const Int nfield);

private:
  std::shared_ptr<const tree::NodeSets> ns_;
  std::vector<Real> bd_;
};

} // namespace cedr

#endif
