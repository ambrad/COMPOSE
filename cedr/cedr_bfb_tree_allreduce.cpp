#include "cedr_bfb_tree_allreduce.hpp"

namespace cedr {
using mpi::Parallel;

BfbTreeAllReducer
::BfbTreeAllReducer (const Parallel::Ptr& p, const tree::Node::Ptr& tree,
                     // A leaf is a leaf node in the reduction tree. Each leaf
                     // has nfield scalars to reduce.
                     const Int nleaf, const Int nfield) {
  
}

} // namespace cedr
