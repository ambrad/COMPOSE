#include "cedr_qlt.hpp"

#include <sys/time.h>

#include <cassert>
#include <cmath>

#include <set>
#include <list>
#include <limits>
#include <algorithm>

namespace cedr {
namespace qlt {

class Timer {
public:
  enum Op { tree, analyze, trcrinit, trcrgen, trcrcheck,
            qltrun, qltrunl2r, qltrunr2l, snp, waitall,
            total, NTIMERS };
  static inline void init () {
#ifdef QLT_TIME
    for (int i = 0; i < NTIMERS; ++i) {
      et_[i] = 0;
      cnt_[i] = 0;
    }
#endif
  }
  static inline void reset (const Op op) {
#ifdef QLT_TIME
    et_[op] = 0;
    cnt_[op] = 0;
#endif
  }
  static inline void start (const Op op) {
#ifdef QLT_TIME
    gettimeofday(&t_start_[op], 0);
    ++cnt_[op];
#endif
  }
  static inline void stop (const Op op) {
#ifdef QLT_TIME
    timeval t2;
    gettimeofday(&t2, 0);
    const timeval& t1 = t_start_[op];
    static const double us = 1.0e6;
    et_[op] += (t2.tv_sec*us + t2.tv_usec - t1.tv_sec*us - t1.tv_usec)/us;
#endif
  }
# define tpr(op) do {                                                   \
    printf("%-20s %10.3e %10.1f (%4d %10.3e)\n",                        \
           #op, et_[op], 100*et_[op]/tot, cnt_[op], et_[op]/cnt_[op]);  \
  } while (0)
  static void print () {
#ifdef QLT_TIME
    const double tot = et_[total];
    tpr(tree); tpr(analyze);
    tpr(trcrinit); tpr(trcrgen); tpr(trcrcheck);
    tpr(qltrun); tpr(qltrunl2r); tpr(qltrunr2l); tpr(snp); tpr(waitall);
    printf("%-20s %10.3e %10.1f\n", "total", tot, 100.0);
#endif
  }
#undef tpr
private:
#ifdef QLT_TIME
  static timeval t_start_[NTIMERS];
  static double et_[NTIMERS];
  static int cnt_[NTIMERS];
#endif
};
#ifdef QLT_TIME
timeval Timer::t_start_[Timer::NTIMERS];
double Timer::et_[Timer::NTIMERS];
int Timer::cnt_[Timer::NTIMERS];
#endif

namespace impl {
struct NodeSets {
  typedef std::shared_ptr<const NodeSets> ConstPtr;
  
  enum : int { mpitag = 42 };

  // A node in the tree that is relevant to this rank.
  struct Node {
    // Rank of the node. If the node is in a level, then its rank is my rank. If
    // it's not in a level, then it is a comm partner of a node on this rank.
    Int rank;
    // Globally unique identifier; cellidx if leaf node, ie, if nkids == 0.
    Int id;
    // This node's parent, a comm partner, if such a partner is required.
    const Node* parent;
    // This node's kids, comm partners, if such partners are required. Parent
    // and kid nodes are pruned relative to the full tree over the mesh to
    // contain just the nodes that matter to this rank.
    Int nkids;
    const Node* kids[2];
    // Offset factor into bulk data. An offset is a unit; actual buffer sizes
    // are multiples of this unit.
    Int offset;

    Node () : rank(-1), id(-1), parent(nullptr), nkids(0), offset(-1) {}
  };

  // A level in the level schedule that is constructed to orchestrate
  // communication. A node in a level depends only on nodes in lower-numbered
  // levels (l2r) or higher-numbered (r2l).
  //
  // The communication patterns are as follows:
  //   > l2r
  //   MPI rcv into kids
  //   sum into node
  //   MPI send from node
  //   > r2l
  //   MPI rcv into node
  //   solve QP for kids
  //   MPI send from kids
  struct Level {
    struct MPIMetaData {
      Int rank;   // Rank of comm partner.
      Int offset; // Offset to start of buffer for this comm.
      Int size;   // Size of this buffer in units of offsets.
    };
    
    // The nodes in the level.
    std::vector<Node*> nodes;
    // MPI information for this level.
    std::vector<MPIMetaData> me, kids;
    // Have to keep requests separate so we can call waitall if we want to.
    mutable std::vector<MPI_Request> me_req, kids_req;
  };
  
  // Levels. nodes[0] is level 0, the leaf level.
  std::vector<Level> levels;
  // Number of data slots this rank needs. Each node owned by this rank, plus
  // kids on other ranks, have an associated slot.
  Int nslots;
  
  // Allocate a node. The list node_mem_ is the mechanism for memory ownership;
  // node_mem_ isn't used for anything other than owning nodes.
  Node* alloc () {
    node_mem_.push_front(Node());
    return &node_mem_.front();
  }

  void print(std::ostream& os) const;
  
private:
  std::list<Node> node_mem_;
};

void NodeSets::print (std::ostream& os) const {
  std::stringstream ss;
  if (levels.empty()) return;
  const Int myrank = levels[0].nodes[0]->rank;
  ss << "pid " << myrank << ":";
  ss << " #levels " << levels.size();
  for (size_t i = 0; i < levels.size(); ++i) {
    const auto& lvl = levels[i];
    ss << "\n  " << i << ": " << lvl.nodes.size();
    std::set<Int> ps, ks;
    for (size_t j = 0; j < lvl.nodes.size(); ++j) {
      const auto n = lvl.nodes[j];
      for (Int k = 0; k < n->nkids; ++k)
        if (n->kids[k]->rank != myrank)
          ks.insert(n->kids[k]->rank);
      if (n->parent && n->parent->rank != myrank)
        ps.insert(n->parent->rank);
    }
    ss << " |";
    for (const auto& e : ks) ss << " " << e;
    if ( ! lvl.kids.empty()) ss << " (" << lvl.kids.size() << ") |";
    for (const auto& e : ps) ss << " " << e;
    if ( ! lvl.me.empty()) ss << " (" << lvl.me.size() << ")";
  }
  ss << "\n";
  os << ss.str();
}

// Find tree depth, assign ranks to non-leaf nodes, and init 'reserved'.
Int init_tree (const tree::Node::Ptr& node, Int& id) {
  node->reserved = nullptr;
  Int depth = 0;
  for (Int i = 0; i < node->nkids; ++i) {
    cedr_assert(node.get() == node->kids[i]->parent);
    depth = std::max(depth, init_tree(node->kids[i], id));
  }
  if (node->nkids) {
    node->rank = node->kids[0]->rank;
    node->cellidx = id++;
  } else {
    cedr_throw_if(node->cellidx < 0 || node->cellidx >= id,
                  "cellidx is " << node->cellidx << " but should be between " <<
                  0 << " and " << id);
  }
  return depth + 1;
}

void level_schedule_and_collect (
  NodeSets& ns, const Int& my_rank, const tree::Node::Ptr& node, Int& level,
  bool& need_parent_ns_node)
{
  cedr_assert(node->rank != -1);
  level = -1;
  bool make_ns_node = false;
  for (Int i = 0; i < node->nkids; ++i) {
    Int kid_level;
    bool kid_needs_ns_node;
    level_schedule_and_collect(ns, my_rank, node->kids[i], kid_level,
                               kid_needs_ns_node);
    level = std::max(level, kid_level);
    if (kid_needs_ns_node) make_ns_node = true;
  }
  ++level;
  // Is parent node needed for isend?
  const bool node_is_owned = node->rank == my_rank;
  need_parent_ns_node = node_is_owned;
  if (node_is_owned || make_ns_node) {
    cedr_assert( ! node->reserved);
    NodeSets::Node* ns_node = ns.alloc();
    // Levels hold only owned nodes.
    if (node_is_owned) ns.levels[level].nodes.push_back(ns_node);
    node->reserved = ns_node;
    ns_node->rank = node->rank;
    ns_node->id = node->cellidx;
    ns_node->parent = nullptr;
    if (node_is_owned) {
      // If this node is owned, it needs to have information about all kids.
      ns_node->nkids = node->nkids;
      for (Int i = 0; i < node->nkids; ++i) {
        const auto& kid = node->kids[i];
        if ( ! kid->reserved) {
          // This kid isn't owned by this rank. But need it for irecv.
          NodeSets::Node* ns_kid;
          kid->reserved = ns_kid = ns.alloc();
          ns_node->kids[i] = ns_kid;
          cedr_assert(kid->rank != my_rank);
          ns_kid->rank = kid->rank;
          ns_kid->id = kid->cellidx;
          ns_kid->parent = nullptr; // Not needed.
          // The kid may have kids in the original tree, but in the tree pruned
          // according to rank, it does not.
          ns_kid->nkids = 0;
        } else {
          // This kid is owned by this rank, so fill in its parent pointer.
          NodeSets::Node* ns_kid = static_cast<NodeSets::Node*>(kid->reserved);
          ns_node->kids[i] = ns_kid;
          ns_kid->parent = ns_node;
        }
      }
    } else {
      // This node is not owned. Update the owned kids with its parent.
      ns_node->nkids = 0;
      for (Int i = 0; i < node->nkids; ++i) {
        const auto& kid = node->kids[i];
        if (kid->reserved && kid->rank == my_rank) {
          NodeSets::Node* ns_kid = static_cast<NodeSets::Node*>(kid->reserved);
          ns_node->kids[ns_node->nkids++] = ns_kid;
          ns_kid->parent = ns_node;
        }
      }
    }
  }
}

void level_schedule_and_collect (NodeSets& ns, const Int& my_rank,
                                 const tree::Node::Ptr& tree) {
  Int iunused;
  bool bunused;
  level_schedule_and_collect(ns, my_rank, tree, iunused, bunused);
}

void consolidate (NodeSets& ns) {
  auto levels = ns.levels;
  ns.levels.clear();
  for (const auto& level : levels)
    if ( ! level.nodes.empty())
      ns.levels.push_back(level);
}

typedef std::pair<Int, NodeSets::Node*> RankNode;

void init_offsets (const Int my_rank, std::vector<RankNode>& rns,
                   std::vector<NodeSets::Level::MPIMetaData>& mmds, Int& offset) {
  // Set nodes on my rank to have rank -1 so that they sort first.
  for (auto& rn : rns)
    if (rn.first == my_rank)
      rn.first = -1;

  // Sort so that all comms with a given rank are contiguous. Stable sort so
  // that rns retains its order, in particular in the leaf node level.
  std::stable_sort(rns.begin(), rns.end());

  // Collect nodes into groups by rank and set up comm metadata for each group.
  Int prev_rank = -1;
  for (auto& rn : rns) {
    const Int rank = rn.first;
    if (rank == -1) {
      if (rn.second->offset == -1)
        rn.second->offset = offset++;
      continue;
    }
    if (rank != prev_rank) {
      cedr_assert(rank > prev_rank);
      prev_rank = rank;
      mmds.push_back(NodeSets::Level::MPIMetaData());
      auto& mmd = mmds.back();
      mmd.rank = rank;
      mmd.offset = offset;
      mmd.size = 0;
    }
    ++mmds.back().size;
    rn.second->offset = offset++;
  }
}

// Set up comm data. Consolidate so that there is only one message between me
// and another rank per level. Determine an offset for each node, to be
// multiplied by data-size factors later, for use in data buffers.
void init_comm (const Int my_rank, NodeSets& ns) {
  ns.nslots = 0;
  for (auto& lvl : ns.levels) {
    Int nkids = 0;
    for (const auto& n : lvl.nodes)
      nkids += n->nkids;

    std::vector<RankNode> me(lvl.nodes.size()), kids(nkids);
    for (size_t i = 0, mi = 0, ki = 0; i < lvl.nodes.size(); ++i) {
      const auto& n = lvl.nodes[i];
      me[mi].first = n->parent ? n->parent->rank : my_rank;
      me[mi].second = const_cast<NodeSets::Node*>(n);
      ++mi;
      for (Int k = 0; k < n->nkids; ++k) {
        kids[ki].first = n->kids[k]->rank;
        kids[ki].second = const_cast<NodeSets::Node*>(n->kids[k]);
        ++ki;
      }
    }

    init_offsets(my_rank, me, lvl.me, ns.nslots);
    lvl.me_req.resize(lvl.me.size());
    init_offsets(my_rank, kids, lvl.kids, ns.nslots);
    lvl.kids_req.resize(lvl.kids.size());
  }
}

// Analyze the tree to extract levels. Levels are run from 0 to #level - 1. Each
// level has nodes whose corresponding operations depend on only nodes in
// lower-indexed levels. This mechanism prevents deadlock in the general case of
// multiple cells per rank, with multiple ranks appearing in a subtree other
// than the root.
//   In addition, the set of nodes collected into levels are just those owned by
// this rank, and those with which owned nodes must communicate.
//   Once this function is done, the tree can be deleted.
NodeSets::ConstPtr analyze (const Parallel::Ptr& p, const Int& ncells,
                            const tree::Node::Ptr& tree) {
  const auto nodesets = std::make_shared<NodeSets>();
  cedr_assert( ! tree->parent);
  Int id = ncells;
  const Int depth = init_tree(tree, id);
  nodesets->levels.resize(depth);
  level_schedule_and_collect(*nodesets, p->rank(), tree);
  consolidate(*nodesets);
  init_comm(p->rank(), *nodesets);
  return nodesets;
}

// Check that the offsets are self consistent.
Int check_comm (const NodeSets::ConstPtr& ns) {
  Int nerr = 0;
  std::vector<Int> offsets(ns->nslots, 0);
  for (const auto& lvl : ns->levels)
    for (const auto& n : lvl.nodes) {
      cedr_assert(n->offset < ns->nslots);
      ++offsets[n->offset];
      for (Int i = 0; i < n->nkids; ++i)
        if (n->kids[i]->rank != n->rank)
          ++offsets[n->kids[i]->offset];
    }
  for (const auto& e : offsets)
    if (e != 1) ++nerr;
  return nerr;
}

// Check that there are the correct number of leaf nodes, and that their offsets
// all come first and are ordered the same as ns->levels[0]->nodes.
Int check_leaf_nodes (const Parallel::Ptr& p, const NodeSets::ConstPtr& ns,
                      const Int ncells) {
  Int nerr = 0;
  cedr_assert( ! ns->levels.empty());
  cedr_assert( ! ns->levels[0].nodes.empty());
  Int my_nleaves = 0;
  for (const auto& n : ns->levels[0].nodes) {
    cedr_assert( ! n->nkids);
    ++my_nleaves;
  }
  for (const auto& n : ns->levels[0].nodes) {
    cedr_assert(n->offset < my_nleaves);
    cedr_assert(n->id < ncells);
  }
  Int glbl_nleaves = 0;
  mpi::all_reduce(*p, &my_nleaves, &glbl_nleaves, 1, MPI_SUM);
  if (glbl_nleaves != ncells)
    ++nerr;
  return nerr;
}

// Sum cellidx using the QLT comm pattern.
Int test_comm_pattern (const Parallel::Ptr& p, const NodeSets::ConstPtr& ns,
                       const Int ncells) {
  Int nerr = 0;
  // Rank-wide data buffer.
  std::vector<Int> data(ns->nslots);
  // Sum this rank's cellidxs.
  for (auto& n : ns->levels[0].nodes)
    data[n->offset] = n->id;
  // Leaves to root.
  for (size_t il = 0; il < ns->levels.size(); ++il) {
    auto& lvl = ns->levels[il];
    // Set up receives.
    for (size_t i = 0; i < lvl.kids.size(); ++i) {
      const auto& mmd = lvl.kids[i];
      mpi::irecv(*p, &data[mmd.offset], mmd.size, mmd.rank, NodeSets::mpitag,
                 &lvl.kids_req[i]);
    }
    //todo Replace with simultaneous waitany and isend.
    mpi::waitall(lvl.kids_req.size(), lvl.kids_req.data());
    // Combine kids' data.
    for (auto& n : lvl.nodes) {
      if ( ! n->nkids) continue;
      data[n->offset] = 0;
      for (Int i = 0; i < n->nkids; ++i)
        data[n->offset] += data[n->kids[i]->offset];
    }
    // Send to parents.
    for (size_t i = 0; i < lvl.me.size(); ++i) {
      const auto& mmd = lvl.me[i];
      mpi::isend(*p, &data[mmd.offset], mmd.size, mmd.rank, NodeSets::mpitag,
                 &lvl.me_req[i]);
    }
    if (il+1 == ns->levels.size())
      mpi::waitall(lvl.me_req.size(), lvl.me_req.data());
  }
  // Root to leaves.
  for (size_t il = ns->levels.size(); il > 0; --il) {
    auto& lvl = ns->levels[il-1];
    // Get the global sum from parent.
    for (size_t i = 0; i < lvl.me.size(); ++i) {
      const auto& mmd = lvl.me[i];
      mpi::irecv(*p, &data[mmd.offset], mmd.size, mmd.rank, NodeSets::mpitag,
                 &lvl.me_req[i]);
    }    
    //todo Replace with simultaneous waitany and isend.
    mpi::waitall(lvl.me_req.size(), lvl.me_req.data());
    // Pass to kids.
    for (auto& n : lvl.nodes) {
      if ( ! n->nkids) continue;
      for (Int i = 0; i < n->nkids; ++i)
        data[n->kids[i]->offset] = data[n->offset];
    }
    // Send.
    for (size_t i = 0; i < lvl.kids.size(); ++i) {
      const auto& mmd = lvl.kids[i];
      mpi::isend(*p, &data[mmd.offset], mmd.size, mmd.rank, NodeSets::mpitag,
                 &lvl.kids_req[i]);
    }
  }
  // Wait on sends to clean up.
  for (size_t il = 0; il < ns->levels.size(); ++il) {
    auto& lvl = ns->levels[il];
    if (il+1 < ns->levels.size())
      mpi::waitall(lvl.me_req.size(), lvl.me_req.data());
    mpi::waitall(lvl.kids_req.size(), lvl.kids_req.data());
  }
  { // Check that all leaf nodes have the right number.
    const Int desired_sum = (ncells*(ncells - 1)) / 2;
    for (const auto& n : ns->levels[0].nodes)
      if (data[n->offset] != desired_sum) ++nerr;
    if (p->amroot()) {
      std::cout << " " << data[ns->levels[0].nodes[0]->offset];
      std::cout.flush();
    }
  }
  return nerr;
}

// Unit tests for NodeSets.
Int unittest (const Parallel::Ptr& p, const NodeSets::ConstPtr& ns,
              const Int ncells) {
  Int nerr = 0;
  nerr += check_comm(ns);
  if (nerr) return nerr;
  nerr += check_leaf_nodes(p, ns, ncells);
  if (nerr) return nerr;
  nerr += test_comm_pattern(p, ns, ncells);
  if (nerr) return nerr;
  return nerr;
}
} // namespace impl

template <typename ES>
void QLT<ES>::init (const std::string& name, IntList& d,
                    typename IntList::HostMirror& h, size_t n) {
  d = IntList(name, n);
  h = Kokkos::create_mirror_view(d);
}

template <typename ES>
int QLT<ES>::MetaData::get_problem_type (const int& idx) {
  return problem_type_[idx];
}
    
// icpc doesn't let us use problem_type_ here, even though it's constexpr.
template <typename ES>
int QLT<ES>::MetaData::get_problem_type_idx (const int& mask) {
  switch (mask) {
  case CPT::s:  case CPT::st:  return 0;
  case CPT::cs: case CPT::cst: return 1;
  case CPT::t:  return 2;
  case CPT::ct: return 3;
  default: cedr_kernel_throw_if(true, "Invalid problem type."); return -1;
  }
}

template <typename ES>
int QLT<ES>::MetaData::get_problem_type_l2r_bulk_size (const int& mask) {
  if (mask & ProblemType::conserve) return 4;
  return 3;
}

template <typename ES>
int QLT<ES>::MetaData::get_problem_type_r2l_bulk_size (const int& mask) {
  if (mask & ProblemType::shapepreserve) return 1;
  return 3;
}

template <typename ES>
void QLT<ES>::MetaData::init (const MetaDataBuilder& mdb) {
  const Int ntracers = mdb.trcr2prob.size();

  Me::init("trcr2prob", a_d_.trcr2prob, a_h_.trcr2prob, ntracers);
  std::copy(mdb.trcr2prob.begin(), mdb.trcr2prob.end(), a_h_.trcr2prob.data());
  Kokkos::deep_copy(a_d_.trcr2prob, a_h_.trcr2prob);

  Me::init("bidx2trcr", a_d_.bidx2trcr, a_h_.bidx2trcr, ntracers);
  Me::init("trcr2bl2r", a_d_.trcr2bl2r, a_h_.trcr2bl2r, ntracers);
  Me::init("trcr2br2l", a_d_.trcr2br2l, a_h_.trcr2br2l, ntracers);
  a_h_.prob2trcrptr[0] = 0;
  a_h_.prob2bl2r[0] = 1; // rho is at 0.
  a_h_.prob2br2l[0] = 0;
  for (Int pi = 0; pi < nprobtypes; ++pi) {
    a_h_.prob2trcrptr[pi+1] = a_h_.prob2trcrptr[pi];
    const Int l2rbulksz = get_problem_type_l2r_bulk_size(get_problem_type(pi));
    const Int r2lbulksz = get_problem_type_r2l_bulk_size(get_problem_type(pi));
    for (Int ti = 0; ti < ntracers; ++ti) {
      const auto problem_type = a_h_.trcr2prob[ti];
      if (problem_type != problem_type_[pi]) continue;
      const auto tcnt = a_h_.prob2trcrptr[pi+1] - a_h_.prob2trcrptr[pi];
      a_h_.trcr2bl2r[ti] = a_h_.prob2bl2r[pi] + tcnt*l2rbulksz;
      a_h_.trcr2br2l[ti] = a_h_.prob2br2l[pi] + tcnt*r2lbulksz;
      a_h_.bidx2trcr[a_h_.prob2trcrptr[pi+1]++] = ti;
    }
    Int ni = a_h_.prob2trcrptr[pi+1] - a_h_.prob2trcrptr[pi];
    a_h_.prob2bl2r[pi+1] = a_h_.prob2bl2r[pi] + ni*l2rbulksz;
    a_h_.prob2br2l[pi+1] = a_h_.prob2br2l[pi] + ni*r2lbulksz;
  }
  Kokkos::deep_copy(a_d_.bidx2trcr, a_h_.bidx2trcr);
  Kokkos::deep_copy(a_d_.trcr2bl2r, a_h_.trcr2bl2r);
  Kokkos::deep_copy(a_d_.trcr2br2l, a_h_.trcr2br2l);

  Me::init("trcr2bidx", a_d_.trcr2bidx, a_h_.trcr2bidx, ntracers);
  for (Int ti = 0; ti < ntracers; ++ti)
    a_h_.trcr2bidx(a_h_.bidx2trcr(ti)) = ti;
  Kokkos::deep_copy(a_d_.trcr2bidx, a_h_.trcr2bidx);
            
  a_h = a_h_;

  // Won't default construct Unmanaged, so have to do pointer stuff and raw
  // array copy explicitly.
  a_d.trcr2prob = a_d_.trcr2prob;
  a_d.bidx2trcr = a_d_.bidx2trcr;
  a_d.trcr2bidx = a_d_.trcr2bidx;
  a_d.trcr2bl2r = a_d_.trcr2bl2r;
  a_d.trcr2br2l = a_d_.trcr2br2l;
  std::copy(a_h_.prob2trcrptr, a_h_.prob2trcrptr + nprobtypes + 1,
            a_d.prob2trcrptr);
  std::copy(a_h_.prob2bl2r, a_h_.prob2bl2r + nprobtypes + 1, a_d.prob2bl2r);
  std::copy(a_h_.prob2br2l, a_h_.prob2br2l + nprobtypes + 1, a_d.prob2br2l);
  cedr_assert(a_d.prob2trcrptr[nprobtypes] == ntracers);
}

template <typename ES>
void QLT<ES>::BulkData::init (const MetaData& md, const Int& nslots) {
  l2r_data_ = RealList("l2r_data", md.a_h.prob2bl2r[md.nprobtypes]*nslots);
  r2l_data_ = RealList("r2l_data", md.a_h.prob2br2l[md.nprobtypes]*nslots);
  l2r_data = l2r_data_;
  r2l_data = r2l_data_;
}

template <typename ES>
void QLT<ES>::init (const Parallel::Ptr& p, const Int& ncells,
                    const tree::Node::Ptr& tree) {
  p_ = p;
  Timer::start(Timer::analyze);
  ns_ = impl::analyze(p, ncells, tree);
  init_ordinals();
  Timer::stop(Timer::analyze);
  mdb_ = std::make_shared<MetaDataBuilder>();
}

template <typename ES>
void QLT<ES>::init_ordinals () {
  for (const auto& n : ns_->levels[0].nodes)
    gci2lci_[n->id] = n->offset;
}

template <typename ES>
QLT<ES>::QLT (const Parallel::Ptr& p, const Int& ncells, const tree::Node::Ptr& tree) {
  init(p, ncells, tree);
}

template <typename ES>
void QLT<ES>::print (std::ostream& os) const {
  ns_->print(os);
}

// Number of cells owned by this rank.
template <typename ES>
Int QLT<ES>::nlclcells () const { return ns_->levels[0].nodes.size(); }

// Cells owned by this rank, in order of local numbering. Thus,
// gci2lci(gcis[i]) == i. Ideally, the caller never actually calls gci2lci(),
// and instead uses the information from get_owned_glblcells to determine
// local cell indices.
template <typename ES>
void QLT<ES>::get_owned_glblcells (std::vector<Int>& gcis) const {
  gcis.resize(ns_->levels[0].nodes.size());
  for (const auto& n : ns_->levels[0].nodes)
    gcis[n->offset] = n->id;
}

// For global cell index cellidx, i.e., the globally unique ordinal associated
// with a cell in the caller's tree, return this rank's local index for
// it. This is not an efficient operation.
template <typename ES>
Int QLT<ES>::gci2lci (const Int& gci) const {
  const auto it = gci2lci_.find(gci);
  if (it == gci2lci_.end()) {
    pr(puf(gci));
    std::vector<Int> gcis;
    get_owned_glblcells(gcis);
    mprarr(gcis);
  }
  cedr_throw_if(it == gci2lci_.end(), "gci " << gci << " not in gci2lci map.");
  return it->second;
}

// Set up QLT tracer metadata. Once end_tracer_declarations is called, it is
// an error to call declare_tracer again. Call declare_tracer in order of the
// tracer index in the caller's numbering.
template <typename ES>
void QLT<ES>::declare_tracer (int problem_type) {
  cedr_throw_if( ! mdb_, "end_tracer_declarations was already called; "
                "it is an error to call declare_tracer now.");
  // For its exception side effect, and to get canonical problem type, since
  // some possible problem types map to the same canonical one:
  problem_type = md_.get_problem_type(md_.get_problem_type_idx(problem_type));
  mdb_->trcr2prob.push_back(problem_type);
}

template <typename ES>
void QLT<ES>::end_tracer_declarations () {
  md_.init(*mdb_);
  mdb_ = nullptr;
  bd_.init(md_, ns_->nslots);
}

template <typename ES>
int QLT<ES>::get_problem_type (const Int& tracer_idx) const {
  cedr_throw_if(tracer_idx < 0 || tracer_idx > md_.a_h.trcr2prob.extent_int(0),
                "tracer_idx is out of bounds: " << tracer_idx);
  return md_.a_h.trcr2prob[tracer_idx];
}

template <typename ES>
Int QLT<ES>::get_num_tracers () const {
  return md_.a_h.trcr2prob.size();
}

template <typename ES>
void QLT<ES>::run () {
  Timer::start(Timer::qltrunl2r);
  using namespace impl;
  // Number of data per slot.
  const Int l2rndps = md_.a_d.prob2bl2r[md_.nprobtypes];
  const Int r2lndps = md_.a_d.prob2br2l[md_.nprobtypes];
  // Leaves to root.
  for (size_t il = 0; il < ns_->levels.size(); ++il) {
    auto& lvl = ns_->levels[il];
    // Set up receives.
    for (size_t i = 0; i < lvl.kids.size(); ++i) {
      const auto& mmd = lvl.kids[i];
      mpi::irecv(*p_, &bd_.l2r_data(mmd.offset*l2rndps), mmd.size*l2rndps, mmd.rank,
                 NodeSets::mpitag, &lvl.kids_req[i]);
    }
    //todo Replace with simultaneous waitany and isend.
    Timer::start(Timer::waitall);
    mpi::waitall(lvl.kids_req.size(), lvl.kids_req.data());
    Timer::stop(Timer::waitall);
    // Combine kids' data.
    //todo Kernelize, interacting with waitany todo above.
    for (const auto& n : lvl.nodes) {
      if ( ! n->nkids) continue;
      cedr_kernel_assert(n->nkids == 2);
      // Total density.
      bd_.l2r_data(n->offset*l2rndps) = (bd_.l2r_data(n->kids[0]->offset*l2rndps) +
                                         bd_.l2r_data(n->kids[1]->offset*l2rndps));
      // Tracers.
      for (Int pti = 0; pti < md_.nprobtypes; ++pti) {
        const Int problem_type = md_.get_problem_type(pti);
        const bool sum_only = problem_type & ProblemType::shapepreserve;
        const Int bsz = md_.get_problem_type_l2r_bulk_size(problem_type);
        const Int bis = md_.a_d.prob2trcrptr[pti], bie = md_.a_d.prob2trcrptr[pti+1];
        for (Int bi = bis; bi < bie; ++bi) {
          const Int bdi = md_.a_d.trcr2bl2r(md_.a_d.bidx2trcr(bi));
          Real* const me = &bd_.l2r_data(n->offset*l2rndps + bdi);
          const Real* const k0 = &bd_.l2r_data(n->kids[0]->offset*l2rndps + bdi);
          const Real* const k1 = &bd_.l2r_data(n->kids[1]->offset*l2rndps + bdi);
          me[0] = sum_only ? k0[0] + k1[0] : cedr::impl::min(k0[0], k1[0]);
          me[1] =            k0[1] + k1[1] ;
          me[2] = sum_only ? k0[2] + k1[2] : cedr::impl::max(k0[2], k1[2]);
          if (bsz == 4)
            me[3] =          k0[3] + k1[3] ;
        }
      }
    }
    // Send to parents.
    for (size_t i = 0; i < lvl.me.size(); ++i) {
      const auto& mmd = lvl.me[i];
      mpi::isend(*p_, &bd_.l2r_data(mmd.offset*l2rndps), mmd.size*l2rndps, mmd.rank,
                 NodeSets::mpitag, &lvl.me_req[i]);
    }
    if (il+1 == ns_->levels.size()) {
      Timer::start(Timer::waitall);
      mpi::waitall(lvl.me_req.size(), lvl.me_req.data());
      Timer::stop(Timer::waitall);
    }
  }
  Timer::stop(Timer::qltrunl2r); Timer::start(Timer::qltrunr2l);
  // Root.
  if ( ! ns_->levels.empty() && ns_->levels.back().nodes.size() == 1 &&
       ! ns_->levels.back().nodes[0]->parent) {
    const auto& n = ns_->levels.back().nodes[0];
    for (Int pti = 0; pti < md_.nprobtypes; ++pti) {
      const Int problem_type = md_.get_problem_type(pti);
      const Int bis = md_.a_d.prob2trcrptr[pti], bie = md_.a_d.prob2trcrptr[pti+1];
      for (Int bi = bis; bi < bie; ++bi) {
        const Int l2rbdi = md_.a_d.trcr2bl2r(md_.a_d.bidx2trcr(bi));
        const Int r2lbdi = md_.a_d.trcr2br2l(md_.a_d.bidx2trcr(bi));
        // If QLT is enforcing global mass conservation, set the root's r2l Qm
        // value to the l2r Qm_prev's sum; otherwise, copy the l2r Qm value to
        // the r2l one.
        const Int os = problem_type & ProblemType::conserve ? 3 : 1;
        bd_.r2l_data(n->offset*r2lndps + r2lbdi) =
          bd_.l2r_data(n->offset*l2rndps + l2rbdi + os);
        if ( ! (problem_type & ProblemType::shapepreserve)) {
          // We now know the global q_{min,max}. Start propagating it
          // leafward.
          bd_.r2l_data(n->offset*r2lndps + r2lbdi + 1) =
            bd_.l2r_data(n->offset*l2rndps + l2rbdi + 0);
          bd_.r2l_data(n->offset*r2lndps + r2lbdi + 2) =
            bd_.l2r_data(n->offset*l2rndps + l2rbdi + 2);
        }
      }
    }
  }
  // Root to leaves.
  for (size_t il = ns_->levels.size(); il > 0; --il) {
    auto& lvl = ns_->levels[il-1];
    for (size_t i = 0; i < lvl.me.size(); ++i) {
      const auto& mmd = lvl.me[i];
      mpi::irecv(*p_, &bd_.r2l_data(mmd.offset*r2lndps), mmd.size*r2lndps, mmd.rank,
                 NodeSets::mpitag, &lvl.me_req[i]);
    }
    //todo Replace with simultaneous waitany and isend.
    Timer::start(Timer::waitall);
    mpi::waitall(lvl.me_req.size(), lvl.me_req.data());
    Timer::stop(Timer::waitall);
    // Solve QP for kids' values.
    //todo Kernelize, interacting with waitany todo above.
    Timer::start(Timer::snp);
    for (const auto& n : lvl.nodes) {
      if ( ! n->nkids) continue;
      for (Int pti = 0; pti < md_.nprobtypes; ++pti) {
        const Int problem_type = md_.get_problem_type(pti);
        const Int bis = md_.a_d.prob2trcrptr[pti], bie = md_.a_d.prob2trcrptr[pti+1];
        for (Int bi = bis; bi < bie; ++bi) {
          const Int l2rbdi = md_.a_d.trcr2bl2r(md_.a_d.bidx2trcr(bi));
          const Int r2lbdi = md_.a_d.trcr2br2l(md_.a_d.bidx2trcr(bi));
          cedr_assert(n->nkids == 2);
          if ( ! (problem_type & ProblemType::shapepreserve)) {
            // Pass q_{min,max} info along. l2r data are updated for use in
            // solve_node_problem. r2l data are updated for use in isend.
            const Real q_min = bd_.r2l_data(n->offset*r2lndps + r2lbdi + 1);
            const Real q_max = bd_.r2l_data(n->offset*r2lndps + r2lbdi + 2);
            bd_.l2r_data(n->offset*l2rndps + l2rbdi + 0) = q_min;
            bd_.l2r_data(n->offset*l2rndps + l2rbdi + 2) = q_max;
            for (Int k = 0; k < 2; ++k) {
              bd_.l2r_data(n->kids[k]->offset*l2rndps + l2rbdi + 0) = q_min;
              bd_.l2r_data(n->kids[k]->offset*l2rndps + l2rbdi + 2) = q_max;
              bd_.r2l_data(n->kids[k]->offset*r2lndps + r2lbdi + 1) = q_min;
              bd_.r2l_data(n->kids[k]->offset*r2lndps + r2lbdi + 2) = q_max;
            }
          }
          const auto& k0 = n->kids[0];
          const auto& k1 = n->kids[1];
          solve_node_problem(
            problem_type,
             bd_.l2r_data( n->offset*l2rndps),
            &bd_.l2r_data( n->offset*l2rndps + l2rbdi),
             bd_.r2l_data( n->offset*r2lndps + r2lbdi),
             bd_.l2r_data(k0->offset*l2rndps),
            &bd_.l2r_data(k0->offset*l2rndps + l2rbdi),
             bd_.r2l_data(k0->offset*r2lndps + r2lbdi),
             bd_.l2r_data(k1->offset*l2rndps),
            &bd_.l2r_data(k1->offset*l2rndps + l2rbdi),
             bd_.r2l_data(k1->offset*r2lndps + r2lbdi));
        }
      }
    }
    Timer::stop(Timer::snp);
    // Send.
    for (size_t i = 0; i < lvl.kids.size(); ++i) {
      const auto& mmd = lvl.kids[i];
      mpi::isend(*p_, &bd_.r2l_data(mmd.offset*r2lndps), mmd.size*r2lndps, mmd.rank,
                 NodeSets::mpitag, &lvl.kids_req[i]);
    }
  }
  // Wait on sends to clean up.
  for (size_t il = 0; il < ns_->levels.size(); ++il) {
    auto& lvl = ns_->levels[il];
    if (il+1 < ns_->levels.size())
      mpi::waitall(lvl.me_req.size(), lvl.me_req.data());
    mpi::waitall(lvl.kids_req.size(), lvl.kids_req.data());
  }
  Timer::stop(Timer::qltrunr2l);
}

template <typename ES>
constexpr Int QLT<ES>::MetaData::problem_type_[];

namespace test {
using namespace impl;

class TestQLT {
  typedef QLT<Kokkos::DefaultExecutionSpace> QLTT;
  typedef Kokkos::View<Real**, QLTT::Device> R2D;

  struct Tracer {
    typedef QLTT::ProblemType PT;
    
    Int idx;
    Int problem_type;
    Int perturbation_type;
    bool no_change_should_hold, safe_should_hold, local_should_hold;
    bool write;

    std::string str () const {
      std::stringstream ss;
      ss << "(ti " << idx;
      if (problem_type & PT::conserve) ss << " c";
      if (problem_type & PT::shapepreserve) ss << " s";
      if (problem_type & PT::consistent) ss << " t";
      ss << " pt " << perturbation_type << " ssh " << safe_should_hold
         << " lsh " << local_should_hold << ")";
      return ss.str();
    }

    Tracer ()
      : idx(-1), problem_type(-1), perturbation_type(-1), no_change_should_hold(false),
        safe_should_hold(true), local_should_hold(true), write(false)
    {}
  };

  struct Values {
    Values (const Int ntracers, const Int ncells)
      : ncells_(ncells), v_((4*ntracers + 1)*ncells)
    {}
    Int ncells () const { return ncells_; }
    Real* rhom () { return v_.data(); }
    Real* Qm_min  (const Int& ti) { return v_.data() + ncells_*(1 + 4*ti    ); }
    Real* Qm      (const Int& ti) { return v_.data() + ncells_*(1 + 4*ti + 1); }
    Real* Qm_max  (const Int& ti) { return v_.data() + ncells_*(1 + 4*ti + 2); }
    Real* Qm_prev (const Int& ti) { return v_.data() + ncells_*(1 + 4*ti + 3); }
    const Real* rhom () const { return const_cast<Values*>(this)->rhom(); }
    const Real* Qm_min  (const Int& ti) const
    { return const_cast<Values*>(this)->Qm_min (ti); }
    const Real* Qm      (const Int& ti) const
    { return const_cast<Values*>(this)->Qm     (ti); }
    const Real* Qm_max  (const Int& ti) const
    { return const_cast<Values*>(this)->Qm_max (ti); }
    const Real* Qm_prev (const Int& ti) const
    { return const_cast<Values*>(this)->Qm_prev(ti); }
  private:
    Int ncells_;
    std::vector<Real> v_;
  };

  // For solution output, if requested.
  struct Writer {
    std::unique_ptr<FILE, cedr::util::FILECloser> fh;
    std::vector<Int> ngcis;  // Number of i'th rank's gcis_ array.
    std::vector<int> displs; // Cumsum of above.
    std::vector<Int> gcis;   // Global cell indices packed by rank's gcis_ vector.
    ~Writer () {
      if ( ! fh) return;
      fprintf(fh.get(), "  return s\n");
    }
  };

private:
  const Parallel::Ptr p_;
  const Int ncells_;
  QLTT qlt_;
  // Caller index (local cell index in the app code) -> QLT lclcellidx.
  std::vector<Int> gcis_, i2lci_;
  std::vector<Tracer> tracers_;
  // For optional output.
  bool write_inited_;
  std::shared_ptr<Writer> w_; // Only on root.

private:
  void init_numbering (const tree::Node::Ptr& node) {
    // TestQLT doesn't actually care about a particular ordering, as there is no
    // geometry to the test problem. However, use *some* ordering to model what
    // a real problem must do.
    if ( ! node->nkids) {
      if (node->rank == p_->rank()) {
        gcis_.push_back(node->cellidx);
        i2lci_.push_back(qlt_.gci2lci(gcis_.back()));
      }
      return;
    }
    for (Int i = 0; i < node->nkids; ++i)
      init_numbering(node->kids[i]);
  }

  void init_tracers () {
    Timer::start(Timer::trcrinit);
    typedef Tracer::PT PT;
    static const Int pts[] = {
      PT::conserve | PT::shapepreserve | PT::consistent,
      PT::shapepreserve, // Test a noncanonical problem type.
      PT::conserve | PT::consistent,
      PT::consistent
    };
    Int tracer_idx = 0;
    for (Int perturb = 0; perturb < 6; ++perturb)
      for (Int ti = 0; ti < 4; ++ti) {
        Tracer t;
        t.problem_type = pts[ti];
        const bool shapepreserve = t.problem_type & PT::shapepreserve;
        t.idx = tracer_idx++;
        t.perturbation_type = perturb;
        t.safe_should_hold = true;
        t.no_change_should_hold = perturb == 0;
        t.local_should_hold = perturb < 4 && shapepreserve;
        t.write = perturb == 2 && ti == 2;
        tracers_.push_back(t);
        qlt_.declare_tracer(t.problem_type);
      }
    qlt_.end_tracer_declarations();
    cedr_assert(qlt_.get_num_tracers() == static_cast<Int>(tracers_.size()));
    for (size_t i = 0; i < tracers_.size(); ++i)
      cedr_assert(qlt_.get_problem_type(i) == (tracers_[i].problem_type |
                                               PT::consistent));
    Timer::stop(Timer::trcrinit);
  }

  static Real urand () { return rand() / ((Real) RAND_MAX + 1.0); }

  static void generate_rho (Values& v) {
    auto r = v.rhom();
    const Int n = v.ncells();
    for (Int i = 0; i < n; ++i)
      r[i] = 0.5 + 1.5*urand();
  }

  static void generate_Q (const Tracer& t, Values& v) {
    Real* rhom = v.rhom(), * Qm_min = v.Qm_min(t.idx), * Qm = v.Qm(t.idx),
      * Qm_max = v.Qm_max(t.idx), * Qm_prev = v.Qm_prev(t.idx);
    const Int n = v.ncells();
    for (Int i = 0; i < n; ++i) {
      const Real
        q_min = 0.1 + 0.8*urand(),
        q_max = std::min<Real>(1, q_min + (0.9 - q_min)*urand()),
        q = q_min + (q_max - q_min)*urand();
      // Check correctness up to FP.
      cedr_assert(q_min >= 0 &&
                  q_max <= 1 + 10*std::numeric_limits<Real>::epsilon() &&
                  q_min <= q && q <= q_max);
      Qm_min[i] = q_min*rhom[i];
      Qm_max[i] = q_max*rhom[i];
      // Protect against FP error.
      Qm[i] = std::max<Real>(Qm_min[i], std::min(Qm_max[i], q*rhom[i]));
      // Set previous Qm to the current unperturbed value.
      Qm_prev[i] = Qm[i];
    }
  }

  static void gen_rand_perm (const size_t n, std::vector<Int>& p) {
    p.resize(n);
    for (size_t i = 0; i < n; ++i)
      p[i] = i;
    for (size_t i = 0; i < n; ++i) {
      const int j = urand()*n, k = urand()*n;
      std::swap(p[j], p[k]);
    }
  }

  // Permuting the Qm array, even just on a rank as long as there is > 1 cell,
  // produces a problem likely requiring considerable reconstruction, which
  // reconstruction assuredly satisfies the properties. But because this is a
  // local operation only, it doesn't test the 1 cell/rank case.
  static void permute_Q (const Tracer& t, Values& v) {
    Real* const Qm = v.Qm(t.idx);
    const Int N = v.ncells();
    std::vector<Int> p;
    gen_rand_perm(N, p);
    std::vector<Real> Qm_orig(N);
    std::copy(Qm, Qm + N, Qm_orig.begin());
    for (Int i = 0; i < N; ++i)
      Qm[i] = Qm_orig[p[i]];
  }

  void add_const_to_Q (const Tracer& t, Values& v,
                       // Move 0 < alpha <= 1 of the way to the QLT or safety
                       // feasibility bound.
                       const Real& alpha,
                       // Whether the modification should be done in a
                       // mass-conserving way.
                       const bool conserve_mass,
                       // Only safety problem is feasible.
                       const bool safety_problem) {
    // Some of these reductions aren't used at present. Might add more test
    // options later that use them.
    Real rhom, Qm, Qm_max; {
      Real Qm_sum_lcl[3] = {0};
      for (Int i = 0; i < v.ncells(); ++i) {
        Qm_sum_lcl[0] += v.rhom()[i];
        Qm_sum_lcl[1] += v.Qm(t.idx)[i];
        Qm_sum_lcl[2] += v.Qm_max(t.idx)[i];
      }
      Real Qm_sum_gbl[3] = {0};
      mpi::all_reduce(*p_, Qm_sum_lcl, Qm_sum_gbl, 3, MPI_SUM);
      rhom = Qm_sum_gbl[0]; Qm = Qm_sum_gbl[1]; Qm_max = Qm_sum_gbl[2];
    }
    Real Qm_max_safety = 0;
    if (safety_problem) {
      Real q_safety_lcl = v.Qm_max(t.idx)[0] / v.rhom()[0];
      for (Int i = 1; i < v.ncells(); ++i)
        q_safety_lcl = std::max(q_safety_lcl, v.Qm_max(t.idx)[i] / v.rhom()[i]);
      Real q_safety_gbl = 0;
      mpi::all_reduce(*p_, &q_safety_lcl, &q_safety_gbl, 1, MPI_MAX);
      Qm_max_safety = q_safety_gbl*rhom;
    }
    const Real dQm = safety_problem ?
      ((Qm_max - Qm) + alpha * (Qm_max_safety - Qm_max)) / ncells_ :
      alpha * (Qm_max - Qm) / ncells_;
    for (Int i = 0; i < v.ncells(); ++i)
      v.Qm(t.idx)[i] += dQm;
    // Now permute Qm so that it's a little more interesting.
    permute_Q(t, v);
    // Adjust Qm_prev. Qm_prev is used to test the PT::conserve case, and also
    // simply to record the correct total mass. The modification above modified
    // Q's total mass. If conserve_mass, then Qm_prev needs to be made to sum to
    // the same new mass. If ! conserve_mass, we want Qm_prev to be modified in
    // an interesting way, so that PT::conserve doesn't trivially undo the mod
    // that was made above when the root fixes the mass discrepancy.
    const Real
      relax = 0.9,
      dQm_prev = (conserve_mass ? dQm :
                  (safety_problem ?
                   ((Qm_max - Qm) + relax*alpha * (Qm_max_safety - Qm_max)) / ncells_ :
                   relax*alpha * (Qm_max - Qm) / ncells_));
    for (Int i = 0; i < v.ncells(); ++i)
      v.Qm_prev(t.idx)[i] += dQm_prev;
  }

  void perturb_Q (const Tracer& t, Values& v) {
    // QLT is naturally mass conserving. But if QLT isn't being asked to impose
    // mass conservation, then the caller better have a conservative
    // method. Here, we model that by saying that Qm_prev and Qm should sum to
    // the same mass.
    const bool cm = ! (t.problem_type & Tracer::PT::conserve);
    // For the edge cases, we cannot be exactly on the edge and still expect the
    // q-limit checks to pass to machine precision. Thus, back away from the
    // edge by an amount that bounds the error in the global mass due to FP,
    // assuming each cell's mass is O(1).
    const Real edg = 1 - ncells_*std::numeric_limits<Real>::epsilon();
    switch (t.perturbation_type) {
    case 0:
      // Do nothing, to test that QLT doesn't make any changes if none is
      // needed.
      break;
    case 1: permute_Q(t, v); break;
    case 2: add_const_to_Q(t, v, 0.5, cm, false); break;
    case 3: add_const_to_Q(t, v, edg, cm, false); break;
    case 4: add_const_to_Q(t, v, 0.5, cm, true ); break;
    case 5: add_const_to_Q(t, v, edg, cm, true ); break;
    }
  }

  static std::string get_tracer_name (const Tracer& t) {
    std::stringstream ss;
    ss << "t" << t.idx;
    return ss.str();
  }

  void init_writer () {
    if (p_->amroot()) {
      w_ = std::make_shared<Writer>();
      w_->fh = std::unique_ptr<FILE, cedr::util::FILECloser>(fopen("out_QLT.py", "w"));
      int n = gcis_.size();
      w_->ngcis.resize(p_->size());
      mpi::gather(*p_, &n, 1, w_->ngcis.data(), 1, p_->root());
      w_->displs.resize(p_->size() + 1);
      w_->displs[0] = 0;
      for (size_t i = 0; i < w_->ngcis.size(); ++i)
        w_->displs[i+1] = w_->displs[i] + w_->ngcis[i];
      cedr_assert(w_->displs.back() == ncells_);
      w_->gcis.resize(ncells_);
      mpi::gatherv(*p_, gcis_.data(), gcis_.size(), w_->gcis.data(), w_->ngcis.data(),
                   w_->displs.data(), p_->root());
    } else {
      int n = gcis_.size();
      mpi::gather(*p_, &n, 1, static_cast<int*>(nullptr), 0, p_->root());
      Int* Inull = nullptr;
      const int* inull = nullptr;
      mpi::gatherv(*p_, gcis_.data(), gcis_.size(), Inull, inull, inull, p_->root());
    }
    write_inited_ = true;
  }

  void gather_field (const Real* Qm_lcl, std::vector<Real>& Qm_gbl,
                     std::vector<Real>& wrk) {
    if (p_->amroot()) {
      Qm_gbl.resize(ncells_);
      wrk.resize(ncells_);
      mpi::gatherv(*p_, Qm_lcl, gcis_.size(), wrk.data(), w_->ngcis.data(),
                   w_->displs.data(), p_->root());
      for (Int i = 0; i < ncells_; ++i)
        Qm_gbl[w_->gcis[i]] = wrk[i];
    } else {
      Real* rnull = nullptr;
      const int* inull = nullptr;
      mpi::gatherv(*p_, Qm_lcl, gcis_.size(), rnull, inull, inull, p_->root());
    }
  }

  void write_field (const std::string& tracer_name, const std::string& field_name,
                    const std::vector<Real>& Qm) {
    if ( ! p_->amroot()) return;
    fprintf(w_->fh.get(), "  s.%s.%s = [", tracer_name.c_str(), field_name.c_str());
    for (const auto& e : Qm)
      fprintf(w_->fh.get(), "%1.15e, ", e);
    fprintf(w_->fh.get(), "]\n");
  }

  void write_pre (const Tracer& t, Values& v) {
    if ( ! t.write) return;
    std::vector<Real> f, wrk;
    if ( ! write_inited_) {
      init_writer();
      if (w_)
        fprintf(w_->fh.get(),
                "def getsolns():\n"
                "  class Struct:\n"
                "    pass\n"
                "  s = Struct()\n"
                "  s.all = Struct()\n");
      gather_field(v.rhom(), f, wrk);
      write_field("all", "rhom", f);
    }
    const auto name = get_tracer_name(t);
    if (w_)
      fprintf(w_->fh.get(), "  s.%s = Struct()\n", name.c_str());
    gather_field(v.Qm_min(t.idx), f, wrk);
    write_field(name, "Qm_min", f);
    gather_field(v.Qm_prev(t.idx), f, wrk);
    write_field(name, "Qm_orig", f);
    gather_field(v.Qm(t.idx), f, wrk);
    write_field(name, "Qm_pre", f);
    gather_field(v.Qm_max(t.idx), f, wrk);
    write_field(name, "Qm_max", f);
  }

  void write_post (const Tracer& t, Values& v) {
    if ( ! t.write) return;
    const auto name = get_tracer_name(t);
    std::vector<Real> Qm, wrk;
    gather_field(v.Qm(t.idx), Qm, wrk);
    write_field(name, "Qm_qlt", Qm);
  }

  static void check (const QLTT& qlt) {
    const Int n = qlt.nlclcells();
    std::vector<Int> gcis;
    qlt.get_owned_glblcells(gcis);
    cedr_assert(static_cast<Int>(gcis.size()) == n);
    for (Int i = 0; i < n; ++i)
      cedr_assert(qlt.gci2lci(gcis[i]) == i);
  }

  static Int check (const Parallel& p, const std::vector<Tracer>& ts, const Values& v) {
    static const bool details = true;
    static const Real ulp3 = 3*std::numeric_limits<Real>::epsilon();
    Int nerr = 0;
    std::vector<Real> lcl_mass(2*ts.size()), q_min_lcl(ts.size()), q_max_lcl(ts.size());
    std::vector<Int> t_ok(ts.size(), 1), local_violated(ts.size(), 0);
    for (size_t ti = 0; ti < ts.size(); ++ti) {
      const auto& t = ts[ti];

      cedr_assert(t.safe_should_hold);
      const bool safe_only = ! t.local_should_hold;
      const Int n = v.ncells();
      const Real* rhom = v.rhom(), * Qm_min = v.Qm_min(t.idx), * Qm = v.Qm(t.idx),
        * Qm_max = v.Qm_max(t.idx), * Qm_prev = v.Qm_prev(t.idx);

      q_min_lcl[ti] = 1;
      q_max_lcl[ti] = 0;
      for (Int i = 0; i < n; ++i) {
        const bool lv = (Qm[i] < Qm_min[i] || Qm[i] > Qm_max[i]);
        if (lv) local_violated[ti] = 1;
        if ( ! safe_only && lv) {
          // If this fails at ~ machine eps, check r2l_nl_adjust_bounds code in
          // solve_node_problem.
          if (details)
            pr("check q " << t.str() << ": " << Qm[i] << " " <<
               (Qm[i] < Qm_min[i] ? Qm[i] - Qm_min[i] : Qm[i] - Qm_max[i]));
          t_ok[ti] = false;
          ++nerr;
        }
        if (t.no_change_should_hold && Qm[i] != Qm_prev[i]) {
          if (details)
            pr("Q should be unchanged but is not: " << Qm_prev[i] << " changed to " <<
               Qm[i] << " in " << t.str());
          t_ok[ti] = false;
          ++nerr;
        }
        lcl_mass[2*ti    ] += Qm_prev[i];
        lcl_mass[2*ti + 1] += Qm[i];
        q_min_lcl[ti] = std::min(q_min_lcl[ti], Qm_min[i]/rhom[i]);
        q_max_lcl[ti] = std::max(q_max_lcl[ti], Qm_max[i]/rhom[i]);
      }
    }

    std::vector<Real> q_min_gbl(ts.size(), 0), q_max_gbl(ts.size(), 0);
    mpi::all_reduce(p, q_min_lcl.data(), q_min_gbl.data(), q_min_lcl.size(), MPI_MIN);
    mpi::all_reduce(p, q_max_lcl.data(), q_max_gbl.data(), q_max_lcl.size(), MPI_MAX);

    for (size_t ti = 0; ti < ts.size(); ++ti) {
      // Check safety problem. If local_should_hold and it does, then the safety
      // problem is by construction also solved (since it's a relaxation of the
      // local problem).
      const auto& t = ts[ti];
      const bool safe_only = ! t.local_should_hold;
      if (safe_only) {
        const Int n = v.ncells();
        const Real* rhom = v.rhom(), * Qm_min = v.Qm_min(t.idx), * Qm = v.Qm(t.idx),
          * Qm_max = v.Qm_max(t.idx);
        const Real q_min = q_min_gbl[ti], q_max = q_max_gbl[ti];
        for (Int i = 0; i < n; ++i) {
          if (Qm[i] < q_min*rhom[i]*(1 - ulp3) ||
              Qm[i] > q_max*rhom[i]*(1 + ulp3)) {
            if (details)
              pr("check q " << t.str() << ": " << q_min*rhom[i] << " " << Qm_min[i] <<
                 " " << Qm[i] << " " << Qm_max[i] << " " << q_max*rhom[i] << " | " <<
                 (Qm[i] < q_min*rhom[i] ?
                  Qm[i] - q_min*rhom[i] :
                  Qm[i] - q_max*rhom[i]));
            t_ok[ti] = false;
            ++nerr;
          }
        }        
      }
    }

    std::vector<Real> glbl_mass(2*ts.size(), 0);
    mpi::reduce(p, lcl_mass.data(), glbl_mass.data(), lcl_mass.size(), MPI_SUM,
                p.root());
    std::vector<Int> t_ok_gbl(ts.size(), 0);
    mpi::reduce(p, t_ok.data(), t_ok_gbl.data(), t_ok.size(), MPI_MIN, p.root());
    // Right now we're not using these:
    std::vector<Int> local_violated_gbl(ts.size(), 0);
    mpi::reduce(p, local_violated.data(), local_violated_gbl.data(),
                local_violated.size(), MPI_MAX, p.root());

    if (p.amroot()) {
      const Real tol = 1e3*std::numeric_limits<Real>::epsilon();
      for (size_t ti = 0; ti < ts.size(); ++ti) {
        // Check mass conservation.
        const Real desired_mass = glbl_mass[2*ti], actual_mass = glbl_mass[2*ti+1],
          rd = cedr::util::reldif(desired_mass, actual_mass);
        const bool mass_failed = rd > tol;
        if (mass_failed) {
          ++nerr;
          t_ok_gbl[ti] = false;
        }
        if ( ! t_ok_gbl[ti]) {
          std::cout << "FAIL " << ts[ti].str();
          if (mass_failed) std::cout << " mass re " << rd;
          std::cout << "\n";
        }
      }
    }

    return nerr;
  }
  
public:
  TestQLT (const Parallel::Ptr& p, const tree::Node::Ptr& tree,
           const Int& ncells, const bool verbose = false)
    : p_(p), ncells_(ncells), qlt_(p_, ncells, tree), write_inited_(false)
  {
    check(qlt_);
    init_numbering(tree);
    init_tracers();
    if (verbose) qlt_.print(std::cout);
  }

  Int run (const Int nrepeat = 1, const bool write=false) {
    Timer::start(Timer::trcrgen);
    const Int nt = qlt_.get_num_tracers(), nlclcells = qlt_.nlclcells();
    Values v(nt, nlclcells);
    generate_rho(v);
    {
      Real* rhom = v.rhom();
      for (Int i = 0; i < nlclcells; ++i)
        qlt_.set_rhom(i2lci_[i], rhom[i]);
    }
    for (Int ti = 0; ti < nt; ++ti) {
      generate_Q(tracers_[ti], v);
      perturb_Q(tracers_[ti], v);
      if (write) write_pre(tracers_[ti], v);
    }
    Timer::stop(Timer::trcrgen);
    for (Int trial = 0; trial <= nrepeat; ++trial) {
      for (Int ti = 0; ti < nt; ++ti) {
        Real* Qm_min = v.Qm_min(ti), * Qm = v.Qm(ti), * Qm_max = v.Qm_max(ti),
          * Qm_prev = v.Qm_prev(ti);
        for (Int i = 0; i < nlclcells; ++i)
          qlt_.set_Qm(i2lci_[i], ti, Qm[i], Qm_min[i], Qm_max[i], Qm_prev[i]);
      }
      MPI_Barrier(p_->comm());
      Timer::start(Timer::qltrun);
      qlt_.run();
      MPI_Barrier(p_->comm());
      Timer::stop(Timer::qltrun);
      if (trial == 0) {
        Timer::reset(Timer::qltrun);
        Timer::reset(Timer::qltrunl2r);
        Timer::reset(Timer::qltrunr2l);
        Timer::reset(Timer::waitall);
        Timer::reset(Timer::snp);
      }
    }
    Timer::start(Timer::trcrcheck);
    Int nerr = 0;
    for (Int ti = 0; ti < nt; ++ti) {
      Real* Qm = v.Qm(ti);
      for (Int i = 0; i < nlclcells; ++i)
        Qm[i] = qlt_.get_Qm(i2lci_[i], ti);
      if (write) write_post(tracers_[ti], v);
    }
    nerr += check(*p_, tracers_, v);
    Timer::stop(Timer::trcrcheck);
    return nerr;
  }
};

// Test all QLT variations and situations.
Int test_qlt (const Parallel::Ptr& p, const tree::Node::Ptr& tree, const Int& ncells,
              const int nrepeat = 1,
              // Diagnostic output for dev and illustration purposes. To be
              // clear, no QLT unit test requires output to be checked; each
              // checks in-memory data and returns a failure count.
              const bool write = false,
              const bool verbose = false) {
  return TestQLT(p, tree, ncells, verbose).run(nrepeat, write);
}
} // namespace test

// Tree for a 1-D periodic domain, for unit testing.
namespace oned {
struct Mesh {
  struct ParallelDecomp {
    enum Enum {
      // The obvious distribution of ranks: 1 rank takes exactly 1 contiguous
      // set of cell indices.
      contiguous,
      // For heavy-duty testing of QLT comm pattern, use a ridiculous assignment
      // of ranks to cell indices. This forces the QLT tree to communicate,
      // pack, and unpack in silly ways.
      pseudorandom
    };
  };
  
  Mesh (const Int nc, const Parallel::Ptr& p,
        const ParallelDecomp::Enum& parallel_decomp = ParallelDecomp::contiguous) {
    init(nc, p, parallel_decomp);
  }
  
  void init (const Int nc, const Parallel::Ptr& p,
             const ParallelDecomp::Enum& parallel_decomp) {
    nc_ = nc;
    nranks_ = p->size();
    p_ = p;
    pd_ = parallel_decomp;
    cedr_assert(nranks_ <= nc_);
  }

  Int ncell () const { return nc_; }

  const Parallel::Ptr& parallel () const { return p_; }

  Int rank (const Int& ci) const {
    switch (pd_) {
    case ParallelDecomp::contiguous:
      return std::min(nranks_ - 1, ci / (nc_ / nranks_));
    default: {
      const auto chunk = ci / nranks_;
      return (ci + chunk) % nranks_;
    }
    }
  }

  static Int unittest (const Parallel::Ptr& p) {
    const Mesh::ParallelDecomp::Enum dists[] = { Mesh::ParallelDecomp::pseudorandom,
                                                 Mesh::ParallelDecomp::contiguous };
    Int ne = 0;
    for (size_t id = 0; id < sizeof(dists)/sizeof(*dists); ++id) {
      Mesh m(std::max(42, 3*p->size()), p, dists[id]);
      const Int nc = m.ncell();
      for (Int ci = 0; ci < nc; ++ci)
        if (m.rank(ci) < 0 || m.rank(ci) >= p->size())
          ++ne;
    }
    return ne;
  }

private:
  Int nc_, nranks_;
  Parallel::Ptr p_;
  ParallelDecomp::Enum pd_;
};

tree::Node::Ptr make_tree (const Mesh& m, const Int cs, const Int ce,
                           const tree::Node* parent, const bool imbalanced) {
  const Int
    cn = ce - cs,
    cn0 = ( imbalanced && cn > 2 ?
            cn/3 :
            cn/2 );
  tree::Node::Ptr n = std::make_shared<tree::Node>();
  n->parent = parent;
  if (cn == 1) {
    n->nkids = 0;
    n->rank = m.rank(cs);
    n->cellidx = cs;
    return n;
  }
  n->nkids = 2;
  n->kids[0] = make_tree(m, cs, cs + cn0, n.get(), imbalanced);
  n->kids[1] = make_tree(m, cs + cn0, ce, n.get(), imbalanced);
  return n;
}

tree::Node::Ptr make_tree (const Mesh& m, const bool imbalanced) {
  return make_tree(m, 0, m.ncell(), nullptr, imbalanced);
}

tree::Node::Ptr make_tree (const Parallel::Ptr& p, const Int& ncells,
                           const bool imbalanced) {
  Mesh m(ncells, p);
  return make_tree(m, imbalanced);
}

namespace test {
void mark_cells (const tree::Node::Ptr& node, std::vector<Int>& cells) {
  if ( ! node->nkids) {
    ++cells[node->cellidx];
    return;
  }
  for (Int i = 0; i < node->nkids; ++i)
    mark_cells(node->kids[i], cells);
}

Int unittest (const Parallel::Ptr& p) {
  const Mesh::ParallelDecomp::Enum dists[] = { Mesh::ParallelDecomp::pseudorandom,
                                               Mesh::ParallelDecomp::contiguous };
  Int ne = 0;
  for (size_t id = 0; id < sizeof(dists)/sizeof(*dists); ++id)
    for (bool imbalanced: {false, true}) {
      Mesh m(std::max(42, 3*p->size()), p, Mesh::ParallelDecomp::pseudorandom);
      tree::Node::Ptr tree = make_tree(m, imbalanced);
      std::vector<Int> cells(m.ncell(), 0);
      mark_cells(tree, cells);
      for (Int i = 0; i < m.ncell(); ++i)
        if (cells[i] != 1) ++ne;
    }
  return ne;
}
} // namespace test
} // namespace oned

tree::Node::Ptr tree::make_tree_over_1d_mesh (const Parallel::Ptr& p, const Int& ncells,
                                              const bool imbalanced) {
  return oned::make_tree(oned::Mesh(ncells, p), imbalanced);
}

namespace test {
Int unittest_NodeSets (const Parallel::Ptr& p) {
  using Mesh = oned::Mesh;
  const Int szs[] = { p->size(), 3*p->size() };
  const Mesh::ParallelDecomp::Enum dists[] = { Mesh::ParallelDecomp::pseudorandom,
                                               Mesh::ParallelDecomp::contiguous };
  Int nerr = 0;
  for (size_t is = 0; is < sizeof(szs)/sizeof(*szs); ++is)
    for (size_t id = 0; id < sizeof(dists)/sizeof(*dists); ++id)
      for (bool imbalanced: {false, true}) {
        Mesh m(szs[is], p, dists[id]);
        tree::Node::Ptr tree = make_tree(m, imbalanced);
        impl::NodeSets::ConstPtr nodesets = impl::analyze(p, m.ncell(), tree);
        tree = nullptr;
        nerr += impl::unittest(p, nodesets, m.ncell());
      }
  return nerr;
}

Int unittest_QLT (const Parallel::Ptr& p, const bool write_requested=false) {
  using Mesh = oned::Mesh;
  const Int szs[] = { p->size(), 2*p->size(), 7*p->size(), 21*p->size() };
  const Mesh::ParallelDecomp::Enum dists[] = { Mesh::ParallelDecomp::contiguous,
                                               Mesh::ParallelDecomp::pseudorandom };
  Int nerr = 0;
  for (size_t is = 0, islim = sizeof(szs)/sizeof(*szs); is < islim; ++is)
    for (size_t id = 0, idlim = sizeof(dists)/sizeof(*dists); id < idlim; ++id)
    for (bool imbalanced: {false, true}) {
      if (p->amroot()) {
        std::cout << " (" << szs[is] << ", " << id << ", " << imbalanced << ")";
        std::cout.flush();
      }
      Mesh m(szs[is], p, dists[id]);
      tree::Node::Ptr tree = make_tree(m, imbalanced);
      const bool write = (write_requested && m.ncell() < 3000 &&
                          is == islim-1 && id == idlim-1);
      nerr += test::test_qlt(p, tree, m.ncell(), 1, write);
    }
  return nerr;
}

Int run_unit_and_randomized_tests (const Parallel::Ptr& p, const Input& in) {
  Int nerr = 0;
  if (in.unittest) {
    Int ne;
    ne = oned::Mesh::unittest(p);
    if (ne && p->amroot()) std::cerr << "FAIL: Mesh::unittest()\n";
    nerr += ne;
    ne = oned::test::unittest(p);
    if (ne && p->amroot()) std::cerr << "FAIL: oned::unittest_tree()\n";
    nerr += ne;
    ne = unittest_NodeSets(p);
    if (ne && p->amroot()) std::cerr << "FAIL: oned::unittest_NodeSets()\n";
    nerr += ne;
    ne = unittest_QLT(p, in.write);
    if (ne && p->amroot()) std::cerr << "FAIL: oned::unittest_QLT()\n";
    nerr += ne;
    if (p->amroot()) std::cout << "\n";
  }
  // Performance test.
  if (in.perftest && in.ncells > 0) {
    oned::Mesh m(in.ncells, p,
                 (in.pseudorandom ?
                  oned::Mesh::ParallelDecomp::pseudorandom :
                  oned::Mesh::ParallelDecomp::contiguous));
    Timer::init();
    Timer::start(Timer::total); Timer::start(Timer::tree);
    tree::Node::Ptr tree = make_tree(m, false);
    Timer::stop(Timer::tree);
    test::test_qlt(p, tree, in.ncells, in.nrepeat, false, in.verbose);
    Timer::stop(Timer::total);
    if (p->amroot()) Timer::print();
  }
  return nerr;
}

} // namespace test
} // namespace qlt
} // namespace cedr

#ifdef KOKKOS_HAVE_SERIAL
template class cedr::qlt::QLT<Kokkos::Serial>;
#endif
#ifdef KOKKOS_HAVE_OPENMP
template class cedr::qlt::QLT<Kokkos::OpenMP>;
#endif
#ifdef KOKKOS_HAVE_CUDA
template class cedr::qlt::QLT<Kokkos::Cuda>;
#endif
