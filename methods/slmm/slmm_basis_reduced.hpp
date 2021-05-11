#ifndef INCLUDE_SLMM_REDUCED_HPP
#define INCLUDE_SLMM_REDUCED_HPP

#include "slmm_islet.hpp"

/* Generate the weights using:
   ./slmm_test -c islet_compute
*/

namespace slmm {

struct UniformNodeReduced : public Basis {
  const char* name () const override { return "UniformNodeReduced"; }
  bool gll_nodes () const override { return false; }
  bool get_x(const Int& np, const Real*& x) const override;
  bool get_w(const Int& np, const Real*& wt) const override;
  Int max_degree(const Int& np) const override;
  bool eval(const Int& np, const Real& x, Real* const v) const override;
  bool eval_derivative(const Int& np, const Real& x, Real* const v) const override;

protected:
  const Real w_np2[2] = { 1.0000000000000000e+00, 1.0000000000000000e+00};
  const Real w_np3[3] = { 5.0000000000000000e-01, 1.0000000000000000e+00, 5.0000000000000000e-01};
  const Real w_np4[4] = { 3.3333333333333337e-01, 6.6666666666666663e-01, 6.6666666666666663e-01, 3.3333333333333337e-01};
  const Real w_np5[5] = { 2.5000000000000000e-01, 5.0000000000000000e-01, 5.0000000000000000e-01, 5.0000000000000000e-01, 2.5000000000000000e-01};
  const Real w_np6[6] = { 2.0000000000000007e-01, 4.0000000000000002e-01, 3.9999999999999991e-01, 3.9999999999999991e-01, 4.0000000000000002e-01, 2.0000000000000007e-01};
  const Real w_np7[7] = { 1.6666666666666663e-01, 3.3333333333333337e-01, 3.3333333333333337e-01, 3.3333333333333337e-01, 3.3333333333333337e-01, 3.3333333333333337e-01, 1.6666666666666663e-01};
  const Real w_np8[8] = { 1.4285714285714288e-01, 2.8571428571428575e-01, 2.8571428571428575e-01, 2.8571428571428575e-01, 2.8571428571428575e-01, 2.8571428571428575e-01, 2.8571428571428575e-01, 1.4285714285714288e-01};
  const Real w_np9[9] = { 1.2500000000000000e-01, 2.5000000000000000e-01, 2.5000000000000000e-01, 2.5000000000000000e-01, 2.5000000000000000e-01, 2.5000000000000000e-01, 2.5000000000000000e-01, 2.5000000000000000e-01, 1.2500000000000000e-01};
  const Real w_np10[10] = { 1.1111111111111108e-01, 2.2222222222222215e-01, 2.2222222222222215e-01, 2.2222222222222215e-01, 2.2222222222222221e-01, 2.2222222222222221e-01, 2.2222222222222215e-01, 2.2222222222222215e-01, 2.2222222222222215e-01, 1.1111111111111108e-01};
  const Real w_np11[11] = { 1.0000000000000001e-01, 2.0000000000000007e-01, 2.0000000000000007e-01, 2.0000000000000007e-01, 2.0000000000000007e-01, 2.0000000000000001e-01, 2.0000000000000007e-01, 2.0000000000000007e-01, 2.0000000000000007e-01, 2.0000000000000007e-01, 1.0000000000000001e-01};
  const Real w_np12[12] = { 9.0909090909090912e-02, 1.8181818181818180e-01, 1.8181818181818174e-01, 1.8181818181818180e-01, 1.8181818181818180e-01, 1.8181818181818180e-01, 1.8181818181818180e-01, 1.8181818181818180e-01, 1.8181818181818180e-01, 1.8181818181818174e-01, 1.8181818181818180e-01, 9.0909090909090912e-02};
  const Real w_np13[13] = { 8.3333333333333301e-02, 1.6666666666666660e-01, 1.6666666666666666e-01, 1.6666666666666666e-01, 1.6666666666666666e-01, 1.6666666666666666e-01, 1.6666666666666660e-01, 1.6666666666666666e-01, 1.6666666666666666e-01, 1.6666666666666666e-01, 1.6666666666666666e-01, 1.6666666666666660e-01, 8.3333333333333301e-02};
  const Real w_np16[16] = { 6.6666666666666652e-02, 1.3333333333333333e-01, 1.3333333333333339e-01, 1.3333333333333339e-01, 1.3333333333333333e-01, 1.3333333333333339e-01, 1.3333333333333339e-01, 1.3333333333333333e-01, 1.3333333333333333e-01, 1.3333333333333339e-01, 1.3333333333333339e-01, 1.3333333333333333e-01, 1.3333333333333339e-01, 1.3333333333333339e-01, 1.3333333333333333e-01, 6.6666666666666652e-02};
};

} // namespace slmm

#endif