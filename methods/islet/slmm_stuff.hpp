#ifndef INCLUDE_SLMM_STUFF_HPP
#define INCLUDE_SLMM_STUFF_HPP

#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <sstream>
#include <limits>
#include <memory>
#include <vector>

namespace slmm {
#define pr(m) do {                              \
    std::stringstream _ss_;                     \
    _ss_ << m << std::endl;                     \
    std::cerr << _ss_.str();                    \
  } while (0)
#define prc(m) pr(#m << " | " << (m))
#define puf(m)"(" << #m << " " << (m) << ")"
#define pu(m) << " " << puf(m)
template<typename T>
static void prarr (const std::string& name, const T* const v, const size_t n) {
  std::cerr << name << ": ";
  for (size_t i = 0; i < n; ++i) std::cerr << " " << v[i];
  std::cerr << "\n";
}

#define SIQK_THROW_IF(condition, message) do {                          \
    if (condition) {                                                    \
      std::stringstream _ss_;                                           \
      _ss_ << __FILE__ << ":" << __LINE__ << ": The condition:\n" << #condition \
        "\nled to the exception\n" << message << "\n";                  \
      throw std::logic_error(_ss_.str());                               \
    }                                                                   \
  } while (0)

#define SIQK_STDERR_IF(condition, message) do { \
  try { SIQK_THROW_IF(condition, message); } \
  catch (const std::logic_error& e) { std::cerr << e.what(); } \
} while (0)

typedef int Int;
typedef double Real;

template<typename T> inline constexpr T square (const T& x) { return x*x; }
template<typename T> inline constexpr T cube (const T& x) { return x*x*x; }

struct consts {
  static constexpr Real earth_radius_m = 6.37122e6;
};

template<typename T> inline T sign (const T& a) { return a >= 0 ? 1 : -1; }

inline Real sec2day (const Real sec) { return sec/(24*3600); }
inline Real day2sec (const Real day) { return day*(24*3600); }

// Output is in radians.
//todo Make a version that lets you pass R = mag(x,y,z).
inline void xyz2ll (const Real x, const Real y, const Real z,
                    Real& lat, Real& lon) {
  const Real r = std::sqrt(square(x) + square(y) + square(z));
  lat = std::asin(z/r);
  lon = std::atan2(y, x);
}

// Input is in radians.
inline void ll2xyz (const Real lat, const Real lon, Real& x, Real& y, Real& z,
                    const Real radius = 1) {
  const Real sinl = std::sin(lat), cosl = std::cos(lat);
  x = radius*std::cos(lon)*cosl;
  y = radius*std::sin(lon)*cosl;
  z = radius*sinl;
}

inline constexpr Real m2radlat (const Real m)
{ return m/consts::earth_radius_m; }

inline Real m2radlon(const Real lat, const Real m)
{ return m2radlat(m)/std::abs(std::cos(lat)); }

inline constexpr Real deg2rad (const Real v) { return v * (M_PI/180); }
inline constexpr Real rad2deg (const Real v) { return v * (180/M_PI); }

inline Real reldif (const Real a, const Real b, const Real abstol = 0)
{ return std::abs(b - a)/(abstol + std::abs(a)); }

// Row-major R.
inline void form_rotation (const Real axis[3], const Real angle, Real r[9]) {
  const Real nrm = std::sqrt(square(axis[0]) + square(axis[1]) +
                             square(axis[2]));
  const Real& x = axis[0] / nrm, & y = axis[1] / nrm, & z = axis[2] / nrm,
    & th = angle;
  const Real cth = std::cos(th), sth = std::sin(th), omcth = 1 - cth;
  r[0] = cth + x*x*omcth;
  r[3] = y*x*omcth + z*sth;
  r[6] = z*x*omcth - y*sth;
  r[1] = x*y*omcth - z*sth;
  r[4] = cth + y*y*omcth;
  r[7] = z*y*omcth + x*sth;
  r[2] = x*z*omcth + y*sth;
  r[5] = y*z*omcth - x*sth;
  r[8] = cth + z*z*omcth;
}

/*! \brief RAII std stream state saver.
 *
 * Example: Preserve std::cout's state so manipulations don't affect others' use
 * of cout.
 */
template<typename Stream> class IosSaver {
  Stream& s_;
  std::ios state_;
public:
  IosSaver (Stream& s) : s_(s), state_(nullptr) { state_.copyfmt(s); }
  IosSaver (const IosSaver& ios) : s_(ios.s_), state_(nullptr)
  { state_.copyfmt(ios.state_); }
  IosSaver operator= (const IosSaver&) = delete;
  ~IosSaver () { s_.copyfmt(state_); }
};
template<typename Stream> inline IosSaver<Stream> save_ios (Stream& s)
{ return IosSaver<Stream>(s); }

template<typename T>
inline T* tin (T* const p, const char* const msg="") {
  if ( ! p)
    throw std::runtime_error(std::string(std::string("Null pointer: ") + msg));
  return p;
}

inline bool
eq (const std::string& a, const char* const b1, const char* const b2 = 0) {
  return (a == std::string(b1) || (b2 && a == std::string(b2)) ||
          a == std::string("-") + std::string(b1));
}

template <typename T> class Array {
  T* p_;
  std::size_t n_, cap_;
public:
  Array () { init(); }
  Array(std::size_t n);
  Array(std::size_t n, const T& init);
  Array(const Array<T>& a);
  ~Array () { clear(); }
  // Initialize the object with the assumption that all variables are uninit'ed
  // prior to calling.
  void init();
  void clear();
  // optclear means optionally clear. The function has the semantics of
  // clearing, but it may not actually release the memory.
  void optclear_and_resize(std::size_t n);
  // _ft indicates first touch.
  void optclear_and_resize_ft(std::size_t n);
  void optclear_and_resize(std::size_t n, const T& i);
  void optclear_and_reserve(std::size_t n);
  void optclear_and_reserve_ft(std::size_t n);
  T& operator[] (std::size_t i) { return p_[i]; }
  const T& operator[] (std::size_t i) const { return p_[i]; }
  T& back () { return p_[n_-1]; }
  const T& back () const { return p_[n_-1]; }
  std::size_t size () const { return n_; }
  bool empty () const { return size() == 0; }
  T* data () const { return p_; }
  // This does not realloc; reserve must provide the necessary memory. It does
  // not throw, either. It asserts.
  void unsafe_push_back(const T& e);
  T* begin () { return p_; }
  T* end () { return p_ + n_; }
  const T* begin () const { return p_; }
  const T* end () const { return p_ + n_; }
  void set (const T& v) { for (std::size_t i = 0; i < n_; ++i) p_[i] = v; }
};

template<typename T> inline Int len (const Array<T>& v)
{ return static_cast<Int>(v.size()); }

template<typename T> inline void touch (T* const p, const size_t n,
                                        const T& init = T()) {
  // 1 KB should be a safe lower bound on page size. Touch enough to touch every
  // page; I don't think there's any need to touch more memory than that.
  for (size_t i = 0; i < n; i += 1024 / sizeof(T))
    p[i] = init;
  // Make sure the last part is touched.
  if (n) p[n-1] = init;
}
template<typename T> inline T*
allocn (const size_t n, const bool first_touch = false) {
  if ( ! n) return 0;
  T* p = new T[n];
  if (first_touch) touch(p, n);
  return p;
}
template<typename T> inline void deln (T*& p) {
  if (p) delete[] p;
  p = 0;
}
template<typename T> inline void deln_const (const T* p) {
  if (p) delete[] p;
}
template<typename T> inline void del (T*& p) {
  if (p) delete p;
  p = 0;
}

template<typename T>
inline void Array<T>::init () {
  n_ = cap_ = 0;
  p_ = 0;
}

template<typename T>
inline Array<T>::Array (std::size_t n)
  : p_(0), n_(0), cap_(0)
{ optclear_and_resize(n); }

template<typename T>
inline Array<T>::Array (std::size_t n, const T& init)
  : p_(0), n_(0), cap_(0)
{ optclear_and_resize(n, init); }

template<typename T>
inline Array<T>::Array (const Array<T>& a) {
  init();
  optclear_and_resize(a.size());
  std::copy(a.begin(), a.end(), begin());
}

template<typename T>
inline void Array<T>::clear () {
  n_ = cap_ = 0;
  deln(p_);
}

template<typename T>
inline void Array<T>::optclear_and_reserve (std::size_t n) {
  n_ = 0;
  if (n <= cap_) return;
  clear();
  p_ = allocn<T>(n);
  cap_ = n;
}

template<typename T>
inline void Array<T>::optclear_and_reserve_ft (std::size_t n) {
  n_ = 0;
  if (n <= cap_) return;
  clear();
  p_ = allocn<T>(n, true);
  cap_ = n;
}

template<typename T>
inline void Array<T>::optclear_and_resize (std::size_t n) {
  if (n <= cap_) {
    n_ = n;
    return;
  }
  optclear_and_reserve(n);
  n_ = n;
}

template<typename T>
inline void Array<T>::optclear_and_resize_ft (std::size_t n) {
  if (n <= cap_) {
    n_ = n;
    return;
  }
  optclear_and_reserve_ft(n);
  n_ = n;
}

template<typename T>
inline void Array<T>::optclear_and_resize (std::size_t n, const T& init) {
  optclear_and_resize(n);
  for (std::size_t i = 0; i < n_; ++i)
    memcpy(p_ + i, &init, sizeof(init));
}

template<typename T>
inline void Array<T>::unsafe_push_back (const T& e) {
  assert(n_ < cap_);
  p_[n_++] = e;
}

#if defined SLMM_NP_GT_4 && ! defined SLMM_NP_MAX
# define SLMM_NP_MAX 6
#endif

struct Basis {
  Int np, monotone_type;
  Basis () : np(0), monotone_type(-1) {} // invalid values
  Basis (const Int inp, const Int imt) : np(inp), monotone_type(imt) {}
};

// These basis functions follow TempestRemap's monotone_type 0 and 1 basis
// functions up to np 4. For the nonmonotone basis functions, see also Appendix
// F.10 of Boyd's Chebyshev and Fourier Spectral Methods.
class GLL {
  const Real oo3 = 1.0/3.0;
  const Real to3 = 2.0/3.0;
  const Real oo6 = 1.0/6.0;
  const Real oo8 = 1.0/8.0;
  const Real sqrt5 = std::sqrt(5.0);
  const Real oosqrt5 = 1.0/sqrt5;
  const Real np2_coord[2] = {-1.0, 1.0};
  const Real np2_wt[2]    = {1.0, 1.0};
  const Real np3_coord[3] = {-1.0, 0.0, 1.0};
  const Real np3_wt[3]    = {oo3, 2.0 - to3, oo3};
  const Real np4_coord[4] = {-1.0, -oosqrt5, oosqrt5, 1.0};
  const Real np4_wt[4]    = {oo6, 1.0 - oo6, 1.0 - oo6, oo6};
#ifdef SLMM_NP_GT_4
  const Real sqrt3o7 = std::sqrt(3.0/7.0);
  const Real np5_coord[5] = {-1.0, -sqrt3o7, 0, sqrt3o7, 1.0};
  const Real np5_wt[5]    = {0.1, 49.0/90.0, 32.0/45.0, 49.0/90.0, 0.1};

  const Real np6_coord[6] = {
# define np6a std::sqrt(1.0/3.0 + 2.0*std::sqrt(7.0)/21.0)
# define np6b std::sqrt(1.0/3.0 - 2.0*std::sqrt(7.0)/21.0)
    -1, -np6a, -np6b, np6b, np6a, 1
# undef np6b
# undef np6a
  };
  const Real np6_wt[6] = {
# define v0 1.0/15.0
# define v1 (14 - std::sqrt(7.0))/30.0
# define v2 (14 + std::sqrt(7.0))/30.0
    v0, v1, v2, v2, v1, v0
# undef v2
# undef v1
# undef v0
  };

# if SLMM_NP_MAX >= 11
  const Real np11_coord[11] = {
    -1, -0.93400143040805913433, -0.78448347366314441862, -0.56523532699620500647,
    -0.29575813558693939143, 0, 0.29575813558693939143, 0.56523532699620500647,
    0.78448347366314441862, 0.93400143040805913433, 1
  };
  const Real np11_wt[11] = {
    0.01818181818181818182, 0.10961227326699486446, 0.18716988178030520411,
    0.24804810426402831404, 0.28687912477900808868, 0.30021759545569069379,
    0.28687912477900808868, 0.24804810426402831404, 0.18716988178030520411,
    0.10961227326699486446, 0.01818181818181818182
  };
# endif
#endif

public:

  enum { np_max =
#ifdef SLMM_NP_GT_4
         SLMM_NP_MAX
#else
         4
#endif
  };

  template <typename Scalar>
  static
  void eval_lagrange_poly (const Int& np, const Scalar* const x_gll,
                           const Scalar& x, Scalar* const y) {
    for (int i = 0; i < np; ++i) {
      Scalar f = 1;
      for (int j = 0; j < np; ++j)
        f *= (j == i ?
              1 :
              (x - x_gll[j]) / (x_gll[i] - x_gll[j]));
      y[i] = f;
    }
  }

  template <typename Scalar>
  static
  void eval_lagrange_poly_derivative (const Int& np, const Scalar* const x_gll,
                                      const Scalar& x, Scalar* const y) {
    for (int i = 0; i < np; ++i) {
      Scalar f = 0;
      for (int j = 0; j < np; ++j) {
        if (j == i) continue;
        Scalar g = 1;
        for (int k = 0; k < np; ++k)
          g *= ((k == i) ? 1 :
                ((k == j ? 1 : (x - x_gll[k])) /
                 (x_gll[i] - x_gll[k])));
        f += g;
      }
      y[i] = f;
    }
  }

  GLL () {}

 
  void get_coef (const Int& np, const Real*& coord, const Real*& wt) const {
    switch (np) {
    case 2:
      coord = np2_coord;
      wt = np2_wt;
      break;
    case 3:
      coord = np3_coord;
      wt = np3_wt;
      break;
    case 4:
      coord = np4_coord;
      wt = np4_wt;
      break;
#ifdef SLMM_NP_GT_4
    case 5:
      coord = np5_coord;
      wt = np5_wt;
      break;      
    case 6:
      coord = np6_coord;
      wt = np6_wt;
      break;
# if SLMM_NP_MAX >= 11
    case 11:
      coord = np11_coord;
      wt = np11_wt;
      break;
# endif
#endif
    default:
      coord = nullptr;
      wt = nullptr;
      SIQK_THROW_IF(true, "GLL::get_coef: order not supported.");
    }
  }
 
  void get_coef (const Basis& b, const Real*& coord, const Real*& wt) const {
    get_coef(b.np, coord, wt);
  }

  // x in [-1, 1].
 
  void eval (const Basis& b, const Real& x, Real* const ge) const {
    if (b.monotone_type < 0 || b.monotone_type > 1)
      SIQK_THROW_IF(true, "GLL::eval: monotone_type not supported.");
    switch (b.np) {
    case 2: {
      ge[0] = 0.5*(1.0 - x);
      ge[1] = 0.5*(1.0 + x);
    } break;
    case 3: {
      const Real x2 = x*x;
      if (b.monotone_type == 0) {
        ge[0] = 0.5*(x2 - x);
        ge[1] = 1.0 - x2;
        ge[2] = 0.5*(x2 + x);
      } else {
        if (x < 0) {
          ge[0] = x2;
          ge[1] = 1 - x2;
          ge[2] = 0;
        } else {
          ge[0] = 0;
          ge[1] = 1 - x2;
          ge[2] = x2;
        }
      }
    } break;
    case 4: {
      const Real x2 = x*x;
      if (b.monotone_type == 0) {
        ge[0] = (1.0 - x)*(5.0*x2 - 1.0)*oo8;
        ge[1] = -sqrt5*oo8*(sqrt5 - 5.0*x)*(x2 - 1.0);
        ge[2] = -sqrt5*oo8*(sqrt5 + 5.0*x)*(x2 - 1.0);
        ge[3] = (1.0 + x)*(5.0*x2 - 1.0)*oo8;
      } else {
        const Real a0 = (1 + sqrt5)/16;
        const Real b0 = (5 + sqrt5)/16;
        const Real c0 = (-5 - 5*sqrt5)/16;
        const Real d0 = (-25 - 5*sqrt5)/16;
        const Real b1 = -0.75*sqrt5;
        const Real d1 = 1.25*sqrt5;
        if (x < -oosqrt5) {
          ge[0] = a0 + x*(b0 + x*(c0 + x*d0));
          ge[1] = 1 - ge[0];
          ge[2] = 0;
          ge[3] = 0;
        } else if (x < oosqrt5) {
          ge[0] = 0;
          ge[1] = 0.5 + x*(b1 + x2*d1);
          ge[2] = 1 - ge[1];
          ge[3] = 0;
        } else {
          ge[0] = 0;
          ge[1] = 0;
          ge[3] = a0 - x*(b0 - x*(c0 - x*d0));
          ge[2] = 1 - ge[3];
        }
      }
    } break;
#ifdef SLMM_NP_GT_4
    case 5: {
      assert(b.monotone_type == 0);
      const Real x2 = x*x;
      const Real sqrt21 = std::sqrt(21.0);
      const Real o7o24 = 7.0/24.0;
      ge[0] = oo8*x*(x - 1)*(7*x2 - 3);
      ge[1] = -o7o24*x*(x2 - 1)*(7*x - sqrt21);
      ge[2] = oo3*(x2 - 1)*(7*x2 - 3);
      ge[3] = -o7o24*x*(x2 - 1)*(7*x + sqrt21);
      ge[4] = oo8*x*(x + 1)*(7*x2 - 3);
    } break;
    case 6: {
      assert(b.monotone_type == 0);
      eval_lagrange_poly<Real>(6, np6_coord, x, ge);
    } break;
# if SLMM_NP_MAX >= 11
    case 11: {
      assert(b.monotone_type == 0);
      eval_lagrange_poly<Real>(11, np11_coord, x, ge);
    } break;
# endif
#endif
    default:
      for (int i = 0; i < b.np; ++i) ge[i] = 0;
      SIQK_THROW_IF(true, "GLL::eval: order not supported.");
    }
  }

  void eval_derivative (const Basis& b, const Real& x, Real* const ge) const {
    if (b.monotone_type != 0)
      SIQK_THROW_IF(true, "GLL::eval: monotone_type not supported.");
    switch (b.np) {
    case 2: {
      ge[0] = -0.5;
      ge[1] =  0.5;
    } break;
    case 3: {
      const Real x2p = 2*x;
      ge[0] =  0.5*(x2p - 1);
      ge[1] = -x2p;
      ge[2] =  0.5*(x2p + 1);
    } break;
    case 4: {
      ge[0] =  oo8*((10 - 15*x)*x + 1);
      ge[1] = -sqrt5*oo8*((2*sqrt5 - 15*x)*x + 5);
      ge[2] = -sqrt5*oo8*((2*sqrt5 + 15*x)*x - 5);
      ge[3] =  oo8*((10 + 15*x)*x - 1);
    } break;
#ifdef SLMM_NP_GT_4
    case 5: {
      const Real x2 = x*x;
      const Real sqrt21 = std::sqrt(21.0);
      const Real o7o24 = 7.0/24.0;
      ge[0] = oo8*(((28*x - 21)*x - 6)*x + 3);
      ge[1] = o7o24*(((-28*x + 3*sqrt21)*x + 14)*x - sqrt21);
      ge[2] = oo3*4*x*(7*x2 - 5);
      ge[3] = o7o24*(((-28*x - 3*sqrt21)*x + 14)*x + sqrt21);
      ge[4] = oo8*(((28*x + 21)*x - 6)*x - 3);
    } break;
    case 6: {
      assert(b.monotone_type == 0);
      eval_lagrange_poly_derivative<Real>(6, np6_coord, x, ge);
    } break;
# if SLMM_NP_MAX >= 11
    case 11: {
      assert(b.monotone_type == 0);
      eval_lagrange_poly_derivative<Real>(11, np11_coord, x, ge);
    } break;
# endif
#endif
    default:
      for (int i = 0; i < b.np; ++i) ge[i] = 0;
      SIQK_THROW_IF(true, "GLL::eval_derivative: order not supported.");
    }
  }
};

}

#endif
