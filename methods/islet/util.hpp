#ifndef INCLUDE_ISLET_UTIL_HPP
#define INCLUDE_ISLET_UTIL_HPP

#include <sstream>
#include <iostream>
#include <fstream>

#include "types.hpp"

namespace islet {
#define throw_if(condition, message) do {                               \
    if (condition) {                                                    \
      std::stringstream _ss_;                                           \
      _ss_ << __FILE__ << ":" << __LINE__ << ": The condition:\n"       \
           << #condition "\nled to the exception\n" << message << "\n"; \
        throw std::logic_error(_ss_.str());                             \
    }                                                                   \
  } while (0)

#define require(condition) do {                                         \
    if ( ! (condition)) {                                               \
      std::stringstream _ss_;                                           \
      _ss_ << __FILE__ << ":" << __LINE__ << ": FAIL:\n" << #condition  \
        << "\n";                                                        \
      throw std::logic_error(_ss_.str());                               \
    }                                                                   \
  } while (0)
#define require_msg(condition, message) do {                            \
    if ( ! (condition)) {                                               \
      std::stringstream _ss_;                                           \
      _ss_ << __FILE__ << ":" << __LINE__ << ": FAIL:\n" << #condition  \
           << "\nmessage:\n" << message << "\n";                        \
      throw std::logic_error(_ss_.str());                               \
    }                                                                   \
  } while (0)

template<typename T> inline T sign (const T& a) { return a >= 0 ? 1 : -1; }
template<typename T> inline constexpr T square (const T& x) { return x*x; }
inline Real reldif (const Real a, const Real b, const Real abstol = 0)
{ return std::abs(b - a)/(abstol + std::abs(a)); }

#define pr(m) do {                              \
    std::stringstream _ss_;                     \
    _ss_ << m << std::endl;                     \
    std::cerr << _ss_.str();                    \
  } while (0)
#define prc(m) pr(#m << " | " << (m))
#define puf(m) "(" << #m << " " << (m) << ")"
#define pu(m) << " " << puf(m)

template <typename T>
static void prarr (const std::string& name, const T* const v, const size_t n) {
  std::stringstream ss;
  ss << name << " = [";
  for (size_t i = 0; i < n; ++i) ss << " " << v[i];
  ss << "]";
  pr(ss.str());
}
template <typename Array>
static void prarr (const std::string& name, const Array& a) {
  prarr(name, a.data(), a.size());
}

#define mprarr(a) prarr(#a, a)

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

inline double urand () { return rand() / ((double) RAND_MAX + 1.0); }

template <typename T>
bool write (std::ofstream& os, const T s) {
  return ! os.write((const char*) &s, sizeof(T)).bad();
}

template <typename T>
bool write (std::ofstream& os, const Int n, const T* const d) {
  return (write(os, n) &&
          ! os.write((const char*) d, n*sizeof(T)).bad());
}

template <typename T>
bool read (std::ifstream& os, T& s) {
  return ! os.read((char*) &s, sizeof(T)).bad();
}

template <typename T>
bool read (std::ifstream& os, Int& n, T* const d) {
  return (read(os, n) &&
          ! os.read((char*) d, n*sizeof(T)).bad());
}

} // namespace islet

#endif
