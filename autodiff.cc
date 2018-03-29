#include <cmath>
#include <type_traits>

#include <benchmark/benchmark.h>

template<typename T>
constexpr T factorial(T n) {
  T rv = 1;
  for (T i = 2; i <= n; ++i) {
    rv *= i;
  }
  return rv;
}

using f32 = float;
using f64 = double;

/**
 * Autodiff
 */
struct Zero {
  typedef Zero grad;
  template<typename T>
  static constexpr T eval(T p) {
    return 0;
  }
};
struct One {
  typedef Zero grad;
  template<typename T>
  static constexpr T eval(T p) {
    return 1;
  }
};

template<typename T>
struct Input {
  typedef One grad;
  static constexpr T eval(T p) {
    return p;
  }
};

template<typename A, typename B>
struct Add {
  typedef Add<typename A::grad, typename B::grad> grad;
  template<typename T>
  static constexpr T eval(T p) {
    return A::eval(p) + B::eval(p);
  }
};

template<typename A, typename B>
struct Sub {
  typedef Sub<typename A::grad, typename B::grad> grad;
  template<typename T>
  static constexpr T eval(T p) {
    return A::eval(p) - B::eval(p);
  }
};

template<typename A, typename B>
struct Mul {
  typedef Add<Mul<typename A::grad, B>, Mul<A, typename B::grad>> grad;
  template<typename T>
  static constexpr T eval(T p) {
    return A::eval(p) * B::eval(p);
  }
};

template<typename A, typename B>
struct Div {
  typedef Div<Sub<Mul<B, typename A::grad>, Mul<A, typename B::grad>>, Mul<B, B>> grad;
  template<typename T>
  static constexpr T eval(T p) {
    return A::eval(p) / B::eval(p);
  }
};

template<typename A>
struct Exp {
  typedef Mul<Exp<A>, typename A::grad> grad;
  template<typename T>
  static constexpr T eval(T p) {
    return exp(A::eval(p));
  }
};

template<typename A>
struct Ln {
  typedef Div<typename A::grad, A> grad;
  template<typename T>
  static constexpr T eval(T p) {
    return log(A::eval(p));
  }
};

template<typename A, typename B>
struct Pow {
  typedef Mul<Pow<A, Sub<B, One>>, Add<Mul<B, typename A::grad>, Mul<A, Mul<Ln<A>, typename B::grad>>>> grad;
  template<typename T>
  static constexpr T eval(T p) {
    return pow(A::eval(p), B::eval(p));
  }
};

#define CONST_VAL(NAME, VALUE)                  \
  struct NAME {                                 \
    typedef Zero grad;                          \
    template<typename T>                        \
    static constexpr T eval(T p) {              \
      return VALUE;                             \
    }                                           \
  }

CONST_VAL(PI, 3.14159265);

// b: basis
// e: Exponent
// Annahmen: Typ T besitzt  T operator*(T, T)  und eine 1.
template<typename T, typename I>
typename std::enable_if<std::is_unsigned<I>::value, T>::type
powi(T b, I e)
{
  if (e <= 0) return T(1);
  I h = ~(~0u >> 1);
  while (!(e & h)) h >>= 1;
  // h maskiert nun das erste gesetzte Bit in e. (#)
  T r = b;

  // solange weitere Bits zu prüfen sind (das erste wurde durch r = b bereits abgearbeitet),
  while (h >>= 1)
    {
      r *= r; // quadrieren
      if (e & h) r *= b; // falls Bit gesetzt, multiplizieren
    }
  // h == 0, d. h. alle Bits geprüft.
  return r;
}
/**
 * Taylor Series:
 * \sum_{n=0}^{\infty} \frac{f^(n)(a)}{n!} (x-a)^n
 */
template<typename F, std::size_t N, std::size_t n=0>
struct Taylor {
  template<typename T>
  static constexpr inline
  typename std::enable_if<n != N, T>::type
  eval(T a, T x) __attribute__((always_inline)) {
    T term = F::eval(a)/factorial(n) * powi(x-a, n);
    return term + Taylor<typename F::grad, N, n+1>::eval(a, x);
  }
  template<typename T>
  static constexpr inline
  typename std::enable_if<n == N, T>::type
  eval(T a, T x) __attribute__((always_inline)) {
    return 0;
  }
};

template<typename F, std::size_t N>
struct derive {
  typedef typename derive<typename F::grad, N-1>::value value;
};

template<typename F>
struct derive<F, 0> {
  typedef F value;
};

typedef Pow<Ln<Input<f32>>, Div<PI, Input<f32>>> fn;
typedef derive<fn, 0>::value fnd;

extern f32 f(f32 x) {
  return fnd::eval(x);
}

extern f32 af(f32 x) {
  return Taylor<fnd, 5>::eval(2.0f, x);
}

constexpr int maxrange = 10000;
constexpr float fromstate(benchmark::State &state) {
    return state.range(0)/static_cast<float>(maxrange)+1.25f;
}

#include <iostream>
using namespace std;
static void BM_approx(benchmark::State& state) {
  volatile float x;
  for (auto _ : state) {
    benchmark::DoNotOptimize(x = af(fromstate(state)));
  }
}

static void BM_real(benchmark::State& state) {
  volatile float x;
  for (auto _ : state) {
    benchmark::DoNotOptimize(x = f(fromstate(state)));
  }
}

BENCHMARK(BM_approx)->Range(0, maxrange);
BENCHMARK(BM_real)->Range(0, maxrange);

//BENCHMARK_MAIN();

int main() {
  std::cout << "x,f(x),af(x)" << std::endl;
  for (float x = 1.25; x <= 3; x+=0.0001) {
    std::cout << x << ",";
    std::cout << f(x) << ",";
    std::cout << af(x) << std::endl;
  }
  return 0;
}
