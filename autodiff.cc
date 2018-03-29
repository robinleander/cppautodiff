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

/**
 * Taylor Series:
 * \sum_{n=0}^{\infty} \frac{f^(n)(a)}{n!} (x-a)^n
 */
template<typename F, std::size_t N, std::size_t n=0>
struct Taylor {
  template<typename T>
  static constexpr
  typename std::enable_if<n != N, T>::type
  eval(T a, T x) {
    T term = F::eval(a)/factorial(n) * pow(x-a, n);
    return term + Taylor<typename F::grad, N, n+1>::eval(a, x);
  }
  template<typename T>
  static constexpr
  typename std::enable_if<n == N, T>::type
  eval(T a, T x) {
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

f32 f(f32 x) {
  return fnd::eval(x);
}

f32 af(f32 x) {
  return Taylor<fnd, 4>::eval(2.0f, x);
}

constexpr float stepsize = 0.0000001;
constexpr float start = 1.25;

#include <iostream>
using namespace std;
static void BM_approx(benchmark::State& state) {
  float x = start;
  for (auto _ : state) {
    benchmark::DoNotOptimize(af(x));
    x+=stepsize;
  }
}

static void BM_real(benchmark::State& state) {
  float x = start;
  for (auto _ : state) {
      benchmark::DoNotOptimize(f(x));
    x += stepsize;
  }
}

BENCHMARK(BM_approx);
BENCHMARK(BM_real);

BENCHMARK_MAIN();

int notmain() {
  for (float x = 1.25; x <= 3; x+=0.25) {
    std::cout << x << std::endl;
    std::cout << af(x) << std::endl;
    std::cout << f(x) << std::endl;
    std::cout << abs(af(x)-f(x)) << std::endl;
    std::cout << std::endl;
  }
  return 0;
}
