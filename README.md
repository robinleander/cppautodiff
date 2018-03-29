# C++ TMP Autodiff

Automatic differentiation at compiletime!

## Usage with Docker

```bash
$ docker build -t approx . && docker run -it -v "$(pwd):/app" approx
/app # ./build.sh
-- Configuring done
-- Generating done
-- Build files have been written to: /app
[100%] Built target autodiff
/app # ./autodiff --help
benchmark [--benchmark_list_tests={true|false}]
          [--benchmark_filter=<regex>]
          [--benchmark_min_time=<min_time>]
          [--benchmark_repetitions=<num_repetitions>]
          [--benchmark_report_aggregates_only={true|false}
          [--benchmark_format=<console|json|csv>]
          [--benchmark_out=<filename>]
          [--benchmark_out_format=<json|console|csv>]
          [--benchmark_color={auto|true|false}]
          [--benchmark_counters_tabular={true|false}]
          [--v=<verbosity>]
/app #
```

## Example code

```C++
// Define a function
// pow(ln(x), pi/x)
typedef Pow<Ln<Input<float>>, Div<PI, Input<float>>> fn;

// Get the first derivative:
typedef fn::grad fngrad;

// Get the nth derivative:
constexpr std::size_t n = 10;
typedef derive<fn, n>::value fndn;

// Evaluate some function at a point
float f(float x) {
    return fn::grad::grad::eval(x);
}
```

