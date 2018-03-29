# C++ TMP Autodiff

Automatic differentiation at compiletime!

## Usage

```bash
docker build -t approx . && docker run -it -v "$(pwd):/app" approx
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

