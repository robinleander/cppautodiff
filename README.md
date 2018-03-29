# C++ TMP Autodiff

Automatic differentiation at compiletime!

## Usage

```bash
docker build -t approx . && docker run -it approx
```

## Example code

```C++
// Define a function
// pow(ln(x), pi/x)
typedef Pow<Ln<Input<float>>, Div<PI, Input<float>>> fn;

// Get the derivative:
typedef fn::grad fngrad;

// Evaluate at a point
float f(float x) {
    fn::grad::eval(x);
}
```

