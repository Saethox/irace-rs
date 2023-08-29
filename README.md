# irace-rs

Rust bindings for [`irace`](https://github.com/MLopez-Ibanez/irace): Iterated Racing for Automatic Algorithm
Configuration.

For more information on `irace`, see its [documentation](https://mlopez-ibanez.github.io/irace/index.html) directly.

## Disclaimer

Strictly speaking, this is not a direct wrapper of the `irace` R package, but of the Python wrapper `iracepy-tiny`.
The reason for this is that there is currently no way to call an R function from Rust with a callback function as
argument that is also defined in Rust without separately compiling the callback function and R function call.
While [`extendr`](https://github.com/extendr/extendr) could be used to implement the compile-twice approach, this
restricts this wrapper to certain purposes.
Specifically the build process gets significantly more complex, as updates to the callback function need to be compiled
and made available to the R interpreter again.

The necessary functionality is available in [`PyO3`](https://github.com/PyO3/pyo3), which is why this approach was
chosen, despite having some overhead by running an additional instance of the Python interpreter.
Another benefit of this is that Python's multiprocessing can be used to run separate instances of `irace` in parallel.

## Getting Started

### Requirements

- [The Rust Programming Language](https://rust-lang.org)
- Either `gcc` or `clang`
- [`irace`](https://mlopez-ibanez.github.io/irace/#installing-the-irace-package) R package
- [`iracepy-tiny`](https://github.com/Saethox/iracepy-tiny) Python package

### Installation

Add the following to your `Cargo.toml`:

```toml
[dependencies]
irace-rs = { git = "https://github.com/Saethox/irace-rs" }
```

## Restrictions

Note that because of FFI, the target runner and instance need to be `Send + 'static` even if no "real" multithreading
happens.

## Examples

See the [examples](./examples) directory.

## MAHF

This wrapper is primarily written for usage with [`mahf`](https://github.com/mahf-opt/mahf), but works with arbitrary
target algorithms.

## License

This project is licensed under
the [GNU General Public License v3.0](https://github.com/mahf-opt/mahf/blob/master/LICENSE).