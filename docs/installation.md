# Installation

```bash
pip install the-hawkes-package
```

The distribution is named `the-hawkes-package`; the import name is
`hawkes_package`:

```python
import hawkes_package as hp
```

## Requirements

Python 3.10 or newer, with NumPy ≥ 1.22 and SciPy ≥ 1.8. Both are pulled in
automatically. The package is pure Python and ships type information
(PEP 561), so downstream `mypy` picks it up with no extra stubs.

## From source

```bash
git clone https://github.com/jeMATHfischer/TheHawkesPackage
cd TheHawkesPackage
python -m pip install -e ".[dev]"
```

The `dev` extra adds the test, docs, lint and type-checking tools. The project
uses a `src/` layout, so an install is required before the tests can import the
package — see [contributing](contributing.md).

## Upgrading from `TheHawkesPackage`

`import TheHawkesPackage` still works and forwards to identical objects, but
emits a `DeprecationWarning`; the shim is removed in 0.4.0. Several behaviours
changed in 0.2.0 in ways that alter previously produced numbers without raising
— read [migration](migration.md) before re-running old scripts.
