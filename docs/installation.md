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

The `TheHawkesPackage` import shim was **removed in 0.4.0**, after two releases
of warning: `import TheHawkesPackage` is now an `ImportError`, and the import
name is `hawkes_package`. 0.4.0 also renamed `Events` to `events` and `Sim_num`
to `n_simulated`, which still work and warn, and dropped the frozen legacy
spatio-temporal class.

Several behaviours changed in 0.2.0 in ways that alter previously produced
numbers without raising — read [migration](migration.md) before re-running old
scripts. 0.3.0 and 0.4.0 leave those numbers alone, but 0.3.0 changed what a
custom `SpatialDomain` subclass must satisfy; the same page says how.
