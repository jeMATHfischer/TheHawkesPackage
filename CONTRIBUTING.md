# Contributing

## Setup

The package uses a `src/` layout, which means **you must install it before the
tests can import it**. This is deliberate: it stops `pytest` from silently
importing the source tree instead of the installed package, so what CI tests is
what users get. Do not work around it with a `sys.path` hack in `conftest.py`.

```bash
python -m venv .venv
. .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
pre-commit install
```

Python 3.10 or newer.

## Everyday commands

```bash
pytest                       # everything
pytest -m "not slow"         # the fast suite, ~25 s — what the CI matrix runs
pytest --cov                 # with the 90% coverage gate
ruff check . && ruff format --check .
mypy
sphinx-build -W --keep-going -b html docs docs/_build/html
```

## Tests

- **Never call `np.random.seed`.** Pass `rng=` to the process instead. The suite
  runs under `pytest-randomly`, which shuffles test order and reseeds the global
  streams between tests — a test that depends on global RNG state will fail
  intermittently, which is the point.
- `filterwarnings = ["error"]` is on. Any unexpected warning fails the suite;
  intentional deprecation paths must be wrapped in `pytest.warns`.
- Statistical tests use **fixed** seeds and loose thresholds (KS p-value floor
  `1e-3`, not `0.05`). If you add one, run it across seeds 0–20 locally and pin
  a few that pass comfortably. They answer "is this broken", not "is this
  publication-grade".
- Anything that takes more than a second or so gets `@pytest.mark.slow`.
- Exact-value reproducibility may only be asserted for the three temporal
  classes. The spatio-temporal path branches on floating-point comparisons whose
  last bits vary across SciPy and BLAS builds; a different branch consumes a
  different number of draws and the streams then diverge completely. Assert
  distributional properties and structural invariants there instead.

## Changing the simulation

Every process class implements two hooks, `_conditional_intensity` (or
`_integrated_intensity`) and `_upper_bound`, and inherits the Ogata thinning
loop from `hawkes_package.base`. If you touch a bound, the test that matters is
`tests/statistical/test_thinning_invariant.py`: thinning is only correct while
`M >= lambda`, and a bound that is too *tight* degrades the process to Poisson
silently rather than raising. Seven such defects shipped undetected during
0.2.0 -- and every one of them lived in a configuration the harness did not then
cover. **Add a case to that file before changing how a bound is computed**,
not after.

Do not define an intensity accessor separately from the hook the simulator thins
against — that is exactly how `ExponentialHawkes` came to plot a curve a
constant `mu` below its own intensity.

## Deprecations

Route everything through `hawkes_package._deprecation`. A renamed method is one
line:

```python
propagate_by_amount = DeprecatedAlias("simulate")
```

Renamed module-level names go through `deprecated_module_getattr`. Add the
old/new pair to `RENAMES` in `tests/test_deprecations.py` — `docs/migration.md`
is written from that list, so the documentation cannot drift from behaviour.

## Pull requests

`master` requires all fourteen checks -- `Lint and type check`, the ten
`Test <version> on <os>` jobs, `Coverage`, `Build and verify the wheel` and
`Build the documentation` -- plus linear history. Those are the workflows'
`name:` values: GitHub matches required checks on the display name, not the
job id.

Branch from `master`, keep formatting changes in their own commit, and add a
`CHANGELOG.md` entry under `[Unreleased]` for anything user-visible.

A behaviour change that alters previously produced numbers **without raising**
is the worst failure mode in a scientific package. Those go under `### Changed`
or `### Fixed` in the changelog and in `docs/migration.md`, in terms a user can
act on.

## Releasing

1. Move the `[Unreleased]` entries under a new version heading in `CHANGELOG.md`.
2. Bump `__version__` in `src/hawkes_package/__init__.py` — the single source of
   truth; `pyproject.toml` reads it.
3. Tag `vX.Y.Z` and push. The release workflow verifies the tag matches
   `__version__`, builds, and publishes to PyPI via trusted publishing.

Before the *first* release, a PyPI pending publisher must already exist for
project `the-hawkes-package`, owner `jeMATHfischer`, repo `TheHawkesPackage`,
workflow `release.yml`, environment `pypi`. Trusted publishing to a project that
does not yet exist is rejected without one.

A second pending publisher on **test.pypi.org** (a separate account) with
environment `testpypi` drives the rehearsal: tag `vX.Y.ZrcN` and the release
workflow routes the upload there instead, marking the GitHub release as a
prerelease. Bump `__version__` to the rc, tag, verify, then restore it.

Both publishers are bound to the **filename** `release.yml`. Renaming that
workflow silently breaks every future release until the publishers are updated
on both indexes.

A published version can never be re-uploaded, on either index. A broken release
is yanked and superseded, never replaced.
