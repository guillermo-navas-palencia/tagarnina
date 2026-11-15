# Contributing to tagarnina

Thanks for your interest in tagarnina. The process is intentionally light; please follow the steps below.

## Getting started

```bash
git clone https://github.com/<your-username>/tagarnina.git
cd tagarnina
git remote add upstream https://github.com/guillermo-navas-palencia/tagarnina.git
git fetch upstream
git checkout -b my-feature upstream/dev
python -m pip install -e ".[dev]"
```

tagarnina requires Python 3.12 or newer. Always work on a feature branch, never on `dev` or `main`.

## Submitting a pull request

- Open PRs against the **`dev`** branch. CI runs on `dev` and `main`.
- Keep each PR focused on one logical change.
- Reference related issues with keywords such as `Fixes #123`.
- Run the checks below locally before pushing.
- Describe what changed and why, and call out any behavior change.
- Update docstrings and the relevant pages in `docs/source/technical/` when behavior changes.
- Add tests for new behavior or bug fixes.

## Development checks

```bash
python -m ruff check tagarnina tests
python -m ruff format --check tagarnina tests
python -m mypy tagarnina
python -m pytest --cov=tagarnina --cov-report=term-missing
```

CI runs the same commands. mypy is currently non-blocking, but new code should still type-check cleanly.

## Coding guidelines

- **Formatting and linting:** Ruff handles both. Line length is 88. Run `ruff format` before committing. Do not add `# noqa` or per-file ignores without a reason in the PR.
- **Imports:** Ruff's isort rules (`I`) group and sort them: standard library, third-party, then local. Use relative imports inside the `tagarnina` package and absolute imports in `tests/`. Avoid wildcard imports.
- **Constructors:** Use keyword-only arguments (`*`) when a class takes more than a couple of parameters.
- **Comments:** Comment only code that needs clarification. Let the names and docstrings carry the rest.

## Naming

- **Classes:** `CapWords`. Constraint classes end in `Constraint`. Algorithm classes use their established acronym (`CRFE`, `CRFECV`).
- **Functions, methods, variables:** `snake_case`. Start method names with a verb (`build_constraint`, `feasibility_check`).
- **Private helpers:** Prefix with `_` (for example, `_build_extra`).
- **Scikit-learn conventions:** Use `X` for the feature matrix and `y` for targets. Use the `n_` prefix for counts (`n_features_to_select`, `n_variables`).
- **Parameters:** Prefer full words. Abbreviate only when the abbreviation is established (`X`, `y`, `n_`, `min_`/`max_`).
- **Consistency:** Match the names used by scikit-learn for analogous parameters.

## Docstrings

- Use **numpydoc** format, as in `CPSolver` and `CRFE`.
- Section order: Parameters, Returns, Raises, See Also, Notes, Examples.
- Parameter entries use `name : type, default=value`. Use `{"a", "b"}` for fixed choices and `X or None` for optional values.
- Every public class, method, and function needs a docstring. Private helpers need one only when their behavior is not obvious.
- New user-facing features should come with an example in `examples/`.

## Typing

- Annotate all new function signatures, including return types.
- Use built-in generics (`list[str]`, `dict[str, int]`) and `X | None` instead of `Optional`.
- Use `typing.Self` for methods that return the instance.
- Use `numpy.typing` (`npt`) for array-like arguments.
- Avoid `# type: ignore`. If it is unavoidable, add the specific error code and a short reason.
- Check types with `python -m mypy tagarnina`.

## Testing

- Tests live in `tests/` and are named `test_<module>.py`. Add new cases to the matching file.
- Use `@pytest.mark.parametrize` with `pytest.raises(..., match=...)` for input validation.
- Cover edge cases, and make sure a new test fails without the change it covers.
- Do not write tests that only check defaults, constants, re-exports, or trivial accessors.
- Keep tests deterministic. The solver uses `random_seed=0` and `num_workers=1` by default. Only use more workers when the test does not depend on the exact solution.

## Documentation

- Keep docstrings and `docs/source/technical/` in sync with behavior.
- Build the docs from `docs/` with the `docs` extra installed: `python -m pip install -e ".[docs]"`, then `make html` (or `make.bat html` on Windows).

## Reporting bugs and requesting features

Open a GitHub issue with a minimal reproducible example, the tagarnina and Python versions, the operating system, and the full traceback.

## License

By contributing, you agree that your contributions are licensed under the Apache-2.0 license in [LICENSE](LICENSE).