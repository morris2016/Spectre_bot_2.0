# Development Guidelines

## Coding Style
- Follow **PEP 8** conventions.
- The project uses **flake8** for linting. Configuration is in `setup.cfg` with a
  max line length of 150 characters.
- Run `flake8` or `pre-commit` to check code style before committing.

## Running Tests
- Use `pytest` to execute the test suite:
  ```bash
  pytest -q
  ```
- Ensure tests pass locally before pushing changes.

## Branching and Commit Conventions
- Start feature work from the `main` branch using short descriptive branch
  names (e.g., `feature/awesome-improvement`).
- Write concise commit messages in the imperative mood ("Add feature" rather
  than "Added" or "Adds").
- Commit small, logically grouped changes for easier review.

## Pre-commit Hooks
- Install hooks with `pre-commit install`.
- Run `pre-commit run --files <file1> <file2>` on changed files before
  committing, or `pre-commit run --all-files` to check the entire repository.
- The hooks enforce formatting (end-of-file fixer) and run `flake8`.

## Make sure there are no conflicts in the code or no characters are added to create merge conflicts.
