# Deployment

`.github/workflows/release.yml` publishes `vocalbin` to PyPI when a `v*` tag is
pushed. Authentication uses PyPI Trusted Publishing (OIDC); no API token is
stored in GitHub.

## One-time setup

1. Create a GitHub environment named `pypi` under **Settings > Environments**.
   Optionally require approval and restrict deployments to tags matching `v*`.
2. Add a GitHub Trusted Publisher to the `vocalbin` project on PyPI:

   | Setting | Value |
   | --- | --- |
   | Owner | `mathisarends` |
   | Repository | `vocalbin` |
   | Workflow | `release.yml` |
   | Environment | `pypi` |

If the PyPI project does not exist yet, create a pending publisher at
<https://pypi.org/manage/account/publishing/> using the same values.

## Release

Set and commit the release version:

```bash
uv version 0.2.0
uv run pytest
git add pyproject.toml uv.lock
git commit -m "Release 0.2.0"
git push origin main
```

After the `Tests` workflow succeeds on `main`, create the matching tag:

```bash
version="$(uv version --short)"
git tag -a "v$version" -m "Release v$version"
git push origin "v$version"
```

The release workflow then:

1. Confirms that the tagged commit belongs to `main` and the tag matches the
   version in `pyproject.toml`.
2. Installs the locked dependencies and runs all tests with 100% statement and
   branch coverage.
3. Builds the wheel and source distribution, checks their metadata, and installs
   the wheel in a clean environment.
4. Publishes the verified artifacts to PyPI through OIDC.
5. Creates a GitHub Release with generated notes and both distributions attached.

## Failures

- PyPI versions are immutable. Fix a broken release with a new version; yank the
  old version if necessary.
- An OIDC error usually means the owner, repository, workflow, or environment
  differs from the PyPI Trusted Publisher configuration.
- A mismatched tag is rejected before publishing. Create a tag matching
  `uv version --short`.
