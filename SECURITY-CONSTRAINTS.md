# Temporary backend security constraints

The `[tool.uv].constraint-dependencies` block in the root `pyproject.toml` is a
**temporary security workaround**, not a list of new application dependencies.
It prevents uv from retaining or selecting vulnerable transitive versions while
upstream dependency ranges still allow them. These are minimum-version
constraints, not exact pins or overrides: upstream compatibility requirements
remain enforced by the resolver.

Scope: the root/backend project only. The frontend has its own lockfile.
These uv-specific constraints are **not published as pip package requirements**;
use the checked-in `uv.lock` for reproducible deployment. They do not guarantee
that an independent pip source install selects the same transitive versions.

## Tracked changes

Baseline: `origin/main` at `da6a1aa` (2026-10-18).
Alert numbers below refer to this repository's GitHub Dependabot alerts.

| Temporary constraint | Previous lock | Updated lock | Dependabot alert numbers |
| --- | --- | --- | --- |
| `pyjwt>=2.15.0` | 2.13.0 | 2.15.1 | #412, #396, #391, #387, #386, #385 (critical), #384, #383, #382, #375, #374, #373, #372, #371 |
| `urllib3>=2.8.0` | 2.7.0 | 2.8.0 | #390, #389, #388 |
| `cryptography>=50.0.0` | 48.0.1 | 50.0.2 | #370, #367, #333 |
| `pillow>=12.3.0` | 12.2.0 | 12.3.0 | #302, #301, #300, #299, #298, #297, #296, #295, #294, #293, #292, #291, #278 |
| `pyasn1>=0.6.4` | 0.6.3 | 0.6.4 | #323, #308, #307 |
| `h2>=4.4.1` | 4.3.0 | 4.4.1 | #334 |
| `hpack>=4.2.0` | 4.1.0 | 4.2.0 | #364 |
| `mako>=1.4.2` | 1.3.12 | 1.4.3 | #411 |
| `multidict>=6.9.1` | 6.7.1 | 7.1.0 | #410 |

Browse alerts in this repository's **Security → Dependabot alerts** page
(requires repository security access).

Alert #396 has no first-patched version recorded by GitHub. The updated PyJWT
version is outside its reported affected range (`>=2.11.0,<=2.13.0`); this is a
version-range check, not an independent verification of that advisory's fix.

## Removal checklist for a later release

Review these constraints during future dependency upgrades. Do **not** remove
all of them just because this PR's tests pass or GitHub closes the old alerts:
the constraints themselves may be what keeps resolution safe.

For each package:

1. Check the current advisories and upstream dependency requirements. Confirm
   that the resolved upstream requirements now enforce at least the safe floor
   (or that the package is no longer a dependency).
2. Remove only that package's constraint from `pyproject.toml`.
3. Regenerate `uv.lock` and also test fresh resolution in a disposable checkout
   without the existing lockfile. Neither path should select an affected version
   on any supported Python/platform branch.
4. Install with `uv sync --locked`, run `make test`, and replay recorded tests
   with `VCR_RECORD_MODE=none make test-vcr`. Check any major-version upgrades
   against the upstream release notes and relevant application behavior.
5. Recheck the resulting lockfile against current Dependabot advisories. Passing
   tests alone do not establish that a dependency is secure.
6. Remove/update the corresponding row here in the same PR. Once all constraints
   are obsolete, remove the `[tool.uv]` block, its temporary comments, and this
   document together.

## Validation for this update

- `uv sync --locked` succeeds without dependency overrides.
- `make test`: 83 passed, 94% coverage.
- Full existing suite with replay-only VCR: 86 passed, 1 skipped (the live
  streaming test requires a running backend).
- All 40 open root-lockfile alerts at review time were compared with the updated
  lockfile; no locked version remains in their reported affected ranges.
- This does not establish exploitability or cover live external service behavior,
  unreported vulnerabilities, or packages outside the root lockfile.
