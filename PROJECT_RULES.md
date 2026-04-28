# DR4GM Project Rules

Authoritative checklist used by the release workflow's audit step
(see `CLAUDE.md` → "Release Workflow" → step 4).

## Rules

1. **All tests must pass.** A release cannot ship while any test is
   failing. The audit step must run the project's tests and treat any
   failure as a blocking finding.
