# CI/CD — Design & Operations Runbook

GeoAirQuality ships a **dual-platform** pipeline: `.gitlab-ci.yml` (GitLab)
and `.github/workflows/ci.yml` (GitHub Actions). They are deliberately
mirrored so the repo is CI-ready on both platforms and a team can migrate
between them without redesign.

---

## 1. Pipeline stages

```
┌─────────┬─────────┬─────────┬─────────┬──────────┬─────────┐
│  LINT   │  TEST   │  BUILD  │  SCAN   │ PUBLISH  │ DEPLOY  │
├─────────┼─────────┼─────────┼─────────┼──────────┼─────────┤
│ enforced│ unit    │ api img │ trivy   │ GHCR /   │ staging │
│ (block) │ (block) │ pipeline│ bandit  │ registry │ (auto)  │
│ legacy  │ integra │ img     │ pip-    │ by SHA   │ prod    │
│ (soft)  │ tion    │         │ audit   │          │ (manual)│
└─────────┴─────────┴─────────┴─────────┴──────────┴─────────┘
```

| Stage | Gate | Why |
|-------|------|-----|
| `lint:enforced` | 🔴 blocking | Black + flake8 on `api/services/` + `tests/` (code we own) |
| `lint:legacy` | 🟡 soft | `main.py`/`models.py`/`cache.py`/`data-pipeline/` carry formatting debt |
| `test:unit` | 🔴 blocking | Full pytest suite + coverage + JUnit report |
| `test:integration` | 🔴 blocking | Runs `alembic upgrade head` + end-to-end endpoint test against **real PostGIS + Redis** |
| `build:*` | 🔴 blocking | Docker images tagged with `$CI_COMMIT_SHA` — the exact artifact tested |
| `scan:*` | 🟡 soft | Trivy (images), bandit (Python), pip-audit (deps) |
| `publish:images` | — | Pushes SHA + `latest` to registry (main + milestone tags only) |
| `deploy:staging` | auto | On `main` merge, with `rollout status` wait |
| `deploy:production` | 👤 manual | Protected branch + environment approval gate |

**Gate philosophy:** *"You may not ship untested code, but you may ship
legacy lint debt."* Blocking gates protect correctness; soft gates surface
risk without stopping delivery.

---

## 2. Why these choices (decisions & tradeoffs)

### Meter-accurate integration tests
The unit suite mocks the DB. The integration stage runs the **real**
PostGIS + Redis services and applies migrations. This is where migration
SQL mistakes are caught (e.g. the `(location::geography)` GiST expression
indexes in migration 003).

### Build once, scan, publish by SHA
Images are built in `build`, scanned in `scan`, and pushed in `publish`
**from the same build artifacts** — never rebuilt at publish time. Deploys
reference `$CI_COMMIT_SHA`, so what's running in staging is byte-for-byte
what passed tests and scans.

### Docker-in-Docker (DIND)
Used for image builds (works on GitLab.com shared runners). For stricter
self-hosted runners, replace with **kaniko** (no privileged daemon):

```yaml
build:api:
  image:
    name: gcr.io/kaniko-project/executor:latest
    entrypoint: [""]
  script:
    - /kaniko/executor --context $CI_PROJECT_DIR --dockerfile $CI_PROJECT_DIR/api/Dockerfile
      --destination "$CI_REGISTRY_IMAGE/api:$CI_COMMIT_SHA"
```

### Soft security gates (first week)
`scan:*` jobs use `allow_failure: true` / `exit-code: 0` and `|| true` so
they **report** vulnerabilities without blocking. After the first week of
green reports, flip them to blocking:
- GitLab: remove `allow_failure: true` and add `TRIVY_FAIL_ON_HIGH: "true"`
- GitHub: set `exit-code: "1"` in the Trivy step and drop `|| true`

### Deploy safety
- `deploy:staging` auto-runs on `main` and waits for `rollout status`.
- `deploy:production` is `when: manual` (GitLab) / `workflow_dispatch`
  (GitHub) and sits behind the `production` environment, which enforces
  **approval rules** and (in GitLab) is restricted to protected branches.
- Every deploy uses `kubectl set image` to a SHA-tagged image — instant
  rollback = `kubectl rollout undo`.

---

## 3. GitLab setup (one-time)

1. **Mirror from GitHub** (Settings → Repository → Mirroring repositories)
   - GitLab → GitHub (pull) keeps GitLab in sync with this repo.
   - Enable *Keep divergent refs* and only mirror `main`.
2. **Enable Container Registry** (Settings → Packages & Registries).
3. **CI/CD variables** (Settings → CI/CD → Variables):
   | Variable | Example | Protected? |
   |----------|---------|-----------|
   | `KUBE_CONFIG_STAGING` | `base64 -w0 ~/.kube/config` | yes |
   | `KUBE_CONFIG_PRODUCTION` | `base64 -w0 ~/.kube/prod-config` | yes (masked) |
   | `STAGING_NAMESPACE` | `geoairquality-staging` | — |
   | `PRODUCTION_NAMESPACE` | `geoairquality-prod` | — |
   - `CI_REGISTRY_USER`/`CI_REGISTRY_PASSWORD`/`CI_REGISTRY` are auto-injected.
4. **Environment approvals** (Deployments → Environments → `production`)
   → *Enable approval rules* → required approvers ≥ 1.

## 4. GitHub Actions setup (one-time)

1. **Secrets** (Settings → Secrets → Actions): `KUBE_CONFIG_STAGING`,
   `KUBE_CONFIG_PRODUCTION` (base64).
2. **Variables**: `STAGING_NAMESPACE`, `PRODUCTION_NAMESPACE`.
3. **Environments** (Settings → Environments): create `staging` and
   `production`; on `production` add required reviewers for the approval
   gate.
4. **GHCR**: images publish to `ghcr.io/<owner>/GeoAirQuality/*` using the
   built-in `GITHUB_TOKEN` (Packages → ensure the token has write access).

---

## 5. Local parity

CI and local development share the same commands via `Makefile`:

```bash
make lint              # what CI enforces
make test              # unit tests + coverage (no services needed)
make test-integration  # needs PostGIS + Redis running (or docker compose)
make build             # local images
make scan              # bandit + pip-audit
```

---

## 6. Lint enforcement path (tracked debt)

Current state: `api/services/` and `tests/` are Black+flake8 clean and
**blocking**. `api/main.py`, `api/models.py`, `api/cache.py` and
`data-pipeline/` are non-blocking until a formatting pass lands.

**To enforce everywhere:**
1. `make lint-fix` (black) then manually fix flake8 findings.
2. Flip `lint:legacy` `allow_failure: false` (GitLab) / remove `|| true`
   (GitHub).
3. Delete this section.

---

## 7. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `test:integration` fails on `alembic upgrade head` | migration SQL invalid against real PostGIS | inspect the failing migration, fix SQL, re-run |
| Build job: `Cannot connect to the Docker daemon` | runner doesn't allow privileged DIND | use kaniko (above) or enable privileged runner |
| `publish` fails: `denied: requested access` | registry auth | confirm Container Registry enabled / GHCR token write |
| `deploy` fails: `Unable to connect to the server` | stale/empty kubeconfig var | re-base64 the kubeconfig, confirm cluster reachable |
| `scan` reports many findings | base image CVEs | pin a patched base (python:3.11-slim is patched monthly), add `--ignore-unfixed` |
| flake8/black crash with *"null bytes"* | macOS `._*` AppleDouble files on external volumes | remove them (`find . -name '._*' -delete`), they are gitignored |
