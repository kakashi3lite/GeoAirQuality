# Security — Secrets Audit & Hardening

*Last audit: 2026-08-10 | Tooling: gitleaks v8.23.1 + custom pattern scan*

---

## 1. Audit summary

| Scope | Result |
|-------|--------|
| **gitleaks full-history scan** (18 commits, 16.5 MB) | ✅ **0 leaks** |
| Custom regex scan (private keys, AWS, GH tokens, OpenAI, Google, Slack) | ✅ 0 matches (working tree + history) |
| Jupyter notebook (`GeoAirQuality.ipynb`) | ✅ 0 matches |
| `.env` files | ✅ none present in tree; `.env` is gitignored |
| TLS/SSL material (`deploy/nginx/ssl/`) | ✅ empty |
| Connection strings | ⚠️ dev-default credentials only (see below) |

**Verdict:** No live secrets are committed to the repository or its history.

---

## 2. Findings & remediation

| # | Location | Finding | Severity | Status |
|---|----------|---------|----------|--------|
| 1 | `api/main.py` | `DATABASE_URL` hardcoded with dev creds in source | 🟠 medium | ✅ **Fixed** — now env-driven (`os.environ.get("DATABASE_URL", <dev-default>)`) with a startup warning when the dev default is in use |
| 2 | `deploy/k8s/production-deployment.yaml` | Plaintext `Secret` committed (values are `CHANGE_THIS_*` placeholders) | 🟠 medium (pattern risk) | ✅ **Hardened** — added security comment; production must use SealedSecrets / External Secrets Operator (values never in git) |
| 3 | `setup-dev.sh` | `SECRET_KEY` hardcoded as `dev-secret-key-change-in-production` | 🟡 low (dev-only) | ✅ **Fixed** — now generates a random key (`openssl rand -hex 32`) with a safe fallback |
| 4 | `docs/deployment.md` | `kubectl create secret` example with `user:pass@host` / `secure-password` | 🟡 low | Placeholder example — acceptable; not treated as real |
| 5 | `docker-compose.yml`, `api/alembic.ini`, `api/migrations/env.py` | `geoair_user:geoair_pass` dev defaults | 🟡 low | Standard local-dev pattern; Docker Compose is development-only |
| 6 | CI/CD (GitLab + GitHub) | Integration tests use `geoair_user:geoair_pass` | 🟡 low | Ephemeral per-run CI service credentials — never exposed externally |

All remaining `geoair_pass` values are **documented local-development defaults**,
never used in production (production reads from k8s Secrets via environment
variables).

---

## 3. Prevention (now enforced by CI)

**Secret scanning is a BLOCKING gate** in both pipelines (baseline verified
clean, so it cannot block legitimate work):

- **GitLab** — `scan:secrets` job (gitleaks via docker, `GIT_DEPTH: 0` for
  full history) on every MR and `main`.
- **GitHub** — `secret-scan` job (`gitleaks/gitleaks-action@v2`,
  `fetch-depth: 0`) on every PR and push.

If gitleaks flags a false positive, add the pattern to the allowlist rather
than disabling the job.

---

## 4. Secret handling policy

1. **Never commit credentials.** API keys, passwords, tokens, private keys —
   all go in environment variables, k8s Secrets, or a secrets manager.
2. **Production secrets are git-excluded by construction.** Use
   [SealedSecrets](https://github.com/bitnami-labs/sealed-secrets) or the
   [External Secrets Operator](https://external-secrets.io) so k8s manifests
   in git contain only placeholders.
3. **`.env` files stay local** — gitignored; `setup-dev.sh` generates them
   with a random `SECRET_KEY`.
4. **CI secret variables are masked/protected** — kubeconfigs and registry
   credentials are referenced as `$VAR` / `secrets.*`, never inlined.
5. **Rotation**: if a secret is ever committed, treat it as compromised —
   rotate immediately, then purge history (BFG) or rewrite the branch and
   force-push.

---

## 6. Secrets inventory & where they live (2026-08-11)

| Secret | Purpose | Local | GitLab CI/CD | GitHub Actions | Cloudflare Pages | k8s Secret |
|--------|---------|-------|--------------|----------------|------------------|------------|
| `DATABASE_URL` | PostGIS connection | `api/.env` | — | — | — | ✅ `geoairquality-secrets` |
| `POSTGRES_PASSWORD` | Postgres superuser pw | compose dev only | — | — | — | ✅ `geoairquality-secrets` |
| `REDIS_PASSWORD` | Redis auth (optional) | — | — | — | — | ✅ `geoairquality-secrets` |
| `SECRET_KEY` | JWT/auth signing | `api/.env` | ✅ masked+protected+hidden | ✅ | — | ✅ `geoairquality-secrets` |
| `AIRNOW_API_KEY` | EPA AirNow news | `api/.env` | ✅ (optional) | ✅ (optional) | — | ✅ `geoairquality-secrets` |
| `NEWSAPI_KEY` | NewsAPI news | `api/.env` | ✅ (optional) | ✅ (optional) | — | ✅ `geoairquality-secrets` |
| `CLOUDFLARE_API_TOKEN` | Pages deploys in CI | — | ✅ masked+protected+hidden | ✅ | — | — |
| `CLOUDFLARE_ACCOUNT_ID` | CF account (non-secret ID) | — | ✅ var | ✅ var | — | — |
| `CLOUDFLARE_PAGES_PROJECT` | Pages project (non-secret) | — | ✅ var | ✅ var | ✅ project name | — |
| `API_ORIGIN` | Backend URL for the edge proxy | `frontend/.dev.vars` | — | — | ✅ binding | — |
| `KUBE_CONFIG_STAGING` | base64 kubeconfig | — | ✅ masked+protected+hidden | ✅ | — | — |
| `KUBE_CONFIG_PRODUCTION` | base64 kubeconfig | — | ✅ masked+protected+hidden | ✅ | — | — |
| `STAGING_NAMESPACE` | k8s namespace (non-secret) | — | ✅ var | ✅ var | — | — |
| `PRODUCTION_NAMESPACE` | k8s namespace (non-secret) | — | ✅ var | ✅ var | — | — |

**Already wired (this pass):** the four non-secret vars and a generated
`SECRET_KEY` are set on **GitLab** (`kakashi3litez/GeoAirQuality`) and
**GitHub** (`kakashi3lite/GeoAirQuality`); the Cloudflare Pages project
`geoairquality-breathe` exists with `API_ORIGIN` bound.

**You still need to provide** (values are real secrets — run the interactive
helper and type them directly into the terminal; they are never sent through
chat):

```bash
./scripts/setup-ci-secrets.sh        # prompts for each, silent input
./scripts/setup-ci-secrets.sh --check   # see what is set
```

The script wires `CLOUDFLARE_API_TOKEN`, both `KUBE_CONFIG_*`, `AIRNOW_API_KEY`
and `NEWSAPI_KEY` into GitLab (masked+protected+hidden) and GitHub. Then fill
the k8s Secret (SealedSecrets/ESO) with the same DB/Redis/news values.

---

## 7. Local quick check

```bash
# Scan working tree + history for secrets
gitleaks detect --source . --redact --no-banner

# Or with Docker (daemon running)
docker run --rm -v "$PWD:/repo" -w /repo gitleaks/gitleaks:latest \
  detect --source /repo --redact --no-banner
```

---

## 8. Related files

- `.gitlab-ci.yml` — `scan:secrets` blocking gate
- `.github/workflows/ci.yml` — `secret-scan` blocking gate
- `.gitignore` — excludes `.env`, `.*` AppleDouble metadata, CI artifacts
- `deploy/k8s/production-deployment.yaml` — placeholder Secret + hardening notes
- `scripts/setup-ci-secrets.sh` — interactive wiring of the real secrets
- `docs/ci-cd.md` — pipeline runbook incl. security scanning setup
