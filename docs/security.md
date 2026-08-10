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
| TLS/SSL material (`nginx/ssl/`) | ✅ empty |
| Connection strings | ⚠️ dev-default credentials only (see below) |

**Verdict:** No live secrets are committed to the repository or its history.

---

## 2. Findings & remediation

| # | Location | Finding | Severity | Status |
|---|----------|---------|----------|--------|
| 1 | `api/main.py` | `DATABASE_URL` hardcoded with dev creds in source | 🟠 medium | ✅ **Fixed** — now env-driven (`os.environ.get("DATABASE_URL", <dev-default>)`) with a startup warning when the dev default is in use |
| 2 | `k8s/production-deployment.yaml` | Plaintext `Secret` committed (values are `CHANGE_THIS_*` placeholders) | 🟠 medium (pattern risk) | ✅ **Hardened** — added security comment; production must use SealedSecrets / External Secrets Operator (values never in git) |
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

## 5. Local quick check

```bash
# Scan working tree + history for secrets
gitleaks detect --source . --redact --no-banner

# Or with Docker (daemon running)
docker run --rm -v "$PWD:/repo" -w /repo gitleaks/gitleaks:latest \
  detect --source /repo --redact --no-banner
```

---

## 6. Related files

- `.gitlab-ci.yml` — `scan:secrets` blocking gate
- `.github/workflows/ci.yml` — `secret-scan` blocking gate
- `.gitignore` — excludes `.env`, `.*` AppleDouble metadata, CI artifacts
- `k8s/production-deployment.yaml` — placeholder Secret + hardening notes
- `docs/ci-cd.md` — pipeline runbook incl. security scanning setup
