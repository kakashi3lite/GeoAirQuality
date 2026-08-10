#!/usr/bin/env bash
# =============================================================================
# GeoAirQuality — CI/CD Secrets Setup
# =============================================================================
# Wires the real secrets the pipelines need into BOTH GitLab and GitHub.
#
# SECURITY MODEL:
#   * You type/paste each value directly into the terminal (silent read).
#     Nothing is echoed, and values are piped to the CLIs via STDIN so they
#     never appear in argv, shell history, or this conversation.
#   * The four non-secret vars (CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_PAGES_PROJECT,
#     STAGING/PRODUCTION_NAMESPACE) are already set — this script only handles
#     real secrets.
#   * GitLab secrets are created as MASKED + PROTECTED + HIDDEN.
#   * You can pre-seed any value via env (e.g. AIRNOW_API_KEY=abc) to skip the
#     prompt — useful for scripting.
#
# Usage:
#   ./scripts/setup-ci-secrets.sh            # interactive prompts
#   ./scripts/setup-ci-secrets.sh --check    # report what is already set
# =============================================================================
set -uo pipefail

GITLAB_REPO="${GITLAB_REPO:-kakashi3litez/GeoAirQuality}"
GITHUB_REPO="${GITHUB_REPO:-kakashi3lite/GeoAirQuality}"

command -v glab >/dev/null || { echo "✗ glab not installed (brew install glab)"; exit 1; }
command -v gh  >/dev/null || { echo "✗ gh not installed (brew install gh)"; exit 1; }

# ---------------------------------------------------------------------------
ask() { # ask NAME "HUMAN LABEL"
  local var="$1" label="$2"
  # skip prompt if already provided via environment
  if [ -n "${!var:-}" ]; then return; fi
  read -rsp "  $label: " "$var"
  echo
}

set_gitlab_secret() { # name label
  local name="$1" label="$2"
  if [ -n "${!name:-}" ]; then
    printf '%s' "${!name}" | glab variable set "$name" --masked --protected --hidden -R "$GITLAB_REPO" >/dev/null 2>&1 \
      && echo "  ✓ GitLab $name (masked+protected+hidden)" || echo "  ✗ GitLab $name FAILED"
  else
    echo "  - GitLab $name skipped (empty)"
  fi
}

set_github_secret() { # name label
  local name="$1" label="$2"
  if [ -n "${!name:-}" ]; then
    printf '%s' "${!name}" | gh secret set "$name" -R "$GITHUB_REPO" >/dev/null 2>&1 \
      && echo "  ✓ GitHub $name" || echo "  ✗ GitHub $name FAILED"
  else
    echo "  - GitHub $name skipped (empty)"
  fi
}

# ---------------------------------------------------------------------------
if [ "${1:-}" = "--check" ]; then
  echo "== GitLab (${GITLAB_REPO}) =="
  glab variable list -R "$GITLAB_REPO" 2>/dev/null || echo "  (unable to list)"
  echo "== GitHub (${GITHUB_REPO}) =="
  gh secret list -R "$GITHUB_REPO" 2>/dev/null || echo "  (unable to list)"
  exit 0
fi

cat <<'EOF'
Set the real secrets for GeoAirQuality CI/CD. Values are read silently and
piped via stdin (never echoed / never in argv). Leave blank to skip any.
EOF

echo
echo "1) Cloudflare API token — CI needs it to deploy the SPA to Pages."
echo "   Create at: https://dash.cloudflare.com/profile/api-tokens"
echo "   Permissions: Account → Cloudflare Pages → Edit; Account → Account Settings → Read"
ask CLOUDFLARE_API_TOKEN "CLOUDFLARE_API_TOKEN"
echo

echo "2) Kubernetes kubeconfigs (base64) — CI needs them for staging/prod deploys."
echo "   Export: cat ~/.kube/config | base64 | pbcopy"
ask KUBE_CONFIG_STAGING "KUBE_CONFIG_STAGING (base64 kubeconfig)"
ask KUBE_CONFIG_PRODUCTION "KUBE_CONFIG_PRODUCTION (base64 kubeconfig)"
echo

echo "3) News API keys (optional — the app runs without them, skipping news)."
ask AIRNOW_API_KEY "AIRNOW_API_KEY (optional)"
ask NEWSAPI_KEY "NEWSAPI_KEY (optional)"
echo

echo "Wiring secrets..."
echo "== GitLab =="
set_gitlab_secret CLOUDFLARE_API_TOKEN
set_gitlab_secret KUBE_CONFIG_STAGING
set_gitlab_secret KUBE_CONFIG_PRODUCTION
set_gitlab_secret AIRNOW_API_KEY
set_gitlab_secret NEWSAPI_KEY
echo "== GitHub =="
set_github_secret CLOUDFLARE_API_TOKEN
set_github_secret KUBE_CONFIG_STAGING
set_github_secret KUBE_CONFIG_PRODUCTION
set_github_secret AIRNOW_API_KEY
set_github_secret NEWSAPI_KEY

echo
echo "Done. Verify with: ./scripts/setup-ci-secrets.sh --check"
