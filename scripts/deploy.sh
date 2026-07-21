#!/usr/bin/env bash
#
# deploy.sh — Deploys DCI-VTON.
# Lives in and runs directly from the Git repository on the EC2 host.
# Invoked over SSH by .github/workflows/deploy-ec2.yml, which has already
# `cd`-ed into the project directory before calling this script.
#
# Usage: scripts/deploy.sh [health_url]
#
# Contract:
#   - Assumes the current working directory IS the project directory
#     (a git checkout containing docker-compose.yml). The caller is
#     responsible for `cd`-ing there.
#   - Pulls the latest code on whatever branch is currently checked out
#     (the workflow only ever triggers on pushes to `ec2`, so that is
#     always the branch in play).
#   - Never runs `docker compose down` — only changed services are
#     recreated via `docker compose up -d`, so postgres/redis are left
#     running untouched unless their own config/image changed.
#   - Never removes volumes or networks.
#   - Exits non-zero (and leaves the current stack running) on any
#     failure up through the build step. If the post-deploy health check
#     fails, the new containers stay up for inspection but the script
#     still reports failure so it is never silently treated as success.

set -Eeuo pipefail

HEALTH_URL="${1:-http://localhost:8000/health}"
FALLBACK_HEALTH_URL="http://localhost:8000/docs"
COMPOSE="docker compose"
HEALTH_RETRIES=10
HEALTH_RETRY_DELAY=6

log()  { printf '\n[deploy] %s — %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$1"; }
fail() { log "FAILED: $1"; exit 1; }

snapshot() {
    log "Resource snapshot: $1"
    echo "--- df -h ---";          df -h
    echo "--- docker system df ---"; docker system df
    echo "--- docker ps ---";       docker ps
    echo "--- docker images ---";   docker images
}

trap 'fail "unexpected error at line $LINENO"' ERR

[[ -f docker-compose.yml || -f compose.yaml ]] || fail "no docker-compose.yml found in $(pwd)"
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || fail "$(pwd) is not a git repository"

BRANCH="$(git rev-parse --abbrev-ref HEAD)"

snapshot "before deployment"

log "Step 1/6: Pulling latest code on branch '$BRANCH'"
git fetch origin
git pull origin "$BRANCH"

log "Step 2/6: Building project images (api, worker, flower only — postgres/redis/base images untouched)"
if ! $COMPOSE build api worker flower; then
    fail "docker compose build failed — running containers were NOT stopped or restarted"
fi
log "Build succeeded"

log "Step 3/6: Recreating only changed services (no 'compose down' — postgres/redis stay up unless their config changed)"
$COMPOSE up -d

log "Step 4/6: Verifying container status"
sleep 5
$COMPOSE ps

if $COMPOSE ps --format json 2>/dev/null | grep -q '"State":"exited"'; then
    fail "one or more containers exited after deployment — check 'docker compose logs'"
fi

log "Step 5/6: FastAPI health check ($HEALTH_URL, up to $HEALTH_RETRIES attempts)"
healthy=0
for attempt in $(seq 1 "$HEALTH_RETRIES"); do
    if curl -fsS -o /dev/null "$HEALTH_URL" || curl -fsS -o /dev/null "$FALLBACK_HEALTH_URL"; then
        log "Health check passed on attempt $attempt"
        healthy=1
        break
    fi
    log "Health check attempt $attempt/$HEALTH_RETRIES failed, retrying in ${HEALTH_RETRY_DELAY}s"
    sleep "$HEALTH_RETRY_DELAY"
done
[[ "$healthy" -eq 1 ]] || fail "FastAPI health check did not pass after $HEALTH_RETRIES attempts (new containers left running for inspection)"

log "Step 6/6: Cleaning up unused images (volumes and networks are preserved)"
docker image prune -f

snapshot "after deployment"

log "Deployment summary: branch=$BRANCH health_url=$HEALTH_URL status=SUCCESS"
