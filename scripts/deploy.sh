#!/usr/bin/env bash
#
# deploy.sh — Deploys DCI-VTON on the EC2 host.
# Invoked over SSH by .github/workflows/deploy-ec2.yml.
#
# Usage: deploy.sh <project_dir> <branch> [health_url]
#
# Contract:
#   - Never touches the running stack until a build has succeeded.
#   - Never runs `docker compose down` — only changed services are
#     recreated via `docker compose up -d`, so postgres/redis are left
#     running untouched unless their own config/image changed.
#   - Never removes volumes or networks.
#   - Exits non-zero (and leaves the current stack running) on any failure
#     up through the build step. If the post-deploy health check fails,
#     the new containers stay up for inspection but the script/job
#     reports failure so it is never silently treated as success.

set -Eeuo pipefail

PROJECT_DIR="${1:?Usage: deploy.sh <project_dir> <branch> [health_url]}"
BRANCH="${2:?Usage: deploy.sh <project_dir> <branch> [health_url]}"
HEALTH_URL="${3:-http://localhost:8000/health}"
FALLBACK_HEALTH_URL="http://localhost:8000/docs"
COMPOSE="docker compose"

HEALTH_RETRIES=10
HEALTH_RETRY_DELAY=6

log() {
    printf '\n[deploy] %s — %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$1"
}

fail() {
    log "FAILED: $1"
    exit 1
}

snapshot() {
    local label="$1"
    log "Docker/host resource snapshot: $label"
    echo "--- df -h ---"
    df -h
    echo "--- docker system df ---"
    docker system df
    echo "--- docker ps ---"
    docker ps
    echo "--- docker images ---"
    docker images
}

trap 'fail "unexpected error at line $LINENO"' ERR

[[ -d "$PROJECT_DIR" ]] || fail "project directory '$PROJECT_DIR' does not exist on this host"
cd "$PROJECT_DIR"
[[ -f docker-compose.yml || -f compose.yaml ]] || fail "no docker-compose.yml found in $PROJECT_DIR"

snapshot "before deployment"

log "Step 1/6: Fetching latest code for branch '$BRANCH'"
git fetch origin
git checkout "$BRANCH"
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

UNHEALTHY=$($COMPOSE ps --format json 2>/dev/null | grep -c '"State":"exited"' || true)
if [[ "${UNHEALTHY:-0}" -gt 0 ]]; then
    fail "one or more containers exited after deployment — check 'docker compose logs'"
fi

log "Step 5/6: FastAPI health check ($HEALTH_URL, up to $HEALTH_RETRIES attempts)"
healthy=0
for attempt in $(seq 1 "$HEALTH_RETRIES"); do
    if curl -fsS -o /dev/null "$HEALTH_URL"; then
        log "Health check passed on attempt $attempt ($HEALTH_URL)"
        healthy=1
        break
    fi
    if curl -fsS -o /dev/null "$FALLBACK_HEALTH_URL"; then
        log "Health check passed on attempt $attempt (fallback $FALLBACK_HEALTH_URL)"
        healthy=1
        break
    fi
    log "Health check attempt $attempt/$HEALTH_RETRIES failed, retrying in ${HEALTH_RETRY_DELAY}s"
    sleep "$HEALTH_RETRY_DELAY"
done

if [[ "$healthy" -ne 1 ]]; then
    log "New containers are still running for inspection (no rollback performed)."
    fail "FastAPI health check did not pass after $HEALTH_RETRIES attempts"
fi

log "Step 6/6: Cleaning up unused images (volumes and networks are preserved)"
docker image prune -f

snapshot "after deployment"

log "Deployment completed successfully"
