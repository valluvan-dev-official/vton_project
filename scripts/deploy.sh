#!/usr/bin/env bash
#
# deploy.sh — Deploys DCI-VTON.
# Lives in and runs directly from the Git repository on the EC2 host.
# Invoked over SSH by .github/workflows/deploy-ec2.yml, which has already
# `cd`-ed into the project root (PROJECT_DIR) before calling this script.
#
# Usage: scripts/deploy.sh [health_url]
#
# Layout on the server:
#   PROJECT_DIR/            <- git root, current working directory on entry
#   PROJECT_DIR/api/        <- docker-compose.yml, Dockerfile, .env live here
#   PROJECT_DIR/scripts/    <- this script
#
# Contract:
#   - PROJECT_DIR is taken from the current working directory (the caller
#     already `cd`-ed there) — never hardcoded.
#   - Git operations (fetch/pull) run in PROJECT_DIR.
#   - All `docker compose` commands run from PROJECT_DIR/api, where the
#     compose file actually lives.
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
    echo "--- df -h ---";            df -h
    echo "--- docker system df ---"; docker system df
    echo "--- docker ps ---";        docker ps
    echo "--- docker images ---";    docker images
}

trap 'fail "unexpected error at line $LINENO"' ERR

# PROJECT_DIR is wherever the caller cd'ed to before invoking this script.
PROJECT_DIR="$(pwd)"
COMPOSE_DIR="$PROJECT_DIR/api"

git -C "$PROJECT_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1 \
    || fail "$PROJECT_DIR is not a git repository"

if [[ ! -f "$COMPOSE_DIR/docker-compose.yml" && ! -f "$COMPOSE_DIR/compose.yaml" ]]; then
    fail "docker-compose.yml not found: $COMPOSE_DIR/docker-compose.yml"
fi

BRANCH="$(git -C "$PROJECT_DIR" rev-parse --abbrev-ref HEAD)"

snapshot "before deployment"

log "Step 1/6: Pulling latest code on branch '$BRANCH' in $PROJECT_DIR"
git -C "$PROJECT_DIR" fetch origin
git -C "$PROJECT_DIR" pull origin "$BRANCH"

cd "$COMPOSE_DIR"
log "Using compose directory: $COMPOSE_DIR"

if [[ ! -e "$PROJECT_DIR/.env" ]]; then
    if [[ -f "$PROJECT_DIR/api/.env" ]]; then
        if ln -s "$PROJECT_DIR/api/.env" "$PROJECT_DIR/.env" 2>/dev/null; then
            log "No .env at $PROJECT_DIR/.env — symlinked it to $PROJECT_DIR/api/.env"
        else
            cp "$PROJECT_DIR/api/.env" "$PROJECT_DIR/.env"
            log "No .env at $PROJECT_DIR/.env — symlink failed, copied it from $PROJECT_DIR/api/.env instead"
        fi
    else
        log "No .env found at $PROJECT_DIR/.env or $PROJECT_DIR/api/.env — continuing without one"
    fi
else
    log ".env already present at $PROJECT_DIR/.env — leaving as is"
fi

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

log "Deployment summary: branch=$BRANCH compose_dir=$COMPOSE_DIR health_url=$HEALTH_URL status=SUCCESS"
