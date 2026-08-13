#!/usr/bin/env bash
#
# deploy.sh — Deploys DCI-VTON.
# Lives in and runs directly from the Git repository on the EC2 host.
# Invoked over SSH by .github/workflows/deploy-ec2.yml, which has already
# `cd`-ed into the project root (PROJECT_DIR) before calling this script.
#
# Usage: scripts/deploy.sh [health_url]
#        FORCE_WORKER_RESTART=1 scripts/deploy.sh   (force worker recreate
#        regardless of what changed — e.g. the worker crashed and needs a
#        restart even though no GPU-relevant file changed since last deploy)
#
# Layout on the server:
#   PROJECT_DIR/            <- git root, current working directory on entry
#   PROJECT_DIR/api/        <- docker-compose.gpu.yml, Dockerfile.gpu, .env live here
#   PROJECT_DIR/scripts/    <- this script
#
# Contract:
#   - PROJECT_DIR is taken from the current working directory (the caller
#     already `cd`-ed there) — never hardcoded.
#   - Git operations (fetch/pull) run in PROJECT_DIR.
#   - All `docker compose` commands run from PROJECT_DIR/api using
#     -f docker-compose.gpu.yml explicitly. api/flower build from the
#     lightweight api/Dockerfile (CPU-only, no torch); only worker builds
#     from api/Dockerfile.gpu (PyTorch + CUDA base image,
#     requirements.txt + requirements-gpu.txt). The CPU-only
#     docker-compose.yml is never referenced.
#   - Pulls the latest code on whatever branch is currently checked out
#     (the workflow only ever triggers on pushes to `ec2`, so that is
#     always the branch in play).
#   - Never runs `docker compose down` — only changed services are
#     recreated via `docker compose up -d`, so postgres/redis are left
#     running untouched unless their own config/image changed.
#   - The GPU worker is only rebuilt+recreated when a GPU-relevant path
#     changed since the last successful deploy (see "GPU-relevant change
#     detection" below) — api/flower always deploy on every push. This is
#     what lets a pure API/frontend/catalog change go out without
#     restarting a worker that may have a model loaded in GPU RAM.
#   - Never removes volumes or networks.
#   - Exits non-zero (and leaves the current stack running) on any
#     failure up through the build step. If the post-deploy health check
#     fails, the new containers stay up for inspection but the script
#     still reports failure so it is never silently treated as success.

set -Eeuo pipefail

HEALTH_URL="${1:-http://localhost:8000/health}"
FALLBACK_HEALTH_URL="http://localhost:8000/docs"
# Production always builds/runs the GPU stack (api/docker-compose.gpu.yml).
# The plain "docker compose" invocation would silently fall back to
# api/docker-compose.yml (CPU image, no torch) since that's Compose's
# default file when no -f is given — so the GPU file must always be named
# explicitly.
COMPOSE_FILE="docker-compose.gpu.yml"
COMPOSE="docker compose -f $COMPOSE_FILE"
HEALTH_RETRIES=10
HEALTH_RETRY_DELAY=6
FORCE_WORKER_RESTART="${FORCE_WORKER_RESTART:-0}"

# Paths that mean "the GPU worker needs to be rebuilt/recreated" — the ML
# pipeline itself, the GPU-specific inference plumbing, GPU dependencies/
# image definition, or the compose file that configures the worker's
# volumes/env/GPU device reservation. Deliberately NOT included: app/
# models/*, app/config.py, and most of app/services/* — those ARE imported
# by the worker too, but changes there (e.g. the shadow-mode fit_analysis/
# garment_catalog code) are safe to reach the worker on its next
# GPU-relevant deploy rather than forcing an immediate restart, since
# nothing on that path can affect an in-flight or future render.
#
# api/requirements.txt and app/services/accessory_engine.py were added
# after two manual deploys in a row silently shipped a stale worker
# container: requirements.txt carries the mediapipe/protobuf pins the
# worker's accessory (wrist/glasses) inference depends on, and
# accessory_engine.py is the render logic itself for those two engines —
# both directly affect live render output, unlike the shadow-mode-only
# services this exclusion list was originally written for. Use
# FORCE_WORKER_RESTART=1 to override this on any given deploy regardless
# of what the diff contains.
GPU_RELEVANT_PATTERN='^ml/|^api/Dockerfile\.gpu$|^api/requirements\.txt$|^api/requirements-gpu\.txt$|^api/app/services/gpu_inference_service\.py$|^api/app/services/inference\.py$|^api/app/services/model_bootstrap\.py$|^api/app/services/accessory_engine\.py$|^api/app/workers/|^api/docker-compose\.gpu\.yml$'

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
# Lives inside .git/ so it's part of the repo's local metadata, never
# tracked, and never touched/conflicted by `git pull`.
DEPLOY_STATE_FILE="$PROJECT_DIR/.git/vton-deploy-last-sha"

git -C "$PROJECT_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1 \
    || fail "$PROJECT_DIR is not a git repository"

if [[ ! -f "$COMPOSE_DIR/$COMPOSE_FILE" ]]; then
    fail "GPU compose file not found: $COMPOSE_DIR/$COMPOSE_FILE"
fi

BRANCH="$(git -C "$PROJECT_DIR" rev-parse --abbrev-ref HEAD)"
PREV_SHA=""
[[ -f "$DEPLOY_STATE_FILE" ]] && PREV_SHA="$(cat "$DEPLOY_STATE_FILE" 2>/dev/null || true)"

snapshot "before deployment"

log "Step 1/7: Pulling latest code on branch '$BRANCH' in $PROJECT_DIR"
git -C "$PROJECT_DIR" fetch origin
git -C "$PROJECT_DIR" pull origin "$BRANCH"

NEW_SHA="$(git -C "$PROJECT_DIR" rev-parse HEAD)"

cd "$COMPOSE_DIR"
log "Using compose directory: $COMPOSE_DIR (file: $COMPOSE_FILE)"

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

# ── GPU-relevant change detection ───────────────────────────────────────────
log "Step 2/7: Determining whether the GPU worker needs to be recreated"
GPU_RELEVANT=0
if [[ "$FORCE_WORKER_RESTART" == "1" ]]; then
    log "FORCE_WORKER_RESTART=1 — worker will be rebuilt/recreated regardless of changed files."
    GPU_RELEVANT=1
elif [[ -z "$PREV_SHA" ]]; then
    log "No previous-deploy marker found ($DEPLOY_STATE_FILE) — treating this as a full deploy."
    GPU_RELEVANT=1
elif ! git -C "$PROJECT_DIR" cat-file -e "${PREV_SHA}^{commit}" 2>/dev/null; then
    log "Previous-deploy marker '$PREV_SHA' is not a valid commit in this repo anymore — treating this as a full deploy."
    GPU_RELEVANT=1
elif [[ "$PREV_SHA" == "$NEW_SHA" ]]; then
    log "HEAD unchanged since last deploy ($NEW_SHA) — no files changed, worker will not be recreated."
else
    CHANGED_FILES="$(git -C "$PROJECT_DIR" diff --name-only "$PREV_SHA" "$NEW_SHA")"
    log "Files changed between $PREV_SHA and $NEW_SHA:"
    echo "$CHANGED_FILES"
    if echo "$CHANGED_FILES" | grep -Eq "$GPU_RELEVANT_PATTERN"; then
        log "GPU-relevant path(s) changed — worker will be rebuilt/recreated."
        GPU_RELEVANT=1
    else
        log "No GPU-relevant paths changed — worker will be left running untouched."
    fi
fi

log "Step 3/7: Building project images (api, flower always; worker only if needed)"
if ! $COMPOSE build api flower; then
    fail "docker compose build (api, flower) failed — running containers were NOT stopped or restarted"
fi
# Always attempt the worker build too, even when GPU_RELEVANT=0: with
# Dockerfile.gpu's COPY scoped to only worker-relevant paths, this is a
# fast no-op via Docker's own layer cache whenever nothing worker-relevant
# changed, and keeps a ready image available immediately in case a manual
# FORCE_WORKER_RESTART is used later without a rebuild step.
if ! $COMPOSE build worker; then
    fail "docker compose build (worker) failed — running containers were NOT stopped or restarted"
fi
log "Build succeeded"

log "Step 4/7: Recreating changed services (no 'compose down' — postgres/redis stay up unless their config changed)"
$COMPOSE up -d api flower
if [[ "$GPU_RELEVANT" -eq 1 ]]; then
    log "Recreating worker (--force-recreate — guarantees a fresh process even when only the ml/ bind mount changed, since that never changes the image itself)"
    $COMPOSE up -d --force-recreate worker
else
    log "Skipping worker recreate — no GPU-relevant changes detected since the last deploy. The running worker (with its already-loaded GPU engine) is left untouched."
fi

log "Step 5/7: Verifying container status"
sleep 5
$COMPOSE ps

if $COMPOSE ps --format json 2>/dev/null | grep -q '"State":"exited"'; then
    fail "one or more containers exited after deployment — check 'docker compose logs'"
fi

log "Step 6/7: FastAPI health check ($HEALTH_URL, up to $HEALTH_RETRIES attempts)"
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

# Only record this deploy as "done" once the health check has actually
# passed — a failed deploy must not advance the marker, so the next run
# still sees (and acts on) whatever GPU-relevant changes were in this push.
echo "$NEW_SHA" > "$DEPLOY_STATE_FILE"

log "Step 7/7: Cleaning up unused images (volumes and networks are preserved)"
docker image prune -f

snapshot "after deployment"

log "Deployment summary: branch=$BRANCH compose_dir=$COMPOSE_DIR compose_file=$COMPOSE_FILE health_url=$HEALTH_URL worker_recreated=$([[ $GPU_RELEVANT -eq 1 ]] && echo yes || echo no) status=SUCCESS"
