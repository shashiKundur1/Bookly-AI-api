#!/bin/sh
set -e

if [ -d /app/models-seed ]; then
  mkdir -p /app/data/models
  cp -n /app/models-seed/* /app/data/models/ || true
fi

alembic upgrade head

exec "$@"
