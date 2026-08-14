#!/usr/bin/env bash
#
# Smoke test for deploy/nginx/ephemeris-serve.conf.
#
#   make smoke-nginx
#
# Starts a stub upstream (no model) and a private nginx instance running the
# real site config, asserts the proxy's behavior, then tears both down. Exits
# non-zero if any check fails, so it is usable in CI.
#
# Requires nginx on PATH. Nothing it starts is left running, and it never
# touches a system nginx install: everything lives in a temp prefix, on ports
# 18000/18080 so a real server on 8000/8080 is undisturbed.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SITE_CONF="${REPO_ROOT}/deploy/nginx/ephemeris-serve.conf"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"

# Non-default ports so the test can run alongside a real server on 8000/8080.
# They are substituted into the temp copy of the site config, so what is
# exercised is still the shipped config, just bound elsewhere.
UPSTREAM_PORT="${UPSTREAM_PORT:-18000}"
PROXY_PORT="${PROXY_PORT:-18080}"

WORK="$(mktemp -d)"
UPSTREAM_PID=""

cleanup() {
  nginx -s stop -c "${WORK}/nginx.conf" -p "${WORK}" 2>/dev/null
  if [ -n "${UPSTREAM_PID}" ]; then
    kill "${UPSTREAM_PID}" 2>/dev/null
    # uvicorn can outlive a plain TERM to the launcher; wait, then insist.
    for _ in 1 2 3 4 5; do
      kill -0 "${UPSTREAM_PID}" 2>/dev/null || break
      sleep 0.4
    done
    kill -9 "${UPSTREAM_PID}" 2>/dev/null
  fi
  sleep 1
  # Last resort: never leave the port held, whatever happened above.
  if lsof -nP -iTCP:"${UPSTREAM_PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
    pkill -f "uvicorn stub_server:app" 2>/dev/null
    sleep 0.5
  fi
  rm -rf "${WORK}"
}
trap cleanup EXIT

command -v nginx >/dev/null || { echo "error: nginx not on PATH (brew install nginx)"; exit 2; }
[ -x "${PYTHON}" ] || { echo "error: no interpreter at ${PYTHON}; set PYTHON="; exit 2; }

for port in "${UPSTREAM_PORT}" "${PROXY_PORT}"; do
  if lsof -nP -iTCP:"${port}" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "error: port ${port} is already in use."
    echo "       override with: UPSTREAM_PORT=... PROXY_PORT=... make smoke-nginx"
    exit 2
  fi
done

mkdir -p "${WORK}/logs" "${WORK}/temp"

# The site config logs to /var/log/nginx, which needs root. Point it at the
# temp prefix instead; nothing else about the config is altered, so what runs
# here is what ships.
sed -e "s#/var/log/nginx#${WORK}/logs#g" \
    -e "s#server 127.0.0.1:8000;#server 127.0.0.1:${UPSTREAM_PORT};#" \
    -e "s#listen 8080;#listen ${PROXY_PORT};#" \
    -e "s#listen \[::\]:8080;#listen [::]:${PROXY_PORT};#" \
    "${SITE_CONF}" > "${WORK}/site.conf"

# The rewrite must actually have taken -- a renamed directive in the site
# config would otherwise leave this pointing at the real server's ports.
grep -q "server 127.0.0.1:${UPSTREAM_PORT};" "${WORK}/site.conf" || {
  echo "error: could not redirect the upstream port; has the upstream block in"
  echo "       ${SITE_CONF} changed shape?"; exit 2; }
grep -q "listen ${PROXY_PORT};" "${WORK}/site.conf" || {
  echo "error: could not redirect the listen port; has the server block in"
  echo "       ${SITE_CONF} changed shape?"; exit 2; }

MIME_TYPES="$(nginx -V 2>&1 | sed -n 's/.*--conf-path=\([^ ]*\)nginx.conf.*/\1mime.types/p')"
[ -f "${MIME_TYPES}" ] || MIME_TYPES=/dev/null

cat > "${WORK}/nginx.conf" <<EOF
worker_processes 1;
error_log ${WORK}/logs/error.log warn;
pid ${WORK}/nginx.pid;
events { worker_connections 64; }
http {
    include ${MIME_TYPES};
    default_type application/octet-stream;
    client_body_temp_path ${WORK}/temp/body;
    proxy_temp_path ${WORK}/temp/proxy;
    fastcgi_temp_path ${WORK}/temp/fastcgi;
    uwsgi_temp_path ${WORK}/temp/uwsgi;
    scgi_temp_path ${WORK}/temp/scgi;
    include ${WORK}/site.conf;
}
EOF

echo "==> syntax check"
nginx -t -c "${WORK}/nginx.conf" -p "${WORK}" 2>&1 | sed 's/^/  /' || exit 1

echo
echo "==> starting stub upstream on 127.0.0.1:${UPSTREAM_PORT}"
( cd "${REPO_ROOT}/deploy/smoke" && exec "${PYTHON}" -m uvicorn stub_server:app \
    --host 127.0.0.1 --port "${UPSTREAM_PORT}" --proxy-headers --forwarded-allow-ips 127.0.0.1 \
    >"${WORK}/upstream.log" 2>&1 ) &
UPSTREAM_PID=$!

for _ in $(seq 1 30); do
  curl -sf http://127.0.0.1:${UPSTREAM_PORT}/health >/dev/null 2>&1 && break
  sleep 0.5
done
curl -sf http://127.0.0.1:${UPSTREAM_PORT}/health >/dev/null || {
  echo "error: stub upstream never came up"; sed 's/^/  /' "${WORK}/upstream.log"; exit 1; }

echo "==> starting nginx on 127.0.0.1:${PROXY_PORT}"
nginx -c "${WORK}/nginx.conf" -p "${WORK}" || exit 1
sleep 1

echo
UPSTREAM_PORT="${UPSTREAM_PORT}" PROXY_PORT="${PROXY_PORT}" \
  "${PYTHON}" "${REPO_ROOT}/deploy/smoke/check_proxy.py"
