#!/usr/bin/env bash
set -euo pipefail

COOKIE_JAR="/tmp/ncsu_guest_cookiejar"

TRIGGER_URL="http://example.com/"

log() { echo "[portal-time] $*"; }

need_root() {
  if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
    echo "Run as root: sudo $0" >&2
    exit 1
  fi
}

have_cmd() { command -v "$1" >/dev/null 2>&1; }

detect_portal_base() {
  local loc
  loc="$(curl -sSIL --max-time 10 "$TRIGGER_URL" | awk -F': ' 'tolower($1)=="location"{print $2}' | tail -n 1 | tr -d '\r' || true)"

  if [[ -z "$loc" ]]; then
    echo ""
    return 0
  fi

  if [[ "$loc" =~ ^(https?://[^/]+) ]]; then
    echo "${BASH_REMATCH[1]}"
  else
    echo ""
  fi
}

sync_time_from_http_date() {
  local date_hdr
  date_hdr="$(curl -sSI --max-time 10 "$TRIGGER_URL" | awk -F': ' 'tolower($1)=="date"{print $2}' | tr -d '\r')"

  if [[ -z "${date_hdr}" ]]; then
    log "Failed to read Date header for time sync."
    return 1
  fi

  log "HTTP Date header: ${date_hdr}"
  date -s "${date_hdr}" >/dev/null
  log "System time set to: $(date)"
}

accept_portal() {
  rm -f "$COOKIE_JAR"

  log "Triggering captive portal (fetching $TRIGGER_URL)..."
  curl -sS -L -c "$COOKIE_JAR" -b "$COOKIE_JAR" --max-time 10 "$TRIGGER_URL" >/dev/null || true

  local portal_base accept_url resp
  portal_base="$(detect_portal_base)"

  if [[ -z "$portal_base" ]]; then
    log "No portal redirect detected (or cannot parse). Skipping portal accept."
    return 0
  fi

  accept_url="${portal_base}/auth/index.html/u"
  log "Detected portal base: $portal_base"
  log "Submitting 'I Agree' to portal ($accept_url)..."

  resp="$(curl -sS -L -c "$COOKIE_JAR" -b "$COOKIE_JAR" \
    -d "email=Guest@ncsu.edu" \
    -d "cmd=cmd" \
    -d "Login=I%20Agree" \
    --max-time 10 \
    "$accept_url" || true)"

  if echo "$resp" | grep -qiE "logout|success|accepted"; then
    log "Portal acceptance likely succeeded."
  else
    log "Portal acceptance response did not clearly indicate success; continuing."
  fi
}

try_write_rtc() {
  if have_cmd hwclock; then
    log "Attempting to write RTC (best effort)..."
    hwclock -w >/dev/null 2>&1 || log "RTC write failed (common on systems without writable RTC)."
  fi
}

verify_https() {
  log "Verifying HTTPS (example.com)..."
  if curl -sSI --max-time 10 https://example.com >/dev/null; then
    log "HTTPS OK."
    return 0
  fi
  log "HTTPS still failing. You may still be captive or time is still wrong."
  return 1
}

main() {
  need_root
  have_cmd curl || { echo "curl is required." >&2; exit 1; }

  accept_portal
  sync_time_from_http_date || true
  accept_portal
  try_write_rtc
  verify_https
  log "Done."
}

main "$@"