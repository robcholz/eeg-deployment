#!/bin/sh
set -eu

# Usage:
#   sudo ./ncsu_guest_portal_time.sh
#
# What it does:
#   1) Hits neverssl.com to trigger Aruba captive portal redirect
#   2) Submits NC State "I Agree" form (guest access)
#   3) Sets system time from the captive portal HTTP Date header (works even if NTP is blocked)
#   4) Tries to write RTC (best-effort)
#   5) Verifies HTTPS connectivity

COOKIE_JAR="/tmp/ncsu_guest_cookiejar"
PORTAL_TRIGGER_URL="http://neverssl.com/"
PORTAL_REDIRECT_URL="http://neverssl.com/?cmd=redirect&arubalp=12345"
PORTAL_ACCEPT_PATH="/auth/index.html/u"
PORTAL_ACCEPT_URL="http://neverssl.com${PORTAL_ACCEPT_PATH}"

log() { echo "[portal-time] $*"; }

need_root() {
  if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
    echo "Run as root: sudo $0" >&2
    exit 1
  fi
}

have_cmd() { command -v "$1" >/dev/null 2>&1; }

sync_time_from_http_date() {
  # We intentionally use HTTP (not HTTPS) so it works on web-only guest networks.
  # The portal's nginx provides a Date header that is "good enough" to fix TLS validity issues.
  local date_hdr
  date_hdr="$(curl -sI --max-time 10 "$PORTAL_TRIGGER_URL" | awk -F': ' 'tolower($1)=="date"{print $2}' | tr -d '\r')"

  if [[ -z "${date_hdr}" ]]; then
    log "Failed to read Date header for time sync."
    return 1
  fi

  log "HTTP Date header: ${date_hdr}"
  # date -s accepts RFC 2822-ish date strings on GNU date
  date -s "${date_hdr}" >/dev/null
  log "System time set to: $(date)"
}

accept_portal() {
  rm -f "$COOKIE_JAR"

  log "Triggering captive portal (fetching $PORTAL_TRIGGER_URL)..."
  curl -sS -L -c "$COOKIE_JAR" -b "$COOKIE_JAR" --max-time 10 "$PORTAL_TRIGGER_URL" >/dev/null || true

  # Attempt accept with the known form fields.
  log "Submitting 'I Agree' to portal ($PORTAL_ACCEPT_URL)..."
  local resp
  resp="$(curl -sS -L -c "$COOKIE_JAR" -b "$COOKIE_JAR" \
    -d "email=Guest@ncsu.edu" \
    -d "cmd=cmd" \
    -d "Login=I%20Agree" \
    --max-time 10 \
    "$PORTAL_ACCEPT_URL" || true)"

  if echo "$resp" | grep -qi "logout"; then
    log "Portal acceptance likely succeeded (logout form detected)."
    return 0
  fi

  # Some setups still show the guest page; that's OK—time sync + retry may be needed.
  log "Portal acceptance response did not clearly indicate success; continuing."
  return 0
}

verify_https() {
  log "Verifying HTTPS (example.com)..."
  if curl -sSI --max-time 10 https://example.com >/dev/null; then
    log "HTTPS OK."
    return 0
  else
    log "HTTPS still failing. You may still be captive or time is still wrong."
    log "Try re-running the script, or switch from ncsu-guest to ncsu SSID for full access."
    return 1
  fi
}

try_write_rtc() {
  if have_cmd hwclock; then
    log "Attempting to write RTC (best effort)..."
    hwclock -w >/dev/null 2>&1 || log "RTC write failed (common on systems without writable RTC)."
  fi
}

main() {
  need_root
  have_cmd curl || { echo "curl is required. Install it first." >&2; exit 1; }

  accept_portal

  # Time sync is the critical piece to fix "certificate not yet valid".
  sync_time_from_http_date || true

  # If the portal was still active, accepting again after time fix sometimes helps.
  accept_portal

  try_write_rtc

  verify_https
  log "Done."
}

main "$@"