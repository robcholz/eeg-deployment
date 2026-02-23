#!/usr/bin/env bash
set -euo pipefail

COOKIE_JAR="/tmp/ncsu_guest_cookiejar"

# Captive portal trigger + connectivity check
TRIGGER_URL="http://captive.apple.com/hotspot-detect.html"
VERIFY_URL="http://connectivitycheck.gstatic.com/generate_204"  # true online => 204

log() { echo "[portal-time] $*"; }

need_root() {
  if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
    echo "Run as root: sudo $0" >&2
    exit 1
  fi
}

have_cmd() { command -v "$1" >/dev/null 2>&1; }

# Return effective URL after following HTTP redirects
effective_url() {
  curl -sS -L -o /dev/null -w '%{url_effective}\n' --max-time 15 "$1" || true
}

# Extract scheme://host from a URL
url_base() {
  local u="$1"
  if [[ "$u" =~ ^(https?://[^/]+) ]]; then
    echo "${BASH_REMATCH[1]}"
  else
    echo ""
  fi
}

# Fetch portal HTML (follows redirects), store cookies, and return body on stdout
fetch_portal_html() {
  curl -sS -L -c "$COOKIE_JAR" -b "$COOKIE_JAR" --max-time 15 "$TRIGGER_URL" || true
}

# Determine portal base by looking at the final effective URL
detect_portal_base() {
  local eff base
  eff="$(effective_url "$TRIGGER_URL")"
  base="$(url_base "$eff")"

  # If we ended up at captive.apple.com, we probably aren't being intercepted into the portal
  if [[ -z "$base" || "$base" == "http://captive.apple.com" || "$base" == "https://captive.apple.com" ]]; then
    echo ""
    return 0
  fi

  echo "$base"
}

# Parse the first <form ... action="..."> from HTML
parse_form_action() {
  # case-insensitive match, tolerant to attributes ordering
  sed -n "s/.*<form[^>]*action=['\"]\([^'\"]*\)['\"].*/\1/ip" | head -n 1
}

sync_time_from_http_date() {
  local date_hdr
  date_hdr="$(curl -sSI --max-time 10 "$TRIGGER_URL" | awk -F': ' 'tolower($1)=="date"{print $2}' | tr -d '\r' || true)"

  if [[ -z "${date_hdr}" ]]; then
    log "Failed to read Date header for time sync."
    return 1
  fi

  log "HTTP Date header: ${date_hdr}"
  date -s "${date_hdr}" >/dev/null
  log "System time set to: $(date)"
}

parse_meta_refresh_url() {
  # Extract url=... from <meta http-equiv='refresh' content='...; url=...'>
  sed -n "s/.*http-equiv=['\"]refresh['\"][^>]*content=['\"][^;]*;\s*url=\([^\"'> ]*\).*/\1/ip" \
    | head -n 1
}

accept_portal() {
  rm -f "$COOKIE_JAR"

  log "Triggering captive portal (fetching $TRIGGER_URL)..."

  local html portal_base action accept_url resp

  # 1) Fetch initial page (may be captive.apple.com meta refresh)
  html="$(curl -sS -L -c "$COOKIE_JAR" -b "$COOKIE_JAR" --max-time 15 "$TRIGGER_URL" || true)"

  # Try to parse form action directly (sometimes portal HTML is already here)
  action="$(printf '%s' "$html" | parse_form_action || true)"

  # 2) If no form action, look for meta refresh and follow it
  if [[ -z "$action" ]]; then
    local next
    next="$(printf '%s' "$html" | parse_meta_refresh_url || true)"

    if [[ -n "$next" ]]; then
      # Make next absolute if needed
      if [[ "$next" =~ ^// ]]; then
        next="http:${next}"
      elif [[ "$next" =~ ^/ ]]; then
        next="http://captive.apple.com${next}"
      fi

      log "Found meta refresh -> $next"

      # Fetch the redirected portal login page; capture effective URL to get base host
      local eff
      eff="$(curl -sS -L -o /dev/null -w '%{url_effective}\n' \
              -c "$COOKIE_JAR" -b "$COOKIE_JAR" --max-time 15 "$next" || true)"
      portal_base="$(url_base "$eff")"

      html="$(curl -sS -L -c "$COOKIE_JAR" -b "$COOKIE_JAR" --max-time 15 "$next" || true)"
      action="$(printf '%s' "$html" | parse_form_action || true)"
    fi
  fi

  # 3) If we still don't have portal_base, try to derive it from TRIGGER effective URL (may be useful when portal uses 302)
  if [[ -z "${portal_base:-}" ]]; then
    portal_base="$(detect_portal_base || true)"
  fi

  # If still no portal base or still captive.apple.com, we can't submit
  if [[ -z "${portal_base:-}" || "$portal_base" == "http://captive.apple.com" || "$portal_base" == "https://captive.apple.com" ]]; then
    log "No portal detected (or still captive.apple.com). Skipping portal accept."
    return 0
  fi

  if [[ -z "$action" ]]; then
    log "Portal detected ($portal_base) but cannot find form action. Skipping."
    return 1
  fi

  # 4) Build accept URL
  if [[ "$action" =~ ^https?:// ]]; then
    accept_url="$action"
  elif [[ "$action" =~ ^// ]]; then
    accept_url="https:${action}"
  elif [[ "$action" =~ ^/ ]]; then
    accept_url="${portal_base}${action}"
  else
    accept_url="${portal_base}/${action}"
  fi

  log "Detected portal base: $portal_base"
  log "Submitting 'I Agree' to portal ($accept_url)..."

  resp="$(curl -sS -L -c "$COOKIE_JAR" -b "$COOKIE_JAR" \
    -d "email=Guest@ncsu.edu" \
    -d "cmd=cmd" \
    -d "Login=I%20Agree" \
    --max-time 15 \
    "$accept_url" || true)"

  if echo "$resp" | grep -qiE "logout|success|accepted|you are connected|welcome"; then
    log "Portal acceptance likely succeeded."
  else
    log "Portal acceptance response unclear; continuing."
  fi
}

try_write_rtc() {
  if have_cmd hwclock; then
    log "Attempting to write RTC (best effort)..."
    hwclock -w >/dev/null 2>&1 || log "RTC write failed (common on systems without writable RTC)."
  fi
}

verify_online_http() {
  log "Verifying connectivity (generate_204 expects 204)..."
  local code
  code="$(curl -sS -o /dev/null -w '%{http_code}\n' --max-time 10 "$VERIFY_URL" || true)"

  if [[ "$code" == "204" ]]; then
    log "Connectivity OK (code=204)."
    return 0
  fi

  log "Connectivity NOT OK (code=$code). Dumping first lines of body:"
  curl -sS -L --max-time 10 "$VERIFY_URL" | head -n 20 || true
  return 1
}

main() {
  need_root
  have_cmd curl || { echo "curl is required." >&2; exit 1; }

  accept_portal
  sync_time_from_http_date || true
  # Run accept again after time sync in case portal behavior depends on correct clock
  accept_portal
  try_write_rtc
  verify_online_http
  log "Done."
}

main "$@"
