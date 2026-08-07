#!/usr/bin/env bash

wandb_has_netrc_login() {
  local wandb_host="$1"
  python - "$wandb_host" <<'PY'
import netrc
import sys
from urllib.parse import urlparse

host = sys.argv[1].rstrip('/')
parsed = urlparse(host if '://' in host else f'https://{host}')
machine = parsed.netloc or parsed.path
try:
    auth = netrc.netrc().authenticators(machine)
except Exception:
    auth = None
print("yes" if auth and auth[2] else "no")
PY
}

wandb_check_host() {
  local wandb_host="$1"
  python - "$wandb_host" <<'PY'
import sys
import urllib.error
import urllib.request

url = sys.argv[1].rstrip('/') + '/'
request = urllib.request.Request(url, headers={"User-Agent": "wandb-preflight"})
try:
    with urllib.request.urlopen(request, timeout=5) as response:
        print(f"W&B host reachable: HTTP {response.status}")
except urllib.error.HTTPError as exc:
    print(f"W&B host reachable with HTTP error: {exc.code}")
except Exception as exc:
    raise SystemExit(f"W&B host check failed for {url}: {exc}")
PY
}

assert_wandb_ready() {
  local wandb_host="$1"

  case "${WANDB_MODE:-online}" in
    offline|disabled)
      echo "WANDB_MODE=${WANDB_MODE:-online}; skipping W&B reachability/login preflight."
      export WANDB_BASE_URL="$wandb_host"
      return
      ;;
  esac

  if ! command -v wandb >/dev/null 2>&1; then
    echo "wandb CLI not found; refusing to launch without W&B." >&2
    exit 1
  fi

  echo "Checking W&B login against ${wandb_host}"
  wandb --version
  wandb status || true
  wandb_check_host "$wandb_host"

  if [ -n "${WANDB_API_KEY:-}" ]; then
    echo "WANDB_API_KEY detected; enabling W&B logging."
  elif [ "$(wandb_has_netrc_login "$wandb_host")" = "yes" ]; then
    echo "Detected W&B credentials for ${wandb_host} in ~/.netrc; enabling W&B logging."
  else
    echo "No W&B credentials detected for ${wandb_host}. Run 'wandb login --relogin --host=${wandb_host}' first." >&2
    exit 1
  fi

  export WANDB_BASE_URL="$wandb_host"
}
