#!/usr/bin/env bash
# Automa from outside the house (D188): runs `automa web --public` together with a
# Cloudflare Tunnel, and stops the tunnel again when Automa stops.
#
# Run it from the Automa clone (the Windows launcher automa-web.ps1 does this with
# -PublicHost); setup steps: docs/REMOTE-ACCESS.md.
#
#   bash scripts/automa-tunnel.sh <tunnel-name> <host-name> --port <port> [--allow-actions]
#
# Exit codes: 2 bad arguments, 3 no Automa install here, 4 cloudflared not installed;
# otherwise Automa's own exit code. Automa still listens only on 127.0.0.1; the tunnel
# reaches it there.
set -u

usage() {
    echo "usage: bash scripts/automa-tunnel.sh <tunnel-name> <host-name> --port <port> [--allow-actions]" >&2
    exit 2
}

[ "$#" -ge 4 ] || usage
name=$1
host=$2
shift 2
[ "$1" = "--port" ] || usage
port=$2
shift 2
actions=()
if [ "$#" -gt 0 ]; then
    [ "$#" -eq 1 ] && [ "$1" = "--allow-actions" ] || usage
    actions=(--allow-actions)
fi

label='[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?'
[[ $name =~ ^[A-Za-z0-9._-]+$ ]] || { echo "invalid tunnel name: $name" >&2; exit 2; }
[[ $host =~ ^$label(\.$label)+$ ]] || {
    echo "invalid host name: $host (use a plain name such as automa.example.com)" >&2
    exit 2
}
[[ $port =~ ^[0-9]{1,5}$ ]] && [ "$port" -ge 1024 ] && [ "$port" -le 65535 ] || {
    echo "invalid port: $port" >&2
    exit 2
}

test -x .venv/bin/automa || exit 3
if ! command -v cloudflared >/dev/null 2>&1; then
    echo "cloudflared is not installed in WSL. See docs/REMOTE-ACCESS.md (step 2)." >&2
    exit 4
fi

log_dir="$HOME/.cloudflared"
mkdir -p "$log_dir"
log="$log_dir/automa-tunnel-$name.log"

tunnel_pid=''
watch_pid=''
cleanup() {
    [ -n "$watch_pid" ] && kill "$watch_pid" 2>/dev/null
    [ -n "$tunnel_pid" ] && kill "$tunnel_pid" 2>/dev/null
    return 0
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP

cloudflared tunnel run --url "http://127.0.0.1:$port" "$name" >"$log" 2>&1 &
tunnel_pid=$!
echo "Tunnel '$name' started (log: $log). Public address: https://$host/"

# A tunnel that dies at once (wrong name, not logged in) would leave the page
# unreachable from outside without any message, so say it here.
(
    sleep 8 >/dev/null 2>&1 </dev/null
    if ! kill -0 "$tunnel_pid" 2>/dev/null; then
        echo "WARNING: the tunnel stopped; Automa works only on this computer now." >&2
        echo "See $log and docs/REMOTE-ACCESS.md (When something goes wrong)." >&2
    fi
) &
watch_pid=$!

.venv/bin/automa web --port "$port" "${actions[@]}" --public --host-name "$host"
exit $?
