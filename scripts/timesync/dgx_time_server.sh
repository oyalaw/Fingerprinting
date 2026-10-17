#!/usr/bin/env bash
# Make this machine (the DGX, FL server at 10.42.0.195) the time server for the testbed.
#
#   sudo bash scripts/timesync/dgx_time_server.sh [SUBNET]      # default 10.42.0.0/24
#
# Installs chrony (it replaces systemd-timesyncd), keeps the internet pools as upstream when
# reachable, serves time to the testbed subnet, and keeps serving from its own clock when the
# hotspot has no internet ("local stratum 10"), so all roles stay aligned with each other.
set -euo pipefail
SUBNET="${1:-10.42.0.0/24}"
[ "$(id -u)" -eq 0 ] || { echo "run with sudo"; exit 1; }

apt-get install -y chrony
mkdir -p /etc/chrony/conf.d
cat > /etc/chrony/conf.d/aifp-testbed.conf <<CONF
# AI fingerprinting testbed: serve time to the FL clients and the proxy
allow ${SUBNET}
local stratum 10
CONF
systemctl enable --now chrony
systemctl restart chrony

if command -v ufw >/dev/null && ufw status | grep -q "Status: active"; then
    ufw allow from "${SUBNET}" to any port 123 proto udp
fi

sleep 3
echo "--- chronyc tracking"
chronyc tracking
echo "--- clients will appear below once they sync:  sudo chronyc clients"
