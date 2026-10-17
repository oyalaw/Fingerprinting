#!/usr/bin/env bash
# Point a Linux testbed machine (Jetson, desktop, proxy laptop) at the DGX time server.
#
#   sudo bash scripts/timesync/linux_client.sh [SERVER]          # default 10.42.0.195
#
# Uses chrony if it is installed, otherwise systemd-timesyncd (present on Ubuntu and Jetson
# Linux, no install needed). Only the DGX is used, so all roles share one clock even offline.
set -euo pipefail
SERVER="${1:-10.42.0.195}"
[ "$(id -u)" -eq 0 ] || { echo "run with sudo"; exit 1; }

if command -v chronyd >/dev/null; then
    mkdir -p /etc/chrony/conf.d
    echo "server ${SERVER} iburst prefer" > /etc/chrony/conf.d/aifp-testbed.conf
    systemctl restart chrony
    sleep 5
    chronyc sources -v | tail -n 3
    chronyc tracking | grep -E "Reference ID|System time|Leap status"
else
    mkdir -p /etc/systemd/timesyncd.conf.d
    printf "[Time]\nNTP=%s\nFallbackNTP=\n" "${SERVER}" > /etc/systemd/timesyncd.conf.d/aifp-testbed.conf
    timedatectl set-ntp true
    systemctl restart systemd-timesyncd
    sleep 5
    timedatectl timesync-status | grep -E "Server|Offset|Poll"
    timedatectl show -p NTPSynchronized
fi
