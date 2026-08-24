#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run this installer with sudo:" >&2
  echo "  sudo bash scripts/install_offload_sudoers.sh <interface> [username]" >&2
  exit 1
fi

INTERFACE="${1:-}"
TARGET_USER="${2:-${SUDO_USER:-}}"

if [[ -z "${INTERFACE}" ]]; then
  echo "Usage: sudo bash scripts/install_offload_sudoers.sh <interface> [username]" >&2
  exit 2
fi

if [[ ! "${INTERFACE}" =~ ^[A-Za-z0-9_.:-]+$ ]]; then
  echo "Invalid interface name: ${INTERFACE}" >&2
  exit 2
fi

if [[ ! -d "/sys/class/net/${INTERFACE}" ]]; then
  echo "Network interface does not exist: ${INTERFACE}" >&2
  exit 2
fi

if [[ -z "${TARGET_USER}" || ! "${TARGET_USER}" =~ ^[A-Za-z_][A-Za-z0-9_-]*[$]?$ ]]; then
  echo "Could not determine a safe target username. Pass it explicitly." >&2
  exit 2
fi

if ! id "${TARGET_USER}" >/dev/null 2>&1; then
  echo "Unknown local user: ${TARGET_USER}" >&2
  exit 2
fi

ETHTOOL="$(command -v ethtool || true)"
if [[ -z "${ETHTOOL}" ]]; then
  echo "ethtool is not installed. Install it first, e.g. sudo apt install ethtool" >&2
  exit 3
fi

if ! command -v visudo >/dev/null 2>&1; then
  echo "visudo is required to validate the sudoers rule." >&2
  exit 3
fi

SAFE_ALIAS="$(printf '%s_%s' "${TARGET_USER}" "${INTERFACE}" | tr '[:lower:].:-' '[:upper:]___' | tr -cd 'A-Z0-9_')"
ALIAS="AIFP_OFFLOAD_${SAFE_ALIAS}"
DEST="/etc/sudoers.d/ai-fingerprint-offload-${TARGET_USER}-${INTERFACE}"
TMP="$(mktemp)"
trap 'rm -f "${TMP}"' EXIT

COMMANDS=()
for FEATURE in gro gso tso lro; do
  COMMANDS+=("${ETHTOOL} -K ${INTERFACE} ${FEATURE} off")
  COMMANDS+=("${ETHTOOL} -K ${INTERFACE} ${FEATURE} on")
done

{
  echo "# AI Fingerprinting: narrowly scoped NIC-offload control"
  echo "# Generated for user=${TARGET_USER}, interface=${INTERFACE}"
  printf 'Cmnd_Alias %s = ' "${ALIAS}"
  local_sep=""
  for CMD in "${COMMANDS[@]}"; do
    printf '%s%s' "${local_sep}" "${CMD}"
    local_sep=", "
  done
  printf '\n'
  printf '%s ALL=(root) NOPASSWD: %s\n' "${TARGET_USER}" "${ALIAS}"
} > "${TMP}"

chmod 0440 "${TMP}"
visudo -cf "${TMP}" >/dev/null
install -o root -g root -m 0440 "${TMP}" "${DEST}"
visudo -cf "${DEST}" >/dev/null

echo "Installed: ${DEST}"
echo "Allowed user: ${TARGET_USER}"
echo "Allowed interface: ${INTERFACE}"
echo "Allowed operations: ethtool -K ${INTERFACE} {gro,gso,tso,lro} {on,off}"
echo
echo "Test without changing anything permanently by running the proxy."
echo "To remove the rule:"
echo "  sudo rm -f '${DEST}'"
