#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUCKET_URL="https://omniverse-content-production.s3-us-west-2.amazonaws.com"
ISAAC_PREFIX="Assets/Isaac/5.0/Isaac/Environments"
FORCE=0
TMP_FILES=()

cleanup() {
  if [[ "${#TMP_FILES[@]}" -gt 0 ]]; then
    rm -f "${TMP_FILES[@]}"
  fi
}
trap cleanup EXIT

if [[ "${1:-}" == "--force" ]]; then
  FORCE=1
fi

download_asset() {
  local url="$1"
  local dest="$2"

  if [[ "${FORCE}" -ne 1 && -s "${dest}" ]]; then
    echo "Already exists: ${dest}"
    return
  fi

  mkdir -p "$(dirname "${dest}")"

  echo "Downloading ${url}"
  echo "        to ${dest}"

  if command -v curl >/dev/null 2>&1; then
    curl --fail --location --retry 3 --output "${dest}" "${url}"
  elif command -v wget >/dev/null 2>&1; then
    wget --tries=3 --output-document="${dest}" "${url}"
  else
    echo "Error: curl or wget is required to download assets." >&2
    return 1
  fi
}

list_s3_prefix() {
  local prefix="$1"

  python3 - "${BUCKET_URL}" "${prefix}" <<'PY'
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

bucket_url = sys.argv[1]
prefix = sys.argv[2]
token = None

while True:
    params = {
        "list-type": "2",
        "prefix": prefix,
        "max-keys": "1000",
    }
    if token:
        params["continuation-token"] = token

    url = f"{bucket_url}/?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url) as response:
        root = ET.fromstring(response.read())

    namespace = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    for item in root.findall("s3:Contents", namespace):
        key = item.findtext("s3:Key", default="", namespaces=namespace)
        if key:
            print(f"{key}\t{urllib.parse.quote(key, safe='/')}")

    if root.findtext("s3:IsTruncated", default="false", namespaces=namespace) != "true":
        break

    token = root.findtext("s3:NextContinuationToken", default="", namespaces=namespace)
    if not token:
        raise RuntimeError(f"Missing continuation token while listing {prefix}")
PY
}

is_scene_dependency() {
  local rel_path="$1"
  local scene_file="$2"
  local extra_root_files="${3:-}"

  [[ "${rel_path}" == "${scene_file}" || "${rel_path}" == Props/* || "${rel_path}" == Materials/* ]] && return 0

  [[ -n "${extra_root_files}" && " ${extra_root_files} " == *" ${rel_path} "* ]]
}

download_scene() {
  local environment="$1"
  local scene_file="$2"
  local dest_dir="$3"
  local extra_root_files="${4:-}"
  local prefix="${ISAAC_PREFIX}/${environment}/"
  local key
  local encoded_key
  local rel_path
  local dest
  local key_list

  echo "Listing ${environment}/${scene_file} assets..."
  key_list="$(mktemp)"
  TMP_FILES+=("${key_list}")
  list_s3_prefix "${prefix}" >"${key_list}"

  while IFS=$'\t' read -r key encoded_key; do
    rel_path="${key#${prefix}}"

    if [[ -z "${rel_path}" || "${rel_path}" == .thumbs/* || "${rel_path}" == */.thumbs/* ]]; then
      continue
    fi

    if ! is_scene_dependency "${rel_path}" "${scene_file}" "${extra_root_files}"; then
      continue
    fi

    dest="${dest_dir}/${rel_path}"
    download_asset "${BUCKET_URL}/${encoded_key}" "${dest}"
  done <"${key_list}"
}

download_scene "Hospital" "hospital.usd" "${SCRIPT_DIR}/Hospital"
download_scene "Simple_Warehouse" "full_warehouse.usd" "${SCRIPT_DIR}/Simple_Warehouse" "A_01.usd"

echo "Done."
echo "Hospital scene: ${SCRIPT_DIR}/Hospital/hospital.usd"
echo "Full warehouse scene: ${SCRIPT_DIR}/Simple_Warehouse/full_warehouse.usd"
