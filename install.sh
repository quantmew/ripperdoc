#!/usr/bin/env bash
set -euo pipefail

REPO="quantmew/ripperdoc"
API_URL="https://api.github.com/repos/${REPO}"
RELEASE_BASE="https://github.com/${REPO}/releases/download"
DEFAULT_BIN_DIR="${HOME}/.local/bin"
DATA_DIR="${HOME}/.local/share/ripperdoc"
VERSIONS_DIR="${DATA_DIR}/versions"
STATE_FILE="${DATA_DIR}/version"

usage() {
  cat <<'USAGE'
Usage:
  ./install.sh [version|latest|stable]           Install a specific or latest ripperdoc
  ./install.sh --uninstall                      Uninstall current active ripperdoc

Options:
  --bin-dir PATH    Directory for ripperdoc entrypoint (default: ~/.local/bin)
  --uninstall       Remove ripperdoc entry and current version files
  --help            Show help

Version format:
  latest, stable, vX.Y.Z, X.Y.Z
USAGE
}

die() {
  echo "Error: $*" >&2
  usage >&2
  exit 1
}

download_text() {
  local url=$1
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- "$url"
  else
    die "curl or wget is required"
  fi
}

download_file() {
  local url=$1
  local output=$2
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL -o "$output" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -q -O "$output" "$url"
  else
    die "curl or wget is required"
  fi
}

get_latest_tag() {
  local payload
  payload=$(download_text "${API_URL}/releases/latest")
  if command -v jq >/dev/null 2>&1; then
    echo "$payload" | jq -r '.tag_name'
  else
    echo "$payload" | sed -n 's/.*\"tag_name\"[[:space:]]*:[[:space:]]*\"\\([^\\"]*\\)\".*/\\1/p' | head -n 1
  fi
}

resolve_version() {
  local version=$1
  if [[ "$version" == "latest" || "$version" == "stable" ]]; then
    version="$(get_latest_tag)"
  elif [[ "$version" != v* ]]; then
    version="v${version}"
  fi

  if [[ ! "$version" =~ ^v[0-9]+\.[0-9]+\.[0-9]+([-.][A-Za-z0-9._-]+)?$ ]]; then
    die "Invalid version: $version"
  fi
  echo "$version"
}

platform() {
  case "$(uname -s)" in
    Linux) os="linux" ;;
    Darwin) os="macos" ;;
    *) die "Unsupported OS: $(uname -s)" ;;
  esac

  case "$(uname -m)" in
    x86_64|amd64) arch="x86_64" ;;
    aarch64|arm64) arch="arm64" ;;
    *) die "Unsupported architecture: $(uname -m)" ;;
  esac

  echo "${os}-${arch}"
}

install_link() {
  local link_path=$1
  local target_path=$2

  mkdir -p "$(dirname "$link_path")"
  rm -f "$link_path"
  mkdir -p "$(dirname "$target_path")"

  if ln -sf "$target_path" "$link_path"; then
    return 0
  fi

  cp "$target_path" "$link_path"
}

read_current_version() {
  if [[ -f "$STATE_FILE" ]]; then
    cat "$STATE_FILE"
  else
    echo ""
  fi
}

do_install() {
  local version=$1
  local bin_dir=$2
  local platform
  local archive
  local url
  local tmpdir
  local archive_path
  local extracted
  local source_binary
  local version_dir
  local version_binary
  local link_path

  platform=$(platform)
  archive="ripperdoc-${version}-${platform}.tar.gz"
  url="${RELEASE_BASE}/${version}/${archive}"

  tmpdir="$(mktemp -d)"
  trap 'rm -rf "$tmpdir"' EXIT

  archive_path="${tmpdir}/${archive}"
  echo "Downloading ${url} ..."
  download_file "$url" "$archive_path"

  tar -xzf "$archive_path" -C "$tmpdir"
  source_binary="$(find "$tmpdir" -type f -name ripperdoc | head -n 1)"
  if [[ -z "$source_binary" ]]; then
    die "Could not find ripperdoc binary in archive"
  fi

  version_dir="${VERSIONS_DIR}/${version}"
  rm -rf "$version_dir"
  mkdir -p "$version_dir"
  version_binary="${version_dir}/ripperdoc"
  cp "$source_binary" "$version_binary"
  chmod +x "$version_binary"

  link_path="${bin_dir}/ripperdoc"
  install_link "$link_path" "$version_binary"

  mkdir -p "$DATA_DIR"
  echo "$version" > "$STATE_FILE"

  echo "Installed ripperdoc ${version}"
  echo "Binary: ${version_binary}"
  echo "Command: ${link_path}"
}

do_uninstall() {
  local bin_dir=$1
  local current_version
  local version_dir

  current_version="$(read_current_version)"
  rm -f "${bin_dir}/ripperdoc"

  if [[ -n "${current_version}" ]]; then
    version_dir="${VERSIONS_DIR}/${current_version}"
    rm -rf "$version_dir"
    rm -f "$STATE_FILE"
    echo "Uninstalled ripperdoc ${current_version}"
  else
    echo "No tracked version found, removed ${bin_dir}/ripperdoc"
  fi
}

ACTION="install"
VERSION="latest"
BIN_DIR="${RIPPERDOC_INSTALL_DIR:-$DEFAULT_BIN_DIR}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h)
      usage
      exit 0
      ;;
    --uninstall)
      ACTION="uninstall"
      shift
      ;;
    --bin-dir)
      [[ $# -ge 2 ]] || die "--bin-dir requires a path"
      BIN_DIR=$2
      shift 2
      ;;
    --*)
      die "Unknown option: $1"
      ;;
    *)
      VERSION=$1
      shift
      ;;
  esac
done

if [[ "$ACTION" == "uninstall" ]]; then
  do_uninstall "$BIN_DIR"
  exit 0
fi

VERSION=$(resolve_version "$VERSION")
do_install "$VERSION" "$BIN_DIR"
