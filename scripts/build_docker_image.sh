#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage:
  build_docker_image.sh [--push] <image-repo> [version]
  build_docker_image.sh [--push] <image-repo:version>

The script builds from the git archive and tags the image with:
  - <version> (from argument, exact git tag, or "dev")
  - sha-<short_sha>
  - <major>, <major>.<minor>, latest (only for semver versions)

Use --push to push tags after a successful build.
EOF
}

push=false
args=()
for arg in "$@"; do
  case "$arg" in
    --push)
      push=true
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --*)
      echo "Error: unknown option: $arg" >&2
      usage
      exit 2
      ;;
    *)
      args+=("$arg")
      ;;
  esac
done

if [[ ${#args[@]} -lt 1 || ${#args[@]} -gt 2 ]]; then
  usage
  exit 2
fi

image_spec="${args[0]}"
version_override="${args[1]:-}"

if [[ -z "$version_override" && "$image_spec" == *":"* ]]; then
  tag_part="${image_spec##*:}"
  repo_part="${image_spec%:*}"
  if [[ "$tag_part" != *"/"* ]]; then
    image_spec="$repo_part"
    version_override="$tag_part"
  fi
fi

image_repo="$image_spec"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v git >/dev/null 2>&1; then
  echo "Error: git is required to build from tracked files." >&2
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "Error: docker is required to build the image." >&2
  exit 1
fi

cd "$repo_root"
if [[ -z "$version_override" ]]; then
  git_tag="$(git describe --tags --exact-match 2>/dev/null || true)"
  if [[ -n "$git_tag" ]]; then
    version_override="${git_tag#v}"
  else
    version_override="dev"
  fi
fi

version_tag="$version_override"
sha_tag="$(git rev-parse --short HEAD)"

tags=("$image_repo:$version_tag" "$image_repo:sha-$sha_tag")
if [[ "$version_tag" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  major="${version_tag%%.*}"
  minor="${version_tag%.*}"
  tags+=("$image_repo:$minor" "$image_repo:$major" "$image_repo:latest")
fi

printf 'Building image with tags:\n' >&2
for tag in "${tags[@]}"; do
  printf '  %s\n' "$tag" >&2
done

build_args=()
for tag in "${tags[@]}"; do
  build_args+=("-t" "$tag")
done

git archive --format=tar HEAD | docker build -f Dockerfile "${build_args[@]}" -

if [[ "$push" == true ]]; then
  printf 'Pushing tags:\n' >&2
  for tag in "${tags[@]}"; do
    printf '  %s\n' "$tag" >&2
    docker push "$tag"
  done
fi
