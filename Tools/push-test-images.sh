#!/usr/bin/env bash
# Copies test images to a connected Quest (or phone) for In-Device Local File Inference:
# Pictures/HoloCube Test Dataset on the device's shared storage, then registers them
# with the media store so they also appear in the device's gallery and file picker.
#
# Usage: bash Tools/push-test-images.sh <folder with .jpg/.jpeg/.png files> [adb serial]
set -euo pipefail
source_dir="${1:?folder with test images}"
serial="${2:-}"
adb_args=()
if [[ -n "$serial" ]]; then adb_args=(-s "$serial"); fi
target="/sdcard/Pictures/HoloCube Test Dataset"

shopt -s nullglob nocaseglob
files=("$source_dir"/*.jpg "$source_dir"/*.jpeg "$source_dir"/*.png)
if (( ${#files[@]} == 0 )); then
    echo "No .jpg/.jpeg/.png files in $source_dir" >&2
    exit 1
fi
adb "${adb_args[@]}" shell mkdir -p "\"$target\""
for file in "${files[@]}"; do
    adb "${adb_args[@]}" push "$file" "$target/$(basename "$file")"
    adb "${adb_args[@]}" shell content call --uri content://media/external/file --method scan_file \
        --arg "\"/storage/emulated/0/Pictures/HoloCube Test Dataset/$(basename "$file")\"" >/dev/null 2>&1 || true
done
echo "${#files[@]} images in $target. In HoloCube: Settings → right trigger (Quest) or Settings → Model Testing (phone)."
