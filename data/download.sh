#!/usr/bin/env bash
set -euo pipefail
target="${1:?Indica un directorio para los archivos de Kaggle}"
mkdir -p "$target"
base=https://www.kaggle.com/api/v1/datasets
slug=gonzajl/riones-cyst-stone-tumor-normal-dataset
curl -fLsS --retry 2 "$base/view/$slug" -o "$target/kaggle.json"
curl -fLsS --retry 2 "$base/download/$slug?datasetVersionNumber=2" -o "$target/renal.zip.part"
python3 - "$target/renal.zip.part" <<'CHECK'
import hashlib
import sys
with open(sys.argv[1], 'rb') as source:
    digest = hashlib.file_digest(source, 'sha256').hexdigest()
expected = '7c4bad073d844775a57a235a3292570726b03603e4f0b8ed43f09e39cf7c09c3'
if digest != expected:
    raise SystemExit('El ZIP no coincide con la versión auditada. No se sustituirá el dataset.')
CHECK
unzip -tq "$target/renal.zip.part"
mv "$target/renal.zip.part" "$target/renal.zip"
unzip -nq "$target/renal.zip" -d "$target/raw"
