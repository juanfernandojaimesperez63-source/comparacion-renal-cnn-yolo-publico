"""Divide el ZIP auditado en partes de 80 MiB para su transferencia a Drive."""
import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_SHA256 = '7c4bad073d844775a57a235a3292570726b03603e4f0b8ed43f09e39cf7c09c3'
PART_BYTES = 80 * 1024 * 1024


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    with args.archive.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != EXPECTED_SHA256:
        parser.error('El ZIP no coincide con la versión 2 auditada.')
    count = (args.archive.stat().st_size + PART_BYTES - 1) // PART_BYTES
    paths = [args.output/f'renal.zip.part{index:02}' for index in range(1, count+1)]
    if any(path.exists() for path in paths):
        parser.error('Ya existen partes en el destino; utiliza otra carpeta.')
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    with args.archive.open('rb') as stream:
        for path in paths:
            chunk = stream.read(PART_BYTES)
            with path.open('xb') as target:
                target.write(chunk)
            records.append({'name': path.name, 'bytes': len(chunk), 'sha256': hashlib.sha256(chunk).hexdigest()})
    (args.output/'parts.json').write_text(json.dumps({'archive_sha256': digest, 'parts': records}, indent=2)+'\n')
    print(f'Preparadas {len(records)} partes del ZIP verificado en {args.output}.')


if __name__ == '__main__':
    main()
