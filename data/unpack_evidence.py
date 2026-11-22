"""Restaura audit/ y full/ desde el archivo técnico del repositorio de GitHub."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import stat
import zipfile


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    results = Path(__file__).resolve().parents[1] / 'results'
    manifest = json.loads((results / 'evidencias_manifest.json').read_text())
    expected = manifest['files']
    if all((results / name).is_file() and digest(results / name) == info['sha256']
           for name, info in expected.items()):
        print('Las evidencias ya están completas y sus hashes coinciden.')
        return
    archive = results / 'evidencias_por_corrida.zip'
    if not archive.is_file():
        raise SystemExit('Faltan evidencias. Descarga results/evidencias_por_corrida.zip del repositorio.')
    if digest(archive) != manifest['archive_sha256']:
        raise SystemExit('El SHA-256 del archivo de evidencias no coincide.')
    with zipfile.ZipFile(archive) as zipped:
        if zipped.testzip() is not None:
            raise SystemExit('Archivo ZIP dañado.')
        if set(zipped.namelist()) != set(expected):
            raise SystemExit('El contenido no coincide con el manifiesto de evidencias.')
        for item in zipped.infolist():
            name = PurePosixPath(item.filename)
            if name.is_absolute() or '..' in name.parts or name.parts[0] not in ('audit', 'full'):
                raise SystemExit('Ruta inválida en el archivo de evidencias.')
            if stat.S_ISLNK(item.external_attr >> 16):
                raise SystemExit('No se admiten enlaces simbólicos.')
            target = results / item.filename
            if target.is_symlink() or any(parent.is_symlink() for parent in target.parents):
                raise SystemExit('La ruta de destino contiene un enlace simbólico.')
            if target.exists() and digest(target) != expected[item.filename]['sha256']:
                raise SystemExit(f'No se sobrescribe un archivo distinto: {item.filename}')
        for item in zipped.infolist():
            target = results / item.filename
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zipped.open(item) as source, target.open('xb') as destination:
                for chunk in iter(lambda: source.read(1024 * 1024), b''):
                    destination.write(chunk)
    assert all(digest(results / name) == info['sha256'] for name, info in expected.items())
    print(f'Restaurados y verificados {len(expected)} archivos técnicos de audit/ y full/.')


if __name__ == '__main__':
    main()
