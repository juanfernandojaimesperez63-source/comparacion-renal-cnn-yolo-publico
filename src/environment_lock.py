"""Registra solo las dependencias transitivas activas del proyecto instalado."""
import argparse
from collections import defaultdict, deque
from importlib.metadata import distribution
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('requirements', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    extras = defaultdict(set)
    queue = deque()
    for line in args.requirements.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        requirement = Requirement(line)
        if requirement.marker and not requirement.marker.evaluate({'extra': ''}):
            continue
        name = canonicalize_name(requirement.name)
        found = distribution(name).version
        if requirement.specifier and not requirement.specifier.contains(found, prereleases=True):
            raise RuntimeError(f'Se requiere {requirement}; está instalado {found}.')
        extras[name].update(requirement.extras)
        queue.append(name)
    processed = {}
    pins = {}
    while queue:
        name = queue.popleft()
        active_extras = frozenset(extras[name])
        if processed.get(name) == active_extras:
            continue
        processed[name] = active_extras
        installed = distribution(name)
        pins[name] = installed.version
        for text in installed.requires or ():
            requirement = Requirement(text)
            if requirement.marker and not any(requirement.marker.evaluate({'extra': extra}) for extra in ('', *active_extras)):
                continue
            dependency = canonicalize_name(requirement.name)
            found = distribution(dependency).version
            if requirement.specifier and not requirement.specifier.contains(found, prereleases=True):
                raise RuntimeError(f'{name} requiere {requirement}; está instalado {found}.')
            extras[dependency].update(requirement.extras)
            queue.append(dependency)
    args.output.write_text('# Entorno instalado para este sistema y versión de Python.\n' + '\n'.join(f'{name}=={pins[name]}' for name in sorted(pins))+'\n')
    print(f'Registradas {len(pins)} dependencias activas en {args.output}.')


if __name__ == '__main__':
    main()
