"""Resolve Composer checkpoints without lexicographic epoch ordering."""
from pathlib import Path
import re


def resolve_checkpoint(directory, label='latest'):
    directory=Path(directory)
    def order(path):
        match=re.search(r'ep(\d+)(?:-ba(\d+))?',path.name)
        return (int(match[1]),int(match[2] or -1)) if match else (-1,-1)
    if label=='latest':
        latest=directory/'latest-rank0.pt'
        if latest.is_file():
            return latest.resolve()
        matches=list(directory.glob('ep*.pt'))
    else:
        explicit=directory/label
        for path in [explicit, directory/(label+'.pt')]:
            if path.is_file():
                return path.resolve()
        base=label.removesuffix('.pt').removesuffix('-rank0')
        matches=list(directory.glob(base+'-ba*-rank0.pt'))
        exact=directory/(base+'-rank0.pt')
        if exact.is_file():matches.append(exact)
    if not matches:
        raise FileNotFoundError(f'No checkpoint matches {label!r} in {directory}')
    return max(matches,key=order).resolve()
