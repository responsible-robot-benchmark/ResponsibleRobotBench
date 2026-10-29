"""Scan Git files/history and evidence archives without printing credential values."""
import argparse
import io
import json
from pathlib import Path
import re
import subprocess
import tarfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
LIMIT = 32 * 1024 * 1024
MEDIA = {'.png', '.jpg', '.jpeg', '.gif', '.mp4', '.webm', '.pdf', '.ttm', '.ttt',
         '.npy', '.npz', '.pt', '.pth', '.safetensors', '.so', '.woff', '.woff2'}
MARKERS = {
    'provider_key': re.compile(r'(?<![\w-])(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16})(?![\w-])'),
    'private_key': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY-----'),
}
ASSIGNMENT = re.compile(
    r'''(?ix)(?<![\w])(?:[a-z][a-z0-9]*_)*(?:api_?key|access_token|auth_token|token|password|passwd|secret|client_secret)["']?\s*[:=]\s*["']?([A-Za-z0-9_./+@=~-]{8,})''')
BEARER = re.compile(r'(?i)\bBearer\s+([A-Za-z0-9_./+~=-]{8,})')
PLACEHOLDERS = {'your_api_key', 'your_api_key_here', 'YOUR_API_KEY', 'your_token',
                'REPLACE_ME', 'changeme', 'placeholder'}


def text_findings(data, location):
    """Return locations and categories only; never include matched text."""
    if b'\x00' in data[:8192]:
        return [], False
    try:
        content = data.decode('utf-8')
    except UnicodeDecodeError:
        return [], False
    rows = []
    for line, value in enumerate(content.splitlines(), 1):
        lowered = value.lower()
        if not any(marker in lowered for marker in
                   ('key', 'token', 'password', 'passwd', 'secret', 'bearer',
                    'sk-', 'ghp_', 'gho_', 'ghu_', 'ghs_', 'ghr_', 'github_pat_', 'hf_', 'akia')):
            continue
        kinds = {kind for kind, pattern in MARKERS.items() if pattern.search(value)}
        for pattern, kind in ((ASSIGNMENT, 'credential_assignment'), (BEARER, 'bearer_literal')):
            if kind == 'bearer_literal' and 'bearer' not in lowered:
                continue
            for match in pattern.finditer(value):
                candidate = match.group(1)
                if kind == 'bearer_literal' and candidate.lower().rstrip('.') == 'authentication':
                    continue
                if (kind == 'credential_assignment' and location.endswith('.py')
                        and value[match.start(1) - 1:match.start(1)] not in {'"', "'"}):
                    # Python references such as os.getenv(...) or tensor tokens
                    # are not credential literals. Provider markers still apply.
                    continue
                if (candidate not in PLACEHOLDERS and not candidate.startswith(('os.environ', 'process.env'))
                        and not candidate.isdecimal()):
                    kinds.add(kind)
        rows.extend({'location': location, 'line': line, 'kind': kind} for kind in sorted(kinds))
    return rows, True


def inspect_blob(data, location, report, depth=0):
    """Inspect archives in memory; never extract paths or execute content."""
    name = location.lower()
    archive = name.endswith(('.tar.gz', '.tgz', '.tar', '.zip'))
    if archive:
        if depth >= 3:
            report['unscanned'].append({'location': location, 'reason': 'archive_depth_limit'})
            return
        report['archives_scanned'] += 1
        try:
            if name.endswith('.zip'):
                with zipfile.ZipFile(io.BytesIO(data)) as bundle:
                    for entry in bundle.infolist():
                        if entry.is_dir():
                            continue
                        read_member(bundle.open, entry, entry.filename, entry.file_size, location, report, depth)
            else:
                with tarfile.open(fileobj=io.BytesIO(data), mode='r:*') as bundle:
                    for entry in bundle:
                        if entry.isfile():
                            read_member(bundle.extractfile, entry, entry.name, entry.size, location, report, depth)
        except (tarfile.TarError, zipfile.BadZipFile, OSError, RuntimeError):
            report['unscanned'].append({'location': location, 'reason': 'unreadable_archive'})
        return
    if Path(location).suffix.lower() in MEDIA:
        report['binary_or_media_skipped'] += 1
        return
    if len(data) > LIMIT:
        report['unscanned'].append({'location': location, 'reason': 'text_size_limit'})
        return
    findings, is_text = text_findings(data, location)
    report['findings'].extend(findings)
    report['text_files_scanned'] += int(is_text)
    report['binary_or_media_skipped'] += int(not is_text)


def read_member(opener, entry, name, size, location, report, depth):
    member = location + '::' + name
    if Path(name).suffix.lower() in MEDIA:
        report['binary_or_media_skipped'] += 1
        return
    if size > LIMIT:
        report['unscanned'].append({'location': member, 'reason': 'archive_member_size_limit'})
        return
    with opener(entry) as stream:
        inspect_blob(stream.read(LIMIT + 1), member, report, depth + 1)


def git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root)


def scan(root, history=False):
    root = Path(root)
    report = {'scope': 'working_tree_and_reachable_history' if history else 'working_tree',
              'files_scanned': 0, 'historical_blobs_scanned': 0, 'archives_scanned': 0,
              'text_files_scanned': 0, 'binary_or_media_skipped': 0,
              'findings': [], 'unscanned': [],
              'limits': 'Heuristic credential scan; binary/media payloads and ignored local files are excluded. '
                        'Archive members and text are limited to 32 MiB; nested archives to three levels. '
                        'History means objects reachable from local refs, not remote-only or dangling objects.'}
    manifest = root / 'distribution-manifest.json'
    if manifest.is_file():
        import sys
        sys.path.insert(0, str(root))
        from rrb.release import inventory
        names = '\x00'.join(inventory(root)).encode()
        if history and not (root / '.git').exists():
            raise ValueError('History scanning requires a Git checkout; omit --history for a ZIP.')
    else:
        names = git(root, 'ls-files', '--cached', '--others', '--exclude-standard', '-z')
    for name in sorted(set(names.decode().strip('\x00').split('\x00')) - {''}):
        path = root / name
        if not path.is_file() or path.is_symlink():
            continue
        inspect_blob(path.read_bytes(), name, report)
        report['files_scanned'] += 1
    if history:
        objects = {}
        for row in git(root, 'rev-list', '--objects', '--all').decode().splitlines():
            parts = row.split(' ', 1)
            if len(parts) == 2:
                objects.setdefault(parts[0], parts[1])
        process = subprocess.Popen(['git', 'cat-file', '--batch'], cwd=root,
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        try:
            for oid, name in objects.items():
                process.stdin.write((oid + '\n').encode())
                process.stdin.flush()
                header = process.stdout.readline().decode().split()
                data = process.stdout.read(int(header[2]))
                process.stdout.read(1)
                if header[1] == 'blob':
                    inspect_blob(data, 'git:' + oid[:12] + ':' + name, report)
                    report['historical_blobs_scanned'] += 1
        finally:
            process.stdin.close()
            process.stdout.close()
            process.wait()
    report['passed'] = not (report['findings'] or report['unscanned'])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--history', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = scan(args.root, args.history)
    rendered = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end='')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
