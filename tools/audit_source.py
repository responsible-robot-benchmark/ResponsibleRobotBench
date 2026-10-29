"""Audit candidate release files without printing potential secret values."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rrb.release import audit_repository


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'validation/precommit-audit.json')
    args = parser.parse_args()
    result = audit_repository(ROOT)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({
        'passed': result['passed'], 'candidate_files': result['candidate_files'],
        'findings_by_kind': dict(Counter(row['kind'] for row in result['findings'])),
        'manifest_errors': len(result['manifest_errors']),
        'files_over_100_MiB': len(result['files_over_100_MiB']),
        'root_license_present': result['root_license_present'],
        'report': str(args.output),
    }, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
