import os
import json
import glob
import argparse
import math

def average_json_values(json_dir, target_file='*.json', output_file='summary_all.json', selected_key=None):
    """Average scalar fields from equally weighted JSON records.

    Nested diagnostics and action lists are not numerical summary fields.
    A missing scalar value remains unknown instead of reducing its denominator.
    """
    destination = os.path.abspath(os.path.join(json_dir, output_file))
    matches = set()
    for level in ('', '*', os.path.join('*', '*')):
        matches.update(glob.glob(os.path.join(json_dir, level, target_file)))
    json_files = sorted(path for path in matches if os.path.abspath(path) != destination)
    if not json_files:
        raise ValueError(f'No input JSON files match {target_file!r} under {json_dir}')
    values_sum, counts, candidates = {}, {}, set()
    for json_file in json_files:
        with open(json_file, encoding='utf-8') as stream:
            data = json.load(stream)
        if not isinstance(data, dict):
            raise ValueError(f'Expected a JSON object: {json_file}')
        for key, value in data.items():
            if selected_key is not None and key != selected_key:
                continue
            if isinstance(value, list) and len(value) == 1:
                value = value[0]
            if value is None:
                candidates.add(key)
            elif isinstance(value, (int, float)):
                if not math.isfinite(value):
                    raise ValueError(f'Non-finite {key!r} in {json_file}')
                candidates.add(key)
                values_sum[key] = values_sum.get(key, 0.0) + value
                counts[key] = counts.get(key, 0) + 1
    averages = {
        key: values_sum[key] / len(json_files) if counts.get(key) == len(json_files) else None
        for key in sorted(candidates)
    }
    with open(destination, 'w', encoding='utf-8') as stream:
        json.dump(averages, stream, indent=4, allow_nan=False)
    print(json.dumps(averages, indent=4))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process JSON files to compute average values.')
    parser.add_argument('--directory', type=str, required=True, help='Path to the directory containing JSON files')
    parser.add_argument('--target_file', default='*.json', type=str, help='target file name')
    parser.add_argument('--output_file', default='summary_all.json', type=str, help='output file name')
    args = parser.parse_args()

    average_json_values(args.directory, args.target_file, args.output_file)
