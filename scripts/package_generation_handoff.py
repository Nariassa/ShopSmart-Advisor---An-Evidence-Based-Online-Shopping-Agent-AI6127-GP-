"""Create a checked local ZIP after the generation batch has finished."""
import hashlib
import argparse
import json
from pathlib import Path
import shutil
from zipfile import ZIP_DEFLATED, ZipFile

from audit_generation_handoff import audit
from handoff_readme import render


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, default=ROOT / 'outputs/member_c_generation_testing_revised_final_2026-10-15')
    parser.add_argument('--preserve-report-docs', action='store_true',
                        help='Keep the experiment-specific report documents already in the package.')
    args = parser.parse_args()
    DIRECTORY = args.directory.resolve()
    if audit(DIRECTORY):
        raise SystemExit('Artifact audit failed; archive was not created.')
    summary = json.loads((DIRECTORY / 'summary.json').read_text(encoding='utf-8'))
    for filename in ['GENERATION_TESTING.md', 'MEMBER_C_REPORT.md']:
        if not args.preserve_report_docs or not (DIRECTORY / filename).exists():
            shutil.copyfile(ROOT / 'docs' / filename, DIRECTORY / filename)
    tools = DIRECTORY / 'tools'
    tools.mkdir(exist_ok=True)
    for filename in ['audit_generation_handoff.py', 'package_generation_handoff.py', 'compare_generation_revision.py', 'run_rewrite_challenge.py', 'handoff_readme.py']:
        shutil.copyfile(ROOT / 'scripts' / filename, tools / filename)
    for filename in ['requirements.txt', 'requirements-generation.txt']:
        shutil.copyfile(ROOT / filename, DIRECTORY / filename)
    (DIRECTORY / 'README.md').write_text(render(DIRECTORY, summary), encoding='utf-8')
    files = sorted(p for p in DIRECTORY.rglob('*') if p.is_file() and p.name != 'SHA256SUMS.txt')
    if any(p.suffix == '.tmp' for p in files):
        raise SystemExit('Unfinished temporary files found; archive was not created.')
    hashes = {p.relative_to(DIRECTORY).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (DIRECTORY / 'SHA256SUMS.txt').write_text(''.join(digest + '  ' + name + chr(10) for name, digest in hashes.items()), encoding='utf-8')
    archive = DIRECTORY.with_suffix('.zip')
    with ZipFile(archive, 'w', compression=ZIP_DEFLATED) as handle:
        for path in sorted(p for p in DIRECTORY.rglob('*') if p.is_file()):
            handle.write(path, DIRECTORY.name + '/' + path.relative_to(DIRECTORY).as_posix())
    with ZipFile(archive) as handle:
        assert handle.testzip() is None
        for name, digest in hashes.items():
            assert hashlib.sha256(handle.read(DIRECTORY.name + '/' + name)).hexdigest() == digest
        member_count = len(handle.namelist())
    receipt = {'archive':str(archive), 'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(), 'size_bytes':archive.stat().st_size, 'members':member_count, 'all_archived_file_hashes_verified':True}
    archive.with_suffix('.zip.audit.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
