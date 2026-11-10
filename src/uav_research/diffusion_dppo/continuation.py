"""Fork a complete online checkpoint without overwriting the original run."""
import csv
import json
from pathlib import Path
import shutil

from ..runtime import atomic_json
from ..topk_diagnostic import file_hash
from .storage import atomic_directory, load


def prepare_continuation(cfg, checkpoint, output):
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    saved = load(checkpoint, cfg, {'dppo_full'})
    source = checkpoint.parent.parent
    if output == source:
        raise ValueError('Continuation must use a new output; preserve the source run')
    if saved['env_steps'] >= cfg.run.total_steps:
        raise ValueError('Continuation budget must exceed checkpoint steps')
    lineage = dict(source_run=str(source), source_checkpoint=str(checkpoint),
        source_checkpoint_sha256=file_hash(checkpoint/'metadata.pt'),
        source_steps=saved['env_steps'], source_updates=saved['learner']['updates'],
        total_steps=cfg.run.total_steps, additional_steps=cfg.run.total_steps-saved['env_steps'],
        note='Copied CSV prefix and original step0; full state resumes via --resume. No pretraining.')
    marker = output/'continuation.json'
    if marker.exists():
        if json.loads(marker.read_text()) != lineage:
            raise ValueError('Output belongs to another continuation')
        return lineage
    if output.exists():
        raise FileExistsError('Output exists without a continuation marker; use a new output')
    if not (source/'step0.pt').is_file():
        raise FileNotFoundError('Source step0 is required for comparable continuation reports')
    with atomic_directory(output) as temporary:
        shutil.copy2(source/'step0.pt', temporary/'step0.pt')
        for name in ('metrics.csv', 'validation.csv'):
            with (source/name).open(newline='') as stream:
                reader = csv.DictReader(stream)
                with (temporary/name).open('w', newline='') as dest:
                    writer = csv.DictWriter(dest, fieldnames=reader.fieldnames)
                    writer.writeheader()
                    for row in reader:
                        if int(row['env_steps']) <= saved['env_steps']:
                            writer.writerow(row)
        if (source/'manifest.json').exists():
            shutil.copy2(source/'manifest.json', temporary/'source_manifest.json')
        atomic_json(temporary/'continuation.json', lineage)
    return lineage
