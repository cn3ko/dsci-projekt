"""Prüfungen am echten Datensatz unter data/datensatz/.

Diese Tests laufen nicht standardmässig, sondern mit:  pytest -m daten
Sie zeigen Probleme in den Daten, nicht im Code. Ein Fehlschlag heisst:
die Validierungswerte sind wahrscheinlich zu optimistisch.
"""
from collections import defaultdict
import hashlib
from pathlib import Path
import re

import pytest

from pcb_inspection import paths

pytestmark = pytest.mark.daten

BILDER = ('.png', '.jpg', '.jpeg')


def bilder(teil: str) -> list[Path]:
    ordner = paths.DATENSATZ / teil
    if not ordner.is_dir():
        pytest.skip(f'{ordner} fehlt')
    return sorted(p for p in ordner.rglob('*') if p.suffix.lower() in BILDER)


def quellfoto(datei: Path) -> str:
    """Aus welchem Originalfoto stammt ein Bild? IMG_0028gehaeuse_001 -> IMG_28."""
    treffer = re.search(r'IMG_0*(\d+)|sample_ic_auf_ruecken(?:_(\d+))?', datei.stem)
    if not treffer:
        return datei.stem
    if treffer.group(1):
        return f'IMG_{treffer.group(1)}'
    return f'ruecken_{treffer.group(2) or 1}'


def pruefsumme(datei: Path) -> str:
    return hashlib.md5(datei.read_bytes()).hexdigest()


def test_quellfoto_erkennung():
    assert quellfoto(Path('IMG_0028gehaeuse_001.png')) == 'IMG_28'
    assert quellfoto(Path('bad_IMG_0035_crop.png')) == 'IMG_35'
    assert quellfoto(Path('sample_ic_auf_ruecken_2_gehaeuse_005.png')) == 'ruecken_2'


def test_kein_foto_in_train_und_val():
    """Ausschnitte desselben Fotos gehören komplett in train ODER val."""
    train = {quellfoto(p) for p in bilder('train')}
    val = {quellfoto(p) for p in bilder('val')}
    gemeinsam = sorted(train & val)
    assert not gemeinsam, f'Fotos in train und val: {gemeinsam}'


def test_keine_doppelten_bilder():
    nach_inhalt = defaultdict(list)
    for p in bilder('train') + bilder('val'):
        nach_inhalt[pruefsumme(p)].append(p.relative_to(paths.DATENSATZ).as_posix())
    doppelt = [namen for namen in nach_inhalt.values() if len(namen) > 1]
    assert not doppelt, f'{len(doppelt)} Bilder mehrfach, z.B. {doppelt[:3]}'


def test_train_und_val_haben_dieselben_klassen():
    klassen = {teil: sorted(p.name for p in (paths.DATENSATZ / teil).iterdir() if p.is_dir())
               for teil in ('train', 'val')}
    assert klassen['train'] == klassen['val']
