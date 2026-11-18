"""Tests für Grafiken und Kennzahlen nach dem Training."""
import matplotlib
import pytest

matplotlib.use('Agg')
pytest.importorskip('sklearn')

from pcb_inspection import bericht  # noqa: E402

KLASSEN = ['Bad_IC', 'Good_IC']
HISTORY = {'accuracy': [0.5, 0.8], 'val_accuracy': [0.4, 0.7],
           'loss': [1.0, 0.5], 'val_loss': [1.1, 0.6]}


def test_kennzahlen_enthalten_klassen_und_gesamtwerte():
    text = bericht.kennzahlen([0, 0, 1, 1], [0, 1, 1, 1], KLASSEN)
    assert 'Bad_IC' in text and 'Good_IC' in text
    assert 'Gesamt Precision' in text and 'Gesamt F1-Score' in text


def test_fehlende_klasse_in_val_bricht_nicht_ab():
    # Kleine val-Mengen enthalten nicht immer jede Klasse.
    bericht.kennzahlen([1, 1], [1, 1], KLASSEN)


def test_erstelle_schreibt_beide_grafiken(tmp_path):
    bericht.erstelle(HISTORY, [0, 1, 1], [0, 1, 0], KLASSEN, tmp_path / 'reports')
    namen = sorted(p.name for p in (tmp_path / 'reports').iterdir())
    assert namen == ['Normalized_Konfusionsmatrix.png', 'Trainingskurven.png']
