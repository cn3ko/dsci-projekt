"""Tests für Klassennamen und Auswertung der Vorhersage (ohne TensorFlow)."""

import json

import pytest

from pcb_inspection.klassifikator import (
    interpretiere,
    klassen_aus_ordner,
    klassen_datei,
    lade_klassen,
    speichere_klassen,
)


@pytest.fixture
def train_dir(tmp_path):
    ordner = tmp_path / "train"
    for name in ["Good_Platine", "Bad_IC", "Good_IC", "Bad_Platine"]:
        (ordner / name).mkdir(parents=True)
    (ordner / "notiz.txt").write_text("keine Klasse")
    return ordner


def test_klassen_alphabetisch_wie_keras(train_dir):
    assert klassen_aus_ordner(train_dir) == ["Bad_IC", "Bad_Platine", "Good_IC", "Good_Platine"]


def test_fehlender_trainingsordner(tmp_path):
    with pytest.raises(FileNotFoundError):
        klassen_aus_ordner(tmp_path / "gibt_es_nicht")


def test_klassen_werden_neben_modell_gespeichert(tmp_path, train_dir):
    modell = tmp_path / "modell.keras"
    speichere_klassen(modell, ["b", "a"])
    assert klassen_datei(modell).name == "modell.klassen.json"
    # Gespeicherte Reihenfolge gewinnt gegen die Ordner
    assert lade_klassen(modell, train_dir) == ["b", "a"]


def test_altes_modell_ohne_json_nutzt_ordner(tmp_path, train_dir):
    assert lade_klassen(tmp_path / "alt.keras", train_dir) == klassen_aus_ordner(train_dir)


def test_interpretiere_waehlt_hoechsten_wert():
    assert interpretiere([[0.1, 0.7, 0.2]], ["a", "b", "c"]) == ("b", pytest.approx(0.7))


def test_interpretiere_meldet_falsche_klassenzahl():
    # Regressionstest: testfile.py nahm 2 Klassen an, das Modell hat 4 Ausgänge.
    with pytest.raises(ValueError, match="4 Werte"):
        interpretiere([0.1, 0.2, 0.3, 0.4], ["Bad_Platine", "Good_Platine"])


def test_json_ist_lesbar(tmp_path):
    modell = tmp_path / "m.keras"
    speichere_klassen(modell, ["Bad_IC", "Good_IC"])
    assert json.loads(klassen_datei(modell).read_text(encoding="utf-8")) == ["Bad_IC", "Good_IC"]
