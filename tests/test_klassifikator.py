"""Tests für die Trainingsoptionen des Klassifikators (ohne echtes Training)."""

import pytest

pytest.importorskip("tensorflow")

from pcb_inspection.klassifikator import Config, berechne_klassengewichte  # noqa: E402


def lege_bilder_an(ordner, anzahl):
    ordner.mkdir(parents=True)
    for i in range(anzahl):
        (ordner / f"bild_{i}.png").write_bytes(b"")


def test_seltene_klasse_bekommt_hoeheres_gewicht(tmp_path):
    lege_bilder_an(tmp_path / "Bad_IC", 10)
    lege_bilder_an(tmp_path / "Good_IC", 30)
    gewichte = berechne_klassengewichte(tmp_path, ["Bad_IC", "Good_IC"])
    # 40 Bilder, 2 Klassen: 40 / (2 * 10) = 2.0 und 40 / (2 * 30) = 0.67
    assert gewichte[0] == pytest.approx(2.0)
    assert gewichte[1] == pytest.approx(40 / 60)


def test_gleich_grosse_klassen_haben_gewicht_eins(tmp_path):
    for klasse in ["a", "b", "c"]:
        lege_bilder_an(tmp_path / klasse, 5)
    assert berechne_klassengewichte(tmp_path, ["a", "b", "c"]) == {
        0: 1.0,
        1: 1.0,
        2: 1.0,
    }


def test_nur_bilder_werden_gezaehlt(tmp_path):
    lege_bilder_an(tmp_path / "a", 2)
    (tmp_path / "a" / "notiz.txt").write_text("kein Bild")
    lege_bilder_an(tmp_path / "b", 2)
    assert berechne_klassengewichte(tmp_path, ["a", "b"]) == {0: 1.0, 1: 1.0}


def test_leere_klasse_gibt_klaren_fehler(tmp_path):
    lege_bilder_an(tmp_path / "a", 3)
    (tmp_path / "b").mkdir()
    with pytest.raises(ValueError, match="ohne Bilder"):
        berechne_klassengewichte(tmp_path, ["a", "b"])


def test_ohne_optionen_bleibt_training_wie_bisher():
    config = Config()
    assert not config.augmentation
    assert not config.klassengewichte
    assert config.early_stopping == 0
