"""Tests für die Trainingsoptionen des Klassifikators (ohne echtes Training)."""

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("tensorflow")

import keras  # noqa: E402

from pcb_inspection.klassifikator import (  # noqa: E402
    Config,
    berechne_klassengewichte,
    lade_bild,
)


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


def test_vorhersage_laedt_bild_wie_training(tmp_path):
    # Zufallsbild mit feinen Details, damit nearest und bilinear verschieden sind
    pixel = np.random.default_rng(0).integers(0, 256, (300, 400, 3), dtype=np.uint8)
    ordner = tmp_path / "klasse"
    ordner.mkdir()
    Image.fromarray(pixel).save(ordner / "bild.png")

    datensatz = keras.utils.image_dataset_from_directory(
        tmp_path, image_size=(224, 224), batch_size=1, shuffle=False
    )
    wie_training = next(iter(datensatz))[0][0].numpy()

    assert np.allclose(lade_bild(ordner / "bild.png", (224, 224)), wie_training)


def test_ohne_optionen_bleibt_training_wie_bisher():
    config = Config()
    assert not config.augmentation
    assert not config.klassengewichte
    assert config.early_stopping == 0
