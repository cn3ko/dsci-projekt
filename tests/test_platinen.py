"""Tests für Schritt 1: Platine finden und freistellen.

Nur synthetische Bilder, die echten Daten werden nicht gebraucht.
"""

import numpy as np
import pytest
from PIL import Image

from pcb_inspection.platinen import Config, detect_board, process_image, select_images

GRUEN = (40, 140, 60)
WEISS = (245, 245, 245)


def foto_mit_platine(breite=400, hoehe=300, platine=(50, 80, 250, 280)):
    """Heller Tisch mit grüner Platine und schwarzem Chip darauf.

    platine: (oben, links, unten, rechts) in Pixeln.
    """
    bild = np.full((hoehe, breite, 3), WEISS, dtype=np.uint8)
    oben, links, unten, rechts = platine
    bild[oben:unten, links:rechts] = GRUEN
    bild[oben + 60 : oben + 100, links + 60 : links + 120] = 0  # Chip
    return bild


def test_findet_platine_mit_rand():
    cfg = Config(crop_padding=5)
    maske, (oben, links, unten, rechts) = detect_board(foto_mit_platine(), cfg)
    # Grenzen der Platine plus 5 px Rand, kleine Abweichung durch Closing erlaubt
    assert oben == pytest.approx(45, abs=2)
    assert links == pytest.approx(75, abs=2)
    assert unten == pytest.approx(255, abs=2)
    assert rechts == pytest.approx(285, abs=2)
    assert maske.shape == (300, 400)


def test_schwarzer_chip_gehoert_zur_platine():
    maske, _ = detect_board(foto_mit_platine(), Config())
    assert maske[50 + 80, 80 + 90]  # Mitte des Chips
    assert not maske[10, 10]  # Tisch


def test_verkleinerte_erkennung_trifft_originalkoordinaten():
    # Erkennung läuft auf max. 100 px; die Maske muss trotzdem zum Original passen.
    maske, (oben, links, unten, rechts) = detect_board(
        foto_mit_platine(), Config(detection_max_side=100, crop_padding=0)
    )
    assert maske.shape == (300, 400)
    assert oben == pytest.approx(50, abs=5)
    assert rechts == pytest.approx(280, abs=5)


def test_ohne_gruen_gibt_klaren_fehler():
    leer = np.full((100, 100, 3), WEISS, dtype=np.uint8)
    with pytest.raises(ValueError, match="Keine grüne"):
        detect_board(leer, Config())


def test_zu_kleine_flaeche_wird_abgelehnt():
    bild = foto_mit_platine(platine=(10, 10, 14, 14))
    with pytest.raises(ValueError, match="zu klein"):
        detect_board(bild, Config(closing_radius=0))


def test_falsches_bildformat():
    with pytest.raises(ValueError, match="RGB"):
        detect_board(np.zeros((10, 10), dtype=np.uint8), Config())


@pytest.mark.parametrize(
    "falsch",
    [
        dict(detection_max_side=0),
        dict(crop_padding=-1),
        dict(min_object_fraction=0),
        dict(green_hsv_lower=(100, 0, 0), green_hsv_upper=(90, 255, 255)),
    ],
)
def test_config_prueft_werte(falsch):
    with pytest.raises(ValueError):
        Config(**falsch)


def test_process_image_schreibt_nur_freigestellt_mit_transparenz(tmp_path):
    quelle = tmp_path / "IMG_0001.png"
    Image.fromarray(foto_mit_platine()).save(quelle)
    cfg = Config(output_path=tmp_path / "out", show_preview=False)

    ergebnis = process_image(quelle, cfg)

    assert ergebnis.name == "IMG_0001_freigestellt.png"
    assert [p.name for p in (tmp_path / "out").iterdir()] == [
        "IMG_0001_freigestellt.png"
    ]
    with Image.open(ergebnis) as bild:
        assert bild.mode == "RGBA"
        assert bild.getpixel((0, 0))[3] == 0  # Rand: transparent
        assert bild.getpixel((bild.width // 2, bild.height // 2))[3] == 255


def test_quick_test_stichprobe_ist_reproduzierbar(tmp_path):
    bilder = [tmp_path / f"{i}.png" for i in range(10)]
    cfg = Config(quick_test=True, sample_size=3, random_seed=1)
    assert select_images(bilder, cfg) == select_images(bilder, cfg)
    assert len(select_images(bilder, cfg)) == 3
    assert select_images(bilder, Config()) == bilder
