"""Klassennamen des Klassifikators speichern, laden und Vorhersagen zuordnen.

Bewusst ohne TensorFlow, damit diese Logik schnell und einzeln testbar bleibt.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def klassen_datei(model_path: Path) -> Path:
    """Pfad der JSON-Datei mit den Klassennamen zu einem Modell bestimmen.

    Args:
        model_path: Pfad des Modells, z. B. models/modell.keras.

    Returns:
        Derselbe Pfad mit der Endung .klassen.json, z. B.
        models/modell.klassen.json.
    """
    return model_path.with_suffix(".klassen.json")


def klassen_aus_ordner(train_dir: Path) -> list[str]:
    """Klassennamen aus den Unterordnern des Trainingsordners lesen.

    Die Reihenfolge ist alphabetisch, genau wie bei
    tf.keras.utils.image_dataset_from_directory.

    Args:
        train_dir: Ordner mit einem Unterordner pro Klasse.

    Returns:
        Die Namen der Unterordner, alphabetisch sortiert.

    Raises:
        FileNotFoundError: Wenn train_dir nicht existiert.
    """
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Trainingsordner nicht gefunden: {train_dir}")
    return sorted(p.name for p in train_dir.iterdir() if p.is_dir())


def speichere_klassen(model_path: Path, klassen: list[str]) -> None:
    """Klassennamen als JSON-Datei neben dem Modell speichern.

    Args:
        model_path: Pfad des gespeicherten Modells.
        klassen: Klassennamen in der Reihenfolge der Modellausgänge.
    """
    klassen_datei(model_path).write_text(
        json.dumps(klassen, indent=2), encoding="utf-8"
    )


def lade_klassen(model_path: Path, train_dir: Path) -> list[str]:
    """Klassennamen zu einem Modell laden.

    Zuerst wird die JSON-Datei neben dem Modell gelesen. Fehlt sie, werden die
    Klassen aus den Trainingsordnern bestimmt. Dieser Rückgriff ist nur für
    ältere Modelle gedacht, die noch ohne JSON gespeichert wurden.

    Args:
        model_path: Pfad des Modells.
        train_dir: Trainingsordner, falls die JSON-Datei fehlt.

    Returns:
        Klassennamen in der Reihenfolge der Modellausgänge.

    Raises:
        FileNotFoundError: Wenn weder die JSON-Datei noch train_dir existiert.
    """
    datei = klassen_datei(model_path)
    if datei.is_file():
        return json.loads(datei.read_text(encoding="utf-8"))
    return klassen_aus_ordner(train_dir)


def interpretiere(wahrscheinlichkeiten, klassen: list[str]) -> tuple[str, float]:
    """Wahrscheinlichkeiten eines Bildes in Klasse und Konfidenz übersetzen.

    Args:
        wahrscheinlichkeiten: Softmax-Ausgabe des Modells für ein Bild, ein
            Wert pro Klasse. Liste oder Array, auch in der Form (1, n).
        klassen: Klassennamen in der Reihenfolge der Modellausgänge.

    Returns:
        Name der wahrscheinlichsten Klasse und ihre Wahrscheinlichkeit
        (0 bis 1).

    Raises:
        ValueError: Wenn die Anzahl Werte nicht zur Anzahl Klassen passt.
    """
    werte = np.asarray(wahrscheinlichkeiten, dtype=float).reshape(-1)
    if len(werte) != len(klassen):
        raise ValueError(
            f"Modell liefert {len(werte)} Werte, "
            f"aber es gibt {len(klassen)} Klassen: {klassen}"
        )
    index = int(werte.argmax())
    return klassen[index], float(werte[index])
