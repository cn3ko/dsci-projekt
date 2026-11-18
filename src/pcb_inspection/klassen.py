"""Klassennamen des Klassifikators speichern, laden und Vorhersagen zuordnen.

Bewusst ohne TensorFlow, damit diese Logik schnell und einzeln testbar bleibt.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def klassen_datei(model_path: Path) -> Path:
    """JSON-Datei mit den Klassennamen: modell.keras -> modell.klassen.json."""
    return model_path.with_suffix(".klassen.json")


def klassen_aus_ordner(train_dir: Path) -> list[str]:
    """Klassen aus den Unterordnern, alphabetisch wie bei Keras."""
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Trainingsordner nicht gefunden: {train_dir}")
    return sorted(p.name for p in train_dir.iterdir() if p.is_dir())


def speichere_klassen(model_path: Path, klassen: list[str]) -> None:
    """Klassennamen in der Reihenfolge der Modellausgänge neben dem Modell ablegen."""
    klassen_datei(model_path).write_text(
        json.dumps(klassen, indent=2), encoding="utf-8"
    )


def lade_klassen(model_path: Path, train_dir: Path) -> list[str]:
    """Klassen aus der JSON-Datei neben dem Modell, sonst aus den Trainingsordnern.

    Der Rückgriff auf die Ordner ist nur für ältere Modelle gedacht, die noch
    ohne JSON gespeichert wurden.
    """
    datei = klassen_datei(model_path)
    if datei.is_file():
        return json.loads(datei.read_text(encoding="utf-8"))
    return klassen_aus_ordner(train_dir)


def interpretiere(wahrscheinlichkeiten, klassen: list[str]) -> tuple[str, float]:
    """Wahrscheinlichkeiten eines Bildes in (Klasse, Konfidenz) übersetzen.

    Raises:
        ValueError: wenn Modell und Klassenliste nicht zusammenpassen.
    """
    werte = np.asarray(wahrscheinlichkeiten, dtype=float).reshape(-1)
    if len(werte) != len(klassen):
        raise ValueError(
            f"Modell liefert {len(werte)} Werte, "
            f"aber es gibt {len(klassen)} Klassen: {klassen}"
        )
    index = int(werte.argmax())
    return klassen[index], float(werte[index])
