"""Trainingskurven, Konfusionsmatrix und Kennzahlen nach dem Training speichern."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


def erstelle(
    history: dict,
    y_wahr,
    y_vorhersage,
    klassen: list[str],
    ordner: Path,
    anzeigen: bool = False,
) -> None:
    """Grafiken in ordner speichern und Kennzahlen ausgeben.

    Schreibt Trainingskurven.png und Normalized_Konfusionsmatrix.png und gibt
    den Classification Report auf der Konsole aus.

    Args:
        history: history.history aus model.fit (accuracy, loss, val_...).
        y_wahr: Wahre Klassenindizes der Validierungsbilder (Array oder Liste).
        y_vorhersage: Vorhergesagte Klassenindizes in derselben Reihenfolge.
        klassen: Klassennamen, Index i gehört zu klassen[i].
        ordner: Zielordner, wird bei Bedarf angelegt.
        anzeigen: Grafiken zusätzlich in einem Fenster zeigen.
    """
    ordner.mkdir(parents=True, exist_ok=True)
    figuren = [
        trainingskurven(history, ordner / "Trainingskurven.png"),
        konfusionsmatrix(
            y_wahr, y_vorhersage, klassen, ordner / "Normalized_Konfusionsmatrix.png"
        ),
    ]
    print(kennzahlen(y_wahr, y_vorhersage, klassen))
    print(f"Grafiken gespeichert in {ordner}")
    if anzeigen:
        plt.show()
    for figur in figuren:
        plt.close(figur)


def trainingskurven(history: dict, datei: Path) -> plt.Figure:
    """Accuracy und Loss für Training und Validierung über die Epochen plotten.

    Args:
        history: history.history aus model.fit mit den Schlüsseln accuracy,
            val_accuracy, loss und val_loss.
        datei: Zieldatei für die Grafik (PNG, 300 dpi).

    Returns:
        Die erzeugte Figur. Der Aufrufer schliesst sie mit plt.close.
    """
    figur, achse = plt.subplots(figsize=(8, 6))
    achse.plot(history["accuracy"], label="Training Accuracy", linewidth=2)
    achse.plot(
        history["val_accuracy"],
        label="Validation Accuracy",
        linewidth=2,
        linestyle="--",
    )
    achse.plot(history["loss"], label="Training Loss", linewidth=2)
    achse.plot(
        history["val_loss"], label="Validation Loss", linewidth=2, linestyle="--"
    )
    achse.set_title("Trainings- und Validierungsverlauf des PCB-Klassifikators")
    achse.set_xlabel("Epochen")
    achse.set_ylabel("Wert (Loss / Accuracy)")
    achse.legend(loc="upper right")
    achse.grid(True, linestyle=":", alpha=0.6)
    figur.savefig(datei, dpi=300, bbox_inches="tight")
    return figur


def konfusionsmatrix(
    y_wahr, y_vorhersage, klassen: list[str], datei: Path
) -> plt.Figure:
    """Zeilenweise normalisierte Konfusionsmatrix zeichnen und speichern.

    Jede Zeile zeigt, wie sich die Bilder einer wahren Klasse auf die
    vorhergesagten Klassen verteilen. Jede Zeile summiert sich zu 1.

    Args:
        y_wahr: Wahre Klassenindizes.
        y_vorhersage: Vorhergesagte Klassenindizes in derselben Reihenfolge.
        klassen: Klassennamen, Index i gehört zu klassen[i].
        datei: Zieldatei für die Grafik (PNG, 300 dpi).

    Returns:
        Die erzeugte Figur. Der Aufrufer schliesst sie mit plt.close.
    """
    figur, achse = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay.from_predictions(
        y_wahr,
        y_vorhersage,
        labels=np.arange(len(klassen)),
        display_labels=klassen,
        normalize="true",
        cmap=plt.cm.Blues,
        ax=achse,
        values_format=".2f",
    )
    achse.set_title("Normalized Confusion Matrix (PCB-Klassifikation)")
    plt.setp(achse.get_xticklabels(), rotation=45)
    figur.tight_layout()
    figur.savefig(datei, dpi=300, bbox_inches="tight")
    return figur


def kennzahlen(y_wahr, y_vorhersage, klassen: list[str]) -> str:
    """Classification Report und gewichtete Gesamtwerte als Text zusammenstellen.

    Args:
        y_wahr: Wahre Klassenindizes.
        y_vorhersage: Vorhergesagte Klassenindizes in derselben Reihenfolge.
        klassen: Klassennamen, Index i gehört zu klassen[i].

    Returns:
        Mehrzeiliger Text: zuerst der Report pro Klasse, danach Precision,
        Recall und F1, gewichtet nach der Anzahl Bilder pro Klasse.
    """
    labels = np.arange(len(klassen))
    text = [
        classification_report(
            y_wahr, y_vorhersage, labels=labels, target_names=klassen, zero_division=0
        )
    ]
    for name, funktion in (
        ("Precision", precision_score),
        ("Recall", recall_score),
        ("F1-Score", f1_score),
    ):
        wert = funktion(y_wahr, y_vorhersage, average="weighted", zero_division=0)
        text.append(f"Gesamt {name + ':':<11}{wert:.4f}")
    return "\n".join(text)
