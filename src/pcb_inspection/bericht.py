"""Trainingskurven, Konfusionsmatrix und Kennzahlen nach dem Training speichern."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np


def erstelle(
    history: dict,
    y_wahr,
    y_vorhersage,
    klassen: list[str],
    ordner: Path,
    anzeigen: bool = False,
) -> None:
    if not anzeigen:
        matplotlib.use("Agg")  # ohne Fenster, z.B. auf einem Server
    import matplotlib.pyplot as plt

    ordner.mkdir(parents=True, exist_ok=True)
    trainingskurven(history, ordner / "Trainingskurven.png")
    konfusionsmatrix(y_wahr, y_vorhersage, klassen, ordner / "Normalized_Konfusionsmatrix.png")
    print(kennzahlen(y_wahr, y_vorhersage, klassen))
    print(f"Grafiken gespeichert in {ordner}")
    if anzeigen:
        plt.show()


def trainingskurven(history: dict, datei: Path) -> None:
    import matplotlib.pyplot as plt

    figur = plt.figure(figsize=(8, 6))
    plt.plot(history["accuracy"], label="Training Accuracy", linewidth=2)
    plt.plot(
        history["val_accuracy"],
        label="Validation Accuracy",
        linewidth=2,
        linestyle="--",
    )
    plt.plot(history["loss"], label="Training Loss", linewidth=2)
    plt.plot(history["val_loss"], label="Validation Loss", linewidth=2, linestyle="--")
    plt.title("Trainings- und Validierungsverlauf des PCB-Klassifikators")
    plt.xlabel("Epochen")
    plt.ylabel("Wert (Loss / Accuracy)")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle=":", alpha=0.6)
    figur.savefig(datei, dpi=300, bbox_inches="tight")


def konfusionsmatrix(y_wahr, y_vorhersage, klassen: list[str], datei: Path) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

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


def kennzahlen(y_wahr, y_vorhersage, klassen: list[str]) -> str:
    from sklearn.metrics import (
        classification_report,
        f1_score,
        precision_score,
        recall_score,
    )

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
