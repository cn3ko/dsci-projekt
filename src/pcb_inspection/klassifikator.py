"""Qualitätsklassifikator (MobileNetV2, Transfer Learning) trainieren und anwenden.

Training:   python -m pcb_inspection.klassifikator train
Vorhersage: python -m pcb_inspection.klassifikator predict data/beispiele/IMG_0100.png

Die Klassen ergeben sich aus den Unterordnern von data/datensatz/train/.
Beim Training werden sie neben dem Modell gespeichert (<modell>.klassen.json),
damit die Vorhersage immer dieselbe Reihenfolge verwendet wie das Training.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf
from keras import layers, models

from pcb_inspection import bericht, paths
from pcb_inspection.klassen import interpretiere, lade_klassen, speichere_klassen


@dataclass(frozen=True)
class Config:
    """Pfade und Hyperparameter für Training und Vorhersage.

    Attributes:
        train_dir: Trainingsbilder, ein Unterordner pro Klasse.
        val_dir: Validierungsbilder mit denselben Unterordnern.
        model_path: Speicherort des Modells (.keras).
        report_dir: Zielordner für die Grafiken nach dem Training.
        image_size: Bildgrösse (Höhe, Breite), auf die alle Bilder skaliert
            werden.
        batch_size: Anzahl Bilder pro Trainingsschritt.
        epochs: Anzahl Durchläufe über die Trainingsdaten.
        learning_rate: Lernrate des Adam-Optimierers.
        show_plots: Grafiken nach dem Training zusätzlich anzeigen.
        augmentation: Trainingsbilder zufällig spiegeln, drehen und im
            Kontrast verändern, damit das Modell weniger auswendig lernt.
        klassengewichte: Seltene Klassen beim Training stärker gewichten.
        early_stopping: Abbrechen, wenn val_loss so viele Epochen lang nicht
            besser wird, und die beste Epoche behalten. 0 heisst aus.
    """

    train_dir: Path = paths.DATENSATZ / "train"
    val_dir: Path = paths.DATENSATZ / "val"
    model_path: Path = paths.MODELLE / "pcb_quality_model_tf.keras"
    report_dir: Path = paths.BERICHTE
    image_size: tuple[int, int] = (224, 224)
    batch_size: int = 30
    epochs: int = 40
    learning_rate: float = 0.001
    show_plots: bool = False
    # Ohne diese Optionen läuft das Training wie bisher (Vergleichsmodell).
    augmentation: bool = False
    klassengewichte: bool = False
    early_stopping: int = 0


BILDENDUNGEN = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}


def berechne_klassengewichte(train_dir: Path, klassen: list[str]) -> dict[int, float]:
    """Gewicht pro Klasse, umgekehrt proportional zur Anzahl Trainingsbilder.

    Formel wie bei scikit-learn "balanced": gesamt / (anzahl_klassen * n_klasse).
    Eine Klasse mit halb so vielen Bildern zählt also doppelt.

    Args:
        train_dir: Trainingsordner mit einem Unterordner pro Klasse.
        klassen: Klassennamen in der Reihenfolge der Modellausgänge.

    Returns:
        Klassenindex -> Gewicht, wie es model.fit als class_weight erwartet.

    Raises:
        ValueError: Wenn eine Klasse keine Bilder hat.
    """
    anzahlen = [
        sum(
            1
            for p in (train_dir / klasse).iterdir()
            if p.is_file() and p.suffix.lower() in BILDENDUNGEN
        )
        for klasse in klassen
    ]
    if 0 in anzahlen:
        raise ValueError(f"Klasse ohne Bilder: {dict(zip(klassen, anzahlen))}")
    gesamt = sum(anzahlen)
    return {i: gesamt / (len(klassen) * n) for i, n in enumerate(anzahlen)}


def lade_datensaetze(
    config: Config,
) -> tuple[tf.data.Dataset, tf.data.Dataset, list[str]]:
    """Trainings- und Validierungsbilder aus den Klassenordnern laden.

    Args:
        config: Ordner, Bildgrösse und Batchgrösse.

    Returns:
        Trainingsdaten (gemischt), Validierungsdaten (feste Reihenfolge) und
        die Klassennamen in der Reihenfolge der Modellausgänge.

    Raises:
        ValueError: Wenn train und val unterschiedliche Klassenordner haben.
    """
    train = keras.utils.image_dataset_from_directory(
        config.train_dir,
        image_size=config.image_size,
        batch_size=config.batch_size,
        shuffle=True,
    )
    val = keras.utils.image_dataset_from_directory(
        config.val_dir,
        image_size=config.image_size,
        batch_size=config.batch_size,
        shuffle=False,  # feste Reihenfolge, damit Labels und Vorhersagen zusammenpassen
    )
    if train.class_names != val.class_names:
        raise ValueError(
            "train und val haben unterschiedliche Klassen: "
            f"{train.class_names} / {val.class_names}"
        )
    klassen = train.class_names
    return (
        train.cache().prefetch(tf.data.AUTOTUNE),
        val.cache().prefetch(tf.data.AUTOTUNE),
        klassen,
    )


def baue_modell(anzahl_klassen: int, config: Config) -> keras.Model:
    """MobileNetV2 als eingefrorene Basis mit neuem Klassifikationskopf bauen.

    Trainiert wird nur der neue Kopf (Dense 128, Dropout, Softmax). Die Basis
    behält ihre ImageNet-Gewichte.

    Args:
        anzahl_klassen: Anzahl Ausgänge der Softmax-Schicht.
        config: Bildgrösse und Lernrate.

    Returns:
        Das kompilierte Modell, bereit für model.fit.
    """
    form = (*config.image_size, 3)
    basis = keras.applications.MobileNetV2(
        input_shape=form, include_top=False, weights="imagenet"
    )
    basis.trainable = False  # zuerst nur den neuen Kopf trainieren
    # Zufällige Veränderungen wirken nur beim Training, bei der Vorhersage
    # lassen diese Schichten das Bild unverändert.
    augmentation = (
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.1),  # bis ±36 Grad
            layers.RandomContrast(0.2),
        ]
        if config.augmentation
        else []
    )
    model = models.Sequential(
        [
            keras.Input(shape=form),
            *augmentation,
            layers.Rescaling(
                1.0 / 127.5, offset=-1
            ),  # auf [-1, 1] wie MobileNetV2 erwartet
            basis,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(anzahl_klassen, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def trainiere(config: Config) -> None:
    """Modell trainieren, speichern und den Bericht schreiben.

    Das Modell landet unter model_path, die Klassennamen daneben als
    .klassen.json und die Grafiken in report_dir.

    Args:
        config: Pfade und Hyperparameter.
    """
    train, val, klassen = lade_datensaetze(config)
    print(f"Erkannte Klassen: {klassen}")
    model = baue_modell(len(klassen), config)
    model.summary()

    gewichte = None
    if config.klassengewichte:
        gewichte = berechne_klassengewichte(config.train_dir, klassen)
        print(
            "Klassengewichte:", {klassen[i]: round(g, 2) for i, g in gewichte.items()}
        )
    callbacks = []
    if config.early_stopping:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=config.early_stopping,
                restore_best_weights=True,  # gespeichert wird die beste Epoche
                verbose=1,
            )
        )

    history = model.fit(
        train,
        validation_data=val,
        epochs=config.epochs,
        class_weight=gewichte,
        callbacks=callbacks,
    )

    config.model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(config.model_path)
    speichere_klassen(config.model_path, klassen)
    print(f"Modell gespeichert: {config.model_path}")

    y_wahr = np.concatenate([labels.numpy() for _, labels in val])
    y_vorhersage = model.predict(val).argmax(axis=1)
    bericht.erstelle(
        history.history,
        y_wahr,
        y_vorhersage,
        klassen,
        config.report_dir,
        config.show_plots,
    )


def sage_vorher(bild: Path, config: Config) -> tuple[str, float]:
    """Ein Bild klassifizieren und alle Klassenwahrscheinlichkeiten ausgeben.

    Args:
        bild: Pfad zum Bild. Es wird auf image_size skaliert.
        config: Modellpfad, Trainingsordner und Bildgrösse.

    Returns:
        Name der wahrscheinlichsten Klasse und ihre Wahrscheinlichkeit
        (0 bis 1).
    """
    # compile=False: Für die Vorhersage braucht es Loss und Optimizer nicht, und das
    # Modell lässt sich so auch mit einer anderen Keras-Version laden.
    model = keras.models.load_model(config.model_path, compile=False)
    klassen = lade_klassen(config.model_path, config.train_dir)
    image = keras.utils.load_img(bild, target_size=config.image_size)
    eingabe = np.expand_dims(keras.utils.img_to_array(image), axis=0)
    wahrscheinlichkeiten = model.predict(eingabe, verbose=0)[0]
    klasse, konfidenz = interpretiere(wahrscheinlichkeiten, klassen)
    print(f"{bild.name}: {klasse} (Sicherheit {konfidenz:.1%})")
    for name, wert in zip(klassen, wahrscheinlichkeiten):
        print(f"  {name:<15} {wert:.3f}")
    return klasse, konfidenz


def main() -> None:
    """Kommandozeile auswerten und train oder predict starten."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", type=Path, default=Config.model_path)
    sub = parser.add_subparsers(dest="befehl", required=True)
    train = sub.add_parser("train", help="Modell trainieren und Berichte schreiben")
    train.add_argument("--epochs", type=int, default=Config.epochs)
    train.add_argument(
        "--zeige", action="store_true", help="Grafiken zusätzlich anzeigen"
    )
    train.add_argument(
        "--berichte", type=Path, default=Config.report_dir, help="Ordner für Grafiken"
    )
    train.add_argument(
        "--augmentation", action="store_true", help="Bilder zufällig verändern"
    )
    train.add_argument(
        "--klassengewichte", action="store_true", help="seltene Klassen stärker zählen"
    )
    train.add_argument(
        "--early-stopping",
        type=int,
        default=0,
        metavar="N",
        help="stoppen, wenn val_loss N Epochen nicht besser wird",
    )
    predict = sub.add_parser("predict", help="Ein Bild klassifizieren")
    predict.add_argument("bild", type=Path)
    args = parser.parse_args()

    if args.befehl == "train":
        trainiere(
            Config(
                model_path=args.model,
                report_dir=args.berichte,
                epochs=args.epochs,
                show_plots=args.zeige,
                augmentation=args.augmentation,
                klassengewichte=args.klassengewichte,
                early_stopping=args.early_stopping,
            )
        )
    else:
        sage_vorher(args.bild, Config(model_path=args.model))


if __name__ == "__main__":
    main()
