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
import json
from pathlib import Path

import numpy as np

from pcb_inspection import paths


@dataclass(frozen=True)
class Config:
    train_dir: Path = paths.DATENSATZ / 'train'
    val_dir: Path = paths.DATENSATZ / 'val'
    model_path: Path = paths.MODELLE / 'pcb_quality_model_tf.keras'
    report_dir: Path = paths.BERICHTE
    image_size: tuple[int, int] = (224, 224)
    batch_size: int = 30
    epochs: int = 40
    learning_rate: float = 0.001
    show_plots: bool = False


# --- Klassennamen ----------------------------------------------------------

def klassen_datei(model_path: Path) -> Path:
    return model_path.with_suffix('.klassen.json')


def klassen_aus_ordner(train_dir: Path) -> list[str]:
    """Gleiche Reihenfolge wie image_dataset_from_directory: alphabetisch sortiert."""
    if not train_dir.is_dir():
        raise FileNotFoundError(f'Trainingsordner nicht gefunden: {train_dir}')
    return sorted(p.name for p in train_dir.iterdir() if p.is_dir())


def speichere_klassen(model_path: Path, klassen: list[str]) -> None:
    klassen_datei(model_path).write_text(json.dumps(klassen, indent=2), encoding='utf-8')


def lade_klassen(model_path: Path, train_dir: Path) -> list[str]:
    """Klassen aus der JSON-Datei neben dem Modell, sonst aus den Trainingsordnern.

    Der Rückgriff auf die Ordner ist nur für ältere Modelle gedacht, die noch
    ohne JSON gespeichert wurden.
    """
    datei = klassen_datei(model_path)
    if datei.is_file():
        return json.loads(datei.read_text(encoding='utf-8'))
    return klassen_aus_ordner(train_dir)


def interpretiere(wahrscheinlichkeiten, klassen: list[str]) -> tuple[str, float]:
    """Wahrscheinlichkeiten eines Bildes in (Klasse, Konfidenz) übersetzen."""
    werte = np.asarray(wahrscheinlichkeiten, dtype=float).reshape(-1)
    if len(werte) != len(klassen):
        raise ValueError(f'Modell liefert {len(werte)} Werte, aber es gibt '
                         f'{len(klassen)} Klassen: {klassen}')
    index = int(werte.argmax())
    return klassen[index], float(werte[index])


# --- Training --------------------------------------------------------------

def lade_datensaetze(config: Config):
    import tensorflow as tf

    train = tf.keras.utils.image_dataset_from_directory(
        config.train_dir, image_size=config.image_size,
        batch_size=config.batch_size, shuffle=True)
    val = tf.keras.utils.image_dataset_from_directory(
        config.val_dir, image_size=config.image_size,
        batch_size=config.batch_size, shuffle=False)
    if train.class_names != val.class_names:
        raise ValueError(f'train und val haben unterschiedliche Klassen: '
                         f'{train.class_names} / {val.class_names}')
    klassen = train.class_names
    autotune = tf.data.AUTOTUNE
    return (train.cache().prefetch(autotune), val.cache().prefetch(autotune), klassen)


def baue_modell(anzahl_klassen: int, config: Config):
    import tensorflow as tf
    from tensorflow.keras import layers, models

    form = (*config.image_size, 3)
    basis = tf.keras.applications.MobileNetV2(
        input_shape=form, include_top=False, weights='imagenet')
    basis.trainable = False  # zuerst nur den neuen Kopf trainieren
    model = models.Sequential([
        tf.keras.Input(shape=form),
        layers.Rescaling(1. / 127.5, offset=-1),  # auf [-1, 1] wie MobileNetV2 erwartet
        basis,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(anzahl_klassen, activation='softmax'),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'])
    return model


def trainiere(config: Config) -> None:
    train, val, klassen = lade_datensaetze(config)
    print(f'Erkannte Klassen: {klassen}')
    model = baue_modell(len(klassen), config)
    model.summary()

    history = model.fit(train, validation_data=val, epochs=config.epochs)

    config.model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(config.model_path)
    speichere_klassen(config.model_path, klassen)
    print(f'Modell gespeichert: {config.model_path}')

    from pcb_inspection import bericht
    y_wahr = np.concatenate([labels.numpy() for _, labels in val])
    y_vorhersage = model.predict(val).argmax(axis=1)
    bericht.erstelle(history.history, y_wahr, y_vorhersage, klassen,
                     config.report_dir, config.show_plots)


# --- Vorhersage ------------------------------------------------------------

def sage_vorher(bild: Path, config: Config) -> tuple[str, float]:
    import tensorflow as tf

    model = tf.keras.models.load_model(config.model_path)
    klassen = lade_klassen(config.model_path, config.train_dir)
    image = tf.keras.utils.load_img(bild, target_size=config.image_size)
    eingabe = np.expand_dims(tf.keras.utils.img_to_array(image), axis=0)
    wahrscheinlichkeiten = model.predict(eingabe, verbose=0)[0]
    klasse, konfidenz = interpretiere(wahrscheinlichkeiten, klassen)
    print(f'{bild.name}: {klasse} (Sicherheit {konfidenz:.1%})')
    for name, wert in zip(klassen, wahrscheinlichkeiten):
        print(f'  {name:<15} {wert:.3f}')
    return klasse, konfidenz


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', type=Path, default=Config.model_path)
    sub = parser.add_subparsers(dest='befehl', required=True)
    train = sub.add_parser('train', help='Modell trainieren und Berichte schreiben')
    train.add_argument('--epochs', type=int, default=Config.epochs)
    train.add_argument('--zeige', action='store_true', help='Grafiken zusätzlich anzeigen')
    predict = sub.add_parser('predict', help='Ein Bild klassifizieren')
    predict.add_argument('bild', type=Path)
    args = parser.parse_args()

    if args.befehl == 'train':
        trainiere(Config(model_path=args.model, epochs=args.epochs, show_plots=args.zeige))
    else:
        sage_vorher(args.bild, Config(model_path=args.model))


if __name__ == '__main__':
    main()
