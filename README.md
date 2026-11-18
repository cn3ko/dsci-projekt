# dsci
Semesterarbeit Platinen

Aus Fotos von Platinen wird zuerst die Platine freigestellt, dann werden die
IC-Gehäuse darauf gesucht. Die Ausschnitte sortieren wir von Hand in gut und
schlecht und trainieren damit einen Klassifikator.

    data/platinen/      Originalfotos
      │  python -m pcb_inspection.platinen
    data/ausschnitte/   freigestellte Platinen (*_freigestellt.png)
      │  python -m pcb_inspection.gehaeuse
    data/gehaeuse/      ein Ordner pro Foto mit den gefundenen IC-Gehäusen
      │  von Hand sortieren
    data/datensatz/     train/ und val/, je ein Unterordner pro Klasse
      │  python -m pcb_inspection.klassifikator train
    models/             trainiertes Modell, reports/ Grafiken

In jedem Ordner liegt eine README.md, die erklärt, was darin liegt und warum
der jeweilige Schritt nötig ist.


Einrichten
==========

Python 3.10 bis 3.12 verwenden, neuere Versionen werden von TensorFlow noch
nicht sauber unterstützt.

    python -m venv .venv
    .venv\Scripts\activate          (Windows)
    source .venv/bin/activate       (Mac/Linux)
    python -m pip install -e ".[alles]"

Wer nur einen Teil braucht, kann auch `.[gehaeuse]` oder `.[training]`
installieren.

Unter Windows die venv nicht mit der Anaconda-Python anlegen. Anaconda bringt
eigene, ältere C++-Laufzeit-DLLs mit, und TensorFlow bricht dann beim Import
mit "Failed to load the native TensorFlow runtime" ab. Mit der Python von
python.org klappt es:

    py -3.12 -m venv .venv


1. Platinen freistellen
=======================

    python -m pcb_inspection.platinen --quick-test
    python -m pcb_inspection.platinen --no-preview

Sucht pro Foto die grüne Aussenkontur und speichert die Platine mit
transparentem Hintergrund in voller Auflösung. Bohrungen und dunkle Bauteile
innerhalb der Kontur bleiben erhalten. Die Schwellwerte stehen in Config in
platinen.py.

Früher wurde zusätzlich eine identische *_crop.png geschrieben, das ist
weggefallen. Die alten Dateien liegen noch in data/ausschnitte.


2. IC-Gehäuse erkennen
======================

Die Erkennung verwendet das spezialisierte Modell
https://huggingface.co/Arshia82sbn/pcb-yolov8s-detection
(Revision 7eb710b, best.pt). Es läuft lokal. Beim ersten Start ist Internet
für den Download nötig. Eigene Gewichte lassen sich mit --model angeben.

    python -m pcb_inspection.gehaeuse --quick-test
    python -m pcb_inspection.gehaeuse --no-preview

Eingabe: data/ausschnitte/*_freigestellt.png. Transparente Bildbereiche werden
für das Modell weiss hinterlegt; exportierte Gehäuse behalten die Transparenz.
Ausgabe: data/gehaeuse/<Originaldateiname>/ mit PNG-Ausschnitten,
uebersicht.jpg und erkennung.json (Koordinaten und Konfidenz).
Wiederholte Läufe ersetzen Ergebnisse desselben Bildes in diesem Ordner.

Das Modell prüft Ganzbild und überlappende Kacheln und entfernt doppelte
Treffer. Nur die Modellklasse ic wird gespeichert. Die erwartete Anzahl 7
ist eine Warnschwelle, kein Auswahlzwang. Die Übersichten bitte kontrollieren:
Die Eignung eines fremdtrainierten Modells für diese Fotos muss geprüft werden.

Bei fehlenden Treffern zunächst die Übersicht ansehen, dann gegebenenfalls:

    python -m pcb_inspection.gehaeuse --quick-test --confidence 0.15

Ein niedrigerer Wert kann zusätzliche Fehlalarme erzeugen. Kachelgrösse,
Überlappung und Ausschnittrand stehen in Config in gehaeuse.py.


3. Klassifikator trainieren
===========================

    python -m pcb_inspection.klassifikator train
    python -m pcb_inspection.klassifikator predict data/beispiele/IMG_0100.png

MobileNetV2 mit ImageNet-Gewichten, nur der neue Kopf wird trainiert. Die
Klassen sind die Unterordner von data/datensatz/train (zurzeit Bad_IC,
Bad_Platine, Good_IC, Good_Platine). Beim Training werden die Klassennamen
neben dem Modell gespeichert, die Vorhersage liest sie wieder ein.
Trainingskurven und Konfusionsmatrix landen in reports/.

Achtung beim Datensatz: Ausschnitte desselben Fotos sollten nur in train oder
nur in val liegen, sonst sind die Validierungswerte zu gut. Prüfen mit:

    pytest -m daten


Tests
=====

    pytest

Die Tests brauchen weder Modell-Download noch die Bilder unter data/.
`pytest -m daten` prüft zusätzlich den echten Datensatz (Überschneidung von
train und val, doppelte Bilder).
