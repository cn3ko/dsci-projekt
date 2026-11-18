pcb_inspection
==============

Der Code der Arbeit als Python-Paket. Als Paket, damit die Schritte mit
python -m pcb_inspection.<modul> von überall gestartet werden können und die
Tests den Code ohne Pfad-Tricks importieren.

platinen.py
    Schritt 1. Sucht auf jedem Originalfoto die grüne Platine und speichert
    sie freigestellt. Für die Suche wird das Bild verkleinert, weil das
    schneller ist und für die Umrisse reicht. Gespeichert wird aber in voller
    Auflösung, damit die kleinen ICs später scharf bleiben. Die Platine wird
    über ihre Farbe gefunden (HSV, weil sich Farbe dort unabhängig von der
    Helligkeit eingrenzen lässt). Kleine Lücken werden geschlossen (Closing),
    und alles innerhalb der äusseren Kontur gehört zur Platine. Fehler bei
    einem Bild brechen den Lauf nicht ab.

gehaeuse.py
    Schritt 2. Sucht die IC-Gehäuse mit dem YOLO-Modell aus models/ und
    schneidet sie aus. Das Modell arbeitet mit 640 Pixeln, unsere Platinen
    sind viel grösser. Auf dem verkleinerten Gesamtbild würden kleine ICs
    übersehen, deshalb wird das Bild zusätzlich in überlappende Kacheln
    zerlegt. Ein IC, der in mehreren Kacheln gefunden wird, wird danach nur
    einmal behalten (Non-Maximum Suppression). Der transparente Hintergrund
    wird für das Modell weiss gefüllt, die gespeicherten Ausschnitte behalten
    ihn.

klassifikator.py
    Schritt 3. Trainiert den Klassifikator und macht Vorhersagen. Wir haben
    nur gut 200 Bilder, das reicht nicht, um ein Netz von Grund auf zu
    trainieren. Deshalb Transfer Learning: MobileNetV2 wurde schon auf
    ImageNet trainiert und kann allgemeine Bildmerkmale erkennen. Diesen Teil
    frieren wir ein und trainieren nur einen kleinen neuen Kopf für unsere
    Klassen. MobileNetV2 ist klein genug, um auch ohne Grafikkarte zu laufen.

klassen.py
    Speichert die Klassennamen neben dem Modell und ordnet die Ausgabe des
    Modells wieder einem Namen zu. Eigene Datei ohne TensorFlow, damit diese
    Logik schnell getestet werden kann. Hier lag früher ein Fehler: Die
    Vorhersage hatte nur zwei Klassennamen fest eingetragen, das Modell hat
    aber vier Ausgänge.

bericht.py
    Erstellt nach dem Training Trainingskurven, Konfusionsmatrix und
    Kennzahlen (siehe reports/). Getrennt vom Training, damit die Auswertung
    ohne langes Training getestet werden kann.

paths.py
    Alle Ordner an einer Stelle. Die Pfade hängen vom Ort dieser Datei ab
    und nicht vom Ordner, in dem man das Skript startet.

__init__.py
    Macht den Ordner zum Paket, sonst nichts.
