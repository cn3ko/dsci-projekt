ausschnitte
===========

Ergebnis von Schritt 1 (src/pcb_inspection/platinen.py): pro Foto eine
Datei <name>_freigestellt.png. Das ist die Platine in voller Auflösung, der
Hintergrund ist transparent.

Warum freistellen: Die Gehäuseerkennung in Schritt 2 soll nur die Platine
sehen und nicht auf Dinge auf dem Tisch reagieren. Gesucht wird die grüne
Fläche der Platine. Alles innerhalb ihrer Aussenkontur bleibt erhalten, also
auch schwarze Chips und Bohrungen, die selbst nicht grün sind.

Die Dateien *_crop.png sind aus einer älteren Version des Skripts und mit den
freigestellten Bildern identisch. Sie werden nicht mehr erzeugt und nicht
mehr gebraucht, liegen aber noch hier, weil Teile von data/datensatz aus
ihnen entstanden sind.
