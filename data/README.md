data
====

Hier liegen alle Bilder der Arbeit, vom Originalfoto bis zum fertigen
Datensatz. Jeder Unterordner ist das Ergebnis eines Schritts und die Eingabe
für den nächsten:

    platinen/      Originalfotos
    ausschnitte/   freigestellte Platinen          (Schritt 1)
    gehaeuse/      gefundene IC-Gehäuse pro Foto   (Schritt 2)
    datensatz/     von Hand sortiert, train/ und val/
    beispiele/     einzelne Bilder zum Ausprobieren der Vorhersage

Warum in Schritten: Ein Foto zeigt neben der Platine auch Tisch, Schatten und
Hintergrund. Würde man das ganze Foto klassifizieren, lernt das Modell
womöglich den Hintergrund statt der Platine. Deshalb schneiden wir zuerst die
Platine aus, dann die einzelnen Bauteile, und erst diese kleinen Bilder werden
bewertet. Weil jeder Zwischenstand als Datei liegen bleibt, kann man jeden
Schritt einzeln prüfen und wiederholen, ohne alles neu zu rechnen.

Die Ordnernamen stehen zentral in src/pcb_inspection/paths.py.
