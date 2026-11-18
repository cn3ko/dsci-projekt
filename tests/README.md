tests
=====

Automatische Tests, gestartet mit:

    pytest              Tests für den Code
    pytest -m daten     zusätzlich Prüfungen am echten Datensatz

Warum Tests: Die Schritte bauen aufeinander auf. Ein Fehler in Schritt 1
fällt sonst erst viel später als schlechtes Modell auf. Die Tests prüfen jede
Funktion für sich, mit kleinen künstlichen Bildern. Sie brauchen weder die
Fotos noch den Modell-Download und laufen in wenigen Sekunden.

conftest.py
    Gemeinsame Einstellung: Grafiken werden nur in Dateien geschrieben,
    es öffnen sich keine Fenster.

test_platinen.py
    Schritt 1 mit einer gezeichneten grünen Platine auf hellem Grund: wird
    sie gefunden, bleibt der schwarze Chip darauf erhalten, passen die
    Koordinaten trotz verkleinerter Erkennung zum Original, ist der
    Hintergrund transparent, gibt es ohne Platine eine klare Fehlermeldung.

test_gehaeuse.py
    Schritt 2 mit einem nachgebauten Modell statt YOLO: Kacheln decken das
    ganze Bild ab, doppelte Treffer werden entfernt, Ausgabedateien entstehen
    wie erwartet.

test_klassen.py
    Klassennamen werden gespeichert, in der richtigen Reihenfolge geladen und
    richtig zugeordnet. Passen Modell und Klassenliste nicht zusammen, gibt
    es eine klare Fehlermeldung statt einer falschen Vorhersage.

test_bericht.py
    Grafiken und Kennzahlen entstehen, auch wenn eine Klasse in val fehlt.

test_datensatz.py
    Keine Tests für den Code, sondern Prüfungen der Daten: Liegen Ausschnitte
    desselben Fotos in train und in val? Gibt es doppelte Bilder? Beides
    würde die Validierungswerte zu gut aussehen lassen. Läuft nur mit
    pytest -m daten, weil es die echten Bilder braucht.
