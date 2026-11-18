datensatz
=========

Der Datensatz für den Klassifikator, von Hand aus data/gehaeuse und
data/ausschnitte sortiert. Die Unterordner von train/ und val/ sind die
Klassen:

    Good_IC        IC in Ordnung
    Bad_IC         IC fehlerhaft
    Good_Platine   ganze Platine in Ordnung
    Bad_Platine    ganze Platine fehlerhaft

Warum von Hand: Für unsere Platinen gibt es keine fertigen Labels. Ob ein
Bauteil gut oder schlecht ist, muss ein Mensch entscheiden. Die Ordner sind
die einfachste Form, das festzuhalten, und Keras liest die Klassen direkt
daraus.

Warum train und val: Mit train lernt das Modell, mit val wird geprüft, wie gut
es auf Bildern ist, die es beim Lernen nicht gesehen hat. Dafür dürfen
Ausschnitte desselben Fotos nur auf einer Seite liegen. Sonst sieht das
Modell in val fast dieselben Bilder wie in train, und die Ergebnisse sehen
besser aus, als sie sind.

Prüfen lässt sich das mit:

    pytest -m daten

Zurzeit schlägt diese Prüfung fehl (einige Fotos in train und val, einige
doppelte Bilder). Siehe tests/test_datensatz.py.

In train/ und val/ selbst bitte keine weiteren Dateien ablegen, nur die
Klassenordner.
