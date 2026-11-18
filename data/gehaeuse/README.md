gehaeuse
========

Ergebnis von Schritt 2 (src/pcb_inspection/gehaeuse.py): ein Unterordner
pro freigestellter Platine, benannt nach der Eingabedatei. Darin:

    gehaeuse_000.png, ...   die gefundenen IC-Gehäuse als einzelne Bilder
    uebersicht.jpg          die Platine mit roten Rahmen um alle Treffer
    erkennung.json          Koordinaten und Konfidenz jedes Treffers

Warum: Bewertet werden soll der einzelne IC, nicht die ganze Platine. Die
Gehäuse von Hand auszuschneiden wäre bei rund 60 Fotos mit je etwa 7 ICs viel
Arbeit und schwer zu wiederholen. Deshalb sucht ein fertig trainiertes
YOLO-Modell für Platinen die ICs automatisch (siehe models/).

Die uebersicht.jpg ist zum Kontrollieren da. Das Modell wurde nicht auf
unseren Fotos trainiert, es kann also ICs übersehen oder etwas anderes für
einen IC halten. Die erkennung.json hält fest, mit welchem Modell und welcher
Schwelle das Ergebnis entstanden ist, damit man es später nachvollziehen
kann.

Aus diesen Ausschnitten wird von Hand der Datensatz in data/datensatz
zusammengestellt.
