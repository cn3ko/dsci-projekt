models
======

Die Modelle der Arbeit.

pcb-yolov8s/best.pt
-------------------

Fremdes, fertig trainiertes YOLOv8-Modell zum Erkennen von Bauteilen auf
Platinen, von https://huggingface.co/Arshia82sbn/pcb-yolov8s-detection
(Revision 7eb710b). Es wird in Schritt 2 benutzt, um die IC-Gehäuse zu
finden. Fehlt die Datei, lädt gehaeuse.py sie beim ersten Start herunter.

Warum ein fremdes Modell: Um selbst einen Detektor zu trainieren, müssten wir
in jedem Foto jeden IC mit einem Rahmen markieren. Dieses Modell kennt ICs
schon, wir müssen nur prüfen, ob es auf unseren Fotos zuverlässig ist. Die
Revision ist festgelegt, damit immer genau dieselben Gewichte verwendet
werden und die Ergebnisse wiederholbar bleiben.

pcb_quality_model_tf.keras
--------------------------

Unser eigener Klassifikator aus Schritt 3 (klassifikator.py). Er bekommt ein
Bild und sagt eine der Klassen aus data/datensatz voraus. Beim nächsten
Training wird hier zusätzlich pcb_quality_model_tf.klassen.json gespeichert,
die Liste der Klassen in der Reihenfolge der Modellausgänge. Ohne diese
Datei könnte die Vorhersage die Klassen vertauschen, wenn sich die Ordner in
data/datensatz ändern.
