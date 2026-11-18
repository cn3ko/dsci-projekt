reports
=======

Grafiken, die nach dem Training des Klassifikators entstehen
(src/pcb_inspection/bericht.py). Sie sind für die Auswertung in der Arbeit
gedacht.

Trainingskurven.png
    Accuracy und Loss über die Epochen, für train und val. Wenn die
    train-Werte weiter besser werden, die val-Werte aber nicht, lernt das
    Modell die Trainingsbilder auswendig (Overfitting).

Normalized_Konfusionsmatrix.png
    Pro wahrer Klasse: welcher Anteil wurde als welche Klasse erkannt. Die
    Matrix ist pro Zeile normalisiert, weil die Klassen sehr unterschiedlich
    gross sind (zum Beispiel viel mehr Good_IC als Bad_IC). So sieht man auch
    bei kleinen Klassen, wie gut sie erkannt werden.

Die Zahlen dazu (Precision, Recall, F1 pro Klasse) werden beim Training in der
Konsole ausgegeben.

Die Grafiken sind nicht im Repository (siehe .gitignore), weil sie bei jedem
Training neu entstehen. Wer sie braucht, trainiert selbst oder legt sie hier
ab.
