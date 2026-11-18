"""Ordner des Projekts an einer Stelle, damit die Skripte von überall starten können."""
from pathlib import Path

PROJEKT = Path(__file__).resolve().parents[2]

DATEN = PROJEKT / 'data'
PLATINEN = DATEN / 'platinen'        # Originalfotos
AUSSCHNITTE = DATEN / 'ausschnitte'  # freigestellte Platinen (Schritt 1)
GEHAEUSE = DATEN / 'gehaeuse'        # erkannte IC-Gehäuse (Schritt 2)
DATENSATZ = DATEN / 'datensatz'      # von Hand sortiert: train/ und val/

MODELLE = PROJEKT / 'models'
BERICHTE = PROJEKT / 'reports'
