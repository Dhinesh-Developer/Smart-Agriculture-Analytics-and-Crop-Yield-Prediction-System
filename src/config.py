"""Central configuration for the Smart Agriculture Analytics project."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "agri_warehouse.db"
RAW_CSV = DATA_DIR / "raw_agri_data.csv"

PROJECT_TITLE = "Smart Agriculture Analytics and Crop Yield Prediction System"
SUBJECT = "Data Warehousing and Data Mining - Capstone Project (Team No. 5, A Section)"

TEAM = [
    ("611223104043", "Giri N"),
    ("611223104032", "Dharshini S"),
    ("611223104060", "Harish R"),
    ("611223104047", "Gokula Krishnan V"),
]

NUMERIC_FEATURES = [
    "rainfall_mm", "temperature_c", "humidity_pct", "nitrogen", "phosphorus",
    "potassium", "soil_ph", "fertilizer_kg_ha", "irrigation_pct", "pesticide_kg_ha",
]
CATEGORICAL_FEATURES = ["crop_name", "season", "soil_type", "zone"]
TARGET_REG = "yield_t_ha"
TARGET_CLS = "yield_class"
CLASS_ORDER = ["Low", "Medium", "High"]

REQUIRED_COLUMNS = [
    "record_id", "year", "season", "state", "district", "zone", "crop_name", "crop_type",
    "soil_type", "area_ha", "rainfall_mm", "temperature_c", "humidity_pct", "nitrogen",
    "phosphorus", "potassium", "soil_ph", "fertilizer_kg_ha", "irrigation_pct",
    "pesticide_kg_ha", "yield_t_ha",
]

LABELS = {
    "rainfall_mm": "Rainfall (mm)", "temperature_c": "Temperature (°C)",
    "humidity_pct": "Humidity (%)", "nitrogen": "Nitrogen (kg/ha)",
    "phosphorus": "Phosphorus (kg/ha)", "potassium": "Potassium (kg/ha)",
    "soil_ph": "Soil pH", "fertilizer_kg_ha": "Fertilizer (kg/ha)",
    "irrigation_pct": "Irrigation Coverage (%)", "pesticide_kg_ha": "Pesticide (kg/ha)",
    "area_ha": "Area (ha)", "yield_t_ha": "Yield (t/ha)", "production_t": "Production (t)",
    "crop_name": "Crop", "season": "Season", "soil_type": "Soil Type", "zone": "Zone",
    "district": "District", "year": "Year",
}
