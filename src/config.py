from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_DATA_PATH = RAW_DATA_DIR / "hillstrom_email_data.csv"

OUTPUTS_DIR = ROOT_DIR / "outputs"
TABLES_DIR = OUTPUTS_DIR / "tables"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
MODEL_RESULTS_DIR = OUTPUTS_DIR / "model_results"
FIGURES_DIR = OUTPUTS_DIR / "figures"

RANDOM_STATE = 42
TEST_SIZE = 0.25

CONTROL_NAME = "No E-Mail"
MENS_EMAIL = "Mens E-Mail"
WOMENS_EMAIL = "Womens E-Mail"

SEGMENT_COL = "segment"
TREATMENT_COL = "treatment"

PRIMARY_OUTCOME = "visit"
SECONDARY_OUTCOMES = ["conversion", "spend"]

PRE_TREATMENT_FEATURES = [
    "recency",
    "history",
    "history_segment",
    "mens",
    "womens",
    "newbie",
    "zip_code",
    "channel",
]

OUTCOME_COLS = [
    "visit",
    "conversion",
    "spend",
]

LEAKAGE_COLS = [
    "visit",
    "conversion",
    "spend",
    "segment",
    "treatment",
    "treatment_binary",
]
