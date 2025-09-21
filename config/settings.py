from pathlib import Path
import os
from datetime import datetime

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Naukri settings
NAUKRI_BASE_URL = "https://www.naukri.com"

# Job search configuration
JOB_KEYWORDS = [
    "data-scientist",
    "software-engineer", 
    "python-developer",
    "machine-learning-engineer",
    "data-analyst",
    "full-stack-developer",
    "devops-engineer",
    "product-manager"
]

# Scraping parameters
MAX_PAGES_PER_KEYWORD = 3  # Start with 3 pages per keyword
DELAY_RANGE = (2, 5)  # Random delay between 2-5 seconds
MAX_RETRIES = 3
TIMEOUT = 15

# ChromeDriver path (your specific path)
CHROMEDRIVER_PATH = "/Users/priyankamalavade/Desktop/chromedriver-mac-x64/chromedriver"

# Output settings
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_CSV = RAW_DATA_DIR / f"naukri_jobs_{TIMESTAMP}.csv"
OUTPUT_JSON = RAW_DATA_DIR / f"naukri_jobs_{TIMESTAMP}.json"

# Create directories
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

print(" Enhanced configuration loaded successfully!")
print(f" Will search for: {len(JOB_KEYWORDS)} job types")
print(f" Output files: {OUTPUT_CSV.name}")