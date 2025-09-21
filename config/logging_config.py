import logging
import sys
from pathlib import Path

def setup_logging():
    """Setup comprehensive logging"""
    
    # Create logs directory
    logs_dir = Path(__file__).parent.parent / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(logs_dir / 'scraper.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

# Create logs directory
logs_dir = Path(__file__).parent.parent / "logs"
logs_dir.mkdir(exist_ok=True)