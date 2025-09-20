from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time
import sys
from pathlib import Path

# Add project root to path so we can import config
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import NAUKRI_BASE_URL

class NaukriScraper:
    """Basic Naukri.com scraper for testing connection"""
    
    def __init__(self, headless=True):
        self.base_url = NAUKRI_BASE_URL
        self.headless = headless
        self.driver = None
        print(" NaukriScraper initialized")
    
    def setup_driver(self):
        """Setup Chrome WebDriver"""
        try:
            chrome_options = Options()
            
            if self.headless:
                chrome_options.add_argument("--headless")
            
            # Basic options for Mac
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            
            # Install and setup ChromeDriver automatically
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            
            print("Chrome WebDriver setup successful")
            return True
            
        except Exception as e:
            print(f" WebDriver setup failed: {e}")
            return False
    
    def test_connection(self):
        """Test basic connection to Naukri.com"""
        try:
            if not self.driver:
                if not self.setup_driver():
                    return False
            
            print(" Connecting to Naukri.com...")
            self.driver.get(self.base_url)
            
            # Wait for page to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Get basic page info
            title = self.driver.title
            current_url = self.driver.current_url
            
            print(f" Connection successful!")
            print(f" Page Title: {title}")
            print(f" URL: {current_url}")
            
            return True
            
        except Exception as e:
            print(f" Connection failed: {e}")
            return False
    
    def close(self):
        """Close the browser"""
        if self.driver:
            self.driver.quit()
            print(" Browser closed")

def test_scraper():
    """Test function to verify scraper works"""
    print(" Testing Naukri Scraper...")
    
    # Create scraper (headless=False so you can see the browser)
    scraper = NaukriScraper(headless=False)
    
    # Test connection
    if scraper.test_connection():
        print("\n SUCCESS: Scraper is working perfectly!")
        time.sleep(3)  # Wait so you can see the browser
    else:
        print("\n FAILED: Scraper connection failed")
    
    # Close browser
    scraper.close()
    print("\n Test completed")

if __name__ == "__main__":
    test_scraper()