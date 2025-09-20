from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time
import sys
from pathlib import Path

# Add project root to path so we can import config
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import NAUKRI_BASE_URL

class NaukriScraper:
    """Basic Naukri.com scraper for testing connection and job extraction"""
    
    def __init__(self, headless=True):
        self.base_url = NAUKRI_BASE_URL
        self.headless = headless
        self.driver = None
        # Your ChromeDriver path
        self.chromedriver_path = "/Users/priyankamalavade/Desktop/chromedriver-mac-x64/chromedriver"
        print(" NaukriScraper initialized with custom ChromeDriver path")
    
    def setup_driver(self):
        """Setup Chrome WebDriver with your specific path"""
        try:
            chrome_options = Options()
            
            if self.headless:
                chrome_options.add_argument("--headless")
            
            # Essential options for Mac
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            
            # Anti-detection settings
            chrome_options.add_argument("--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            
            # Use your specific ChromeDriver path
            service = Service(self.chromedriver_path)
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            
            # Remove webdriver property
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            print(" Chrome WebDriver setup successful with custom path")
            return True
            
        except Exception as e:
            print(f" WebDriver setup failed: {e}")
            print(f" Make sure ChromeDriver is executable: chmod +x {self.chromedriver_path}")
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
    
    def test_job_search(self, search_term="analyst"):
        """Test job search functionality with actual Naukri structure"""
        try:
            search_url = f"{self.base_url}/{search_term}-jobs"
            print(f"🔍 Testing job search: {search_url}")
            
            self.driver.get(search_url)
            
            # Wait for job listings to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "cust-job-tuple"))
            )
            
            # Find job cards using the structure you provided
            job_cards = self.driver.find_elements(By.CLASS_NAME, "cust-job-tuple")
            print(f" Found {len(job_cards)} job listings")
            
            # Test extracting data from first job
            if job_cards:
                first_job = job_cards[0]
                
                # Extract job title
                title_element = first_job.find_element(By.CSS_SELECTOR, "h2 a.title")
                job_title = title_element.get_attribute("title") or title_element.text
                
                # Extract company name
                company_element = first_job.find_element(By.CSS_SELECTOR, "a.comp-name")
                company_name = company_element.get_attribute("title") or company_element.text
                
                # Extract location
                location_element = first_job.find_element(By.CSS_SELECTOR, ".loc-wrap .locWdth")
                location = location_element.get_attribute("title") or location_element.text
                
                # Extract experience
                exp_element = first_job.find_element(By.CSS_SELECTOR, ".exp-wrap .expwdth")
                experience = exp_element.get_attribute("title") or exp_element.text
                
                # Extract skills from tags
                skill_elements = first_job.find_elements(By.CSS_SELECTOR, ".tags-gt .tag-li")
                skills = [skill.text for skill in skill_elements]
                
                print("\n Sample Job Data Extracted:")
                print(f" Title: {job_title}")
                print(f" Company: {company_name}")
                print(f" Location: {location}")
                print(f"  Experience: {experience}")
                print(f" Skills: {', '.join(skills[:5])}...")  # Show first 5 skills
                
            # Check for pagination
            try:
                pagination = self.driver.find_element(By.CLASS_NAME, "styles_pagination__oIvXh")
                page_links = pagination.find_elements(By.CSS_SELECTOR, ".styles_pages__v1rAK a")
                print(f" Pagination found: {len(page_links)} pages available")
            except:
                print(" No pagination found")
                
            return True
            
        except Exception as e:
            print(f" Job search test failed: {e}")
            return False
    
    def close(self):
        """Close the browser"""
        if self.driver:
            self.driver.quit()
            print(" Browser closed")

def test_scraper():
    """Complete test function"""
    print(" Testing Naukri Scraper with Real Job Data...")
    
    # Create scraper (headless=False so you can see what's happening)
    scraper = NaukriScraper(headless=False)
    
    # Test basic connection
    if scraper.test_connection():
        print("\n Step 1: Basic connection successful")
        time.sleep(2)
        
        # Test job search
        if scraper.test_job_search():
            print("\n SUCCESS: Complete scraper test passed!")
            print(" Can connect to Naukri.com")
            print(" Can find job listings")
            print(" Can extract job data")
            print(" Ready for Day 2 development!")
        else:
            print("\n  Basic connection works but job extraction needs refinement")
    else:
        print("\n FAILED: Basic connection failed")
    
    # Keep browser open for 5 seconds so you can see the results
    time.sleep(5)
    scraper.close()
    print("\n Test completed")

if __name__ == "__main__":
    test_scraper()