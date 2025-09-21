from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import pandas as pd
import json
import time
import random
import logging
from datetime import datetime
from typing import List, Dict, Optional
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import *
from config.logging_config import setup_logging

class ProductionNaukriScraper:
    """Production-ready Naukri.com job scraper with comprehensive features"""
    
    def __init__(self, headless: bool = True):
        self.base_url = NAUKRI_BASE_URL
        self.headless = headless
        self.driver = None
        self.scraped_data = []
        self.stats = {
            'total_jobs': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'keywords_processed': 0,
            'pages_processed': 0
        }
        
        # Setup logging
        self.logger = setup_logging()
        self.logger.info("🤖 Production Naukri Scraper initialized")
        
    def setup_driver(self) -> bool:
        """Setup Chrome WebDriver with anti-detection measures"""
        try:
            chrome_options = Options()
            
            if self.headless:
                chrome_options.add_argument("--headless")
            
            # Anti-detection and stability options
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
            chrome_options.add_argument("--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
            
            # Experimental options
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            chrome_options.add_experimental_option("detach", True)
            
            # Use your ChromeDriver path
            service = Service(CHROMEDRIVER_PATH)
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            
            # Remove webdriver property
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            # Set timeouts
            self.driver.implicitly_wait(10)
            self.driver.set_page_load_timeout(TIMEOUT)
            
            self.logger.info(" Chrome WebDriver setup successful")
            return True
            
        except Exception as e:
            self.logger.error(f" WebDriver setup failed: {e}")
            return False
    
    def extract_job_data(self, job_element) -> Dict:
        """Extract comprehensive job data from job card element"""
        job_data = {
            'job_id': None,
            'title': None,
            'company': None,
            'location': None,
            'experience': None,
            'salary': None,
            'description': None,
            'skills': [],
            'job_url': None,
            'company_rating': None,
            'posted_date': None,
            'scraped_at': datetime.now().isoformat()
        }
        
        try:
            # Job Title and URL
            try:
                title_element = job_element.find_element(By.CSS_SELECTOR, "h2 a.title")
                job_data['title'] = title_element.get_attribute("title") or title_element.text
                job_data['job_url'] = title_element.get_attribute("href")
                
                # Extract job ID from URL
                if job_data['job_url']:
                    job_data['job_id'] = job_data['job_url'].split('-')[-1].split('?')[0]
            except NoSuchElementException:
                pass
            
            # Company Name
            try:
                company_element = job_element.find_element(By.CSS_SELECTOR, "a.comp-name")
                job_data['company'] = company_element.get_attribute("title") or company_element.text
            except NoSuchElementException:
                pass
            
            # Company Rating
            try:
                rating_element = job_element.find_element(By.CSS_SELECTOR, ".rating .main-2")
                job_data['company_rating'] = rating_element.text
            except NoSuchElementException:
                pass
            
            # Location
            try:
                location_element = job_element.find_element(By.CSS_SELECTOR, ".loc-wrap .locWdth")
                job_data['location'] = location_element.get_attribute("title") or location_element.text
            except NoSuchElementException:
                pass
            
            # Experience
            try:
                exp_element = job_element.find_element(By.CSS_SELECTOR, ".exp-wrap .expwdth")
                job_data['experience'] = exp_element.get_attribute("title") or exp_element.text
            except NoSuchElementException:
                pass
            
            # Job Description
            try:
                desc_element = job_element.find_element(By.CSS_SELECTOR, ".job-desc")
                job_data['description'] = desc_element.text
            except NoSuchElementException:
                pass
            
            # Skills
            try:
                skill_elements = job_element.find_elements(By.CSS_SELECTOR, ".tags-gt .tag-li")
                job_data['skills'] = [skill.text for skill in skill_elements if skill.text.strip()]
            except NoSuchElementException:
                pass
            
            # Posted Date
            try:
                date_element = job_element.find_element(By.CSS_SELECTOR, ".job-post-day")
                job_data['posted_date'] = date_element.text
            except NoSuchElementException:
                pass
            
            return job_data
            
        except Exception as e:
            self.logger.error(f"Error extracting job data: {e}")
            return job_data
    
    def scrape_jobs_for_keyword(self, keyword: str, max_pages: int = MAX_PAGES_PER_KEYWORD) -> List[Dict]:
        """Scrape jobs for a specific keyword with pagination"""
        self.logger.info(f" Starting scrape for keyword: {keyword}")
        keyword_jobs = []
        
        for page in range(1, max_pages + 1):
            try:
                # Construct URL for current page
                if page == 1:
                    url = f"{self.base_url}/{keyword}-jobs"
                else:
                    url = f"{self.base_url}/{keyword}-jobs-{page}"
                
                self.logger.info(f" Scraping page {page}: {url}")
                
                # Navigate to page
                self.driver.get(url)
                
                # Wait for job listings to load
                WebDriverWait(self.driver, TIMEOUT).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "cust-job-tuple"))
                )
                
                # Find all job cards
                job_cards = self.driver.find_elements(By.CLASS_NAME, "cust-job-tuple")
                self.logger.info(f" Found {len(job_cards)} jobs on page {page}")
                
                # Extract data from each job card
                page_jobs = []
                for i, job_card in enumerate(job_cards):
                    try:
                        job_data = self.extract_job_data(job_card)
                        job_data['search_keyword'] = keyword
                        job_data['page_number'] = page
                        
                        # Only add jobs with at least a title
                        if job_data['title']:
                            page_jobs.append(job_data)
                            self.stats['successful_extractions'] += 1
                        else:
                            self.stats['failed_extractions'] += 1
                            
                    except Exception as e:
                        self.logger.error(f"Error processing job {i+1}: {e}")
                        self.stats['failed_extractions'] += 1
                
                keyword_jobs.extend(page_jobs)
                self.stats['pages_processed'] += 1
                
                self.logger.info(f" Page {page} completed: {len(page_jobs)} jobs extracted")
                
                # Random delay between pages
                delay = random.uniform(*DELAY_RANGE)
                self.logger.info(f"  Waiting {delay:.1f} seconds before next page...")
                time.sleep(delay)
                
            except TimeoutException:
                self.logger.warning(f" Timeout on page {page} for keyword {keyword}")
                break
            except Exception as e:
                self.logger.error(f"Error on page {page} for keyword {keyword}: {e}")
                break
        
        self.stats['keywords_processed'] += 1
        self.stats['total_jobs'] += len(keyword_jobs)
        
        self.logger.info(f" Keyword '{keyword}' completed: {len(keyword_jobs)} total jobs")
        return keyword_jobs
    
    def scrape_all_keywords(self, keywords: List[str] = None) -> List[Dict]:
        """Scrape jobs for all specified keywords"""
        if keywords is None:
            keywords = JOB_KEYWORDS
            
        self.logger.info(f" Starting comprehensive scrape for {len(keywords)} keywords")
        
        # Setup driver
        if not self.setup_driver():
            self.logger.error(" Failed to setup driver")
            return []
        
        all_jobs = []
        
        try:
            for i, keyword in enumerate(keywords, 1):
                self.logger.info(f" Processing keyword {i}/{len(keywords)}: {keyword}")
                
                keyword_jobs = self.scrape_jobs_for_keyword(keyword)
                all_jobs.extend(keyword_jobs)
                
                # Longer delay between keywords
                if i < len(keywords):
                    delay = random.uniform(5, 10)
                    self.logger.info(f" Waiting {delay:.1f} seconds before next keyword...")
                    time.sleep(delay)
                
        except KeyboardInterrupt:
            self.logger.info(" Scraping interrupted by user")
        except Exception as e:
            self.logger.error(f" Unexpected error during scraping: {e}")
        finally:
            self.close()
        
        self.scraped_data = all_jobs
        self.print_stats()
        
        return all_jobs
    
    def save_data(self) -> None:
        """Save scraped data to CSV and JSON files"""
        if not self.scraped_data:
            self.logger.warning("  No data to save")
            return
        
        try:
            # Save to CSV
            df = pd.DataFrame(self.scraped_data)
            df.to_csv(OUTPUT_CSV, index=False)
            self.logger.info(f" Data saved to CSV: {OUTPUT_CSV}")
            
            # Save to JSON
            with open(OUTPUT_JSON, 'w') as f:
                json.dump(self.scraped_data, f, indent=2, default=str)
            self.logger.info(f"Data saved to JSON: {OUTPUT_JSON}")
            
            # Save summary statistics
            stats_file = RAW_DATA_DIR / f"scraping_stats_{TIMESTAMP}.json"
            with open(stats_file, 'w') as f:
                json.dump(self.stats, f, indent=2)
            self.logger.info(f" Statistics saved: {stats_file}")
            
        except Exception as e:
            self.logger.error(f" Error saving data: {e}")
    
    def print_stats(self) -> None:
        """Print comprehensive scraping statistics"""
        print("\n" + "="*60)
        print(" SCRAPING STATISTICS")
        print("="*60)
        print(f" Keywords processed: {self.stats['keywords_processed']}")
        print(f" Pages processed: {self.stats['pages_processed']}")
        print(f" Total jobs found: {self.stats['total_jobs']}")
        print(f" Successful extractions: {self.stats['successful_extractions']}")
        print(f" Failed extractions: {self.stats['failed_extractions']}")
        
        if self.stats['total_jobs'] > 0:
            success_rate = (self.stats['successful_extractions'] / self.stats['total_jobs']) * 100
            print(f" Success rate: {success_rate:.1f}%")
        
        print("="*60 + "\n")
    
    def close(self):
        """Safely close the WebDriver"""
        if self.driver:
            self.driver.quit()
            self.logger.info(" WebDriver closed successfully")

def main():
    """Main scraping function"""
    print(" Starting Production Naukri Job Scraper")
    print(f" Target: {len(JOB_KEYWORDS)} job types × {MAX_PAGES_PER_KEYWORD} pages each")
    print(f" Expected jobs: ~{len(JOB_KEYWORDS) * MAX_PAGES_PER_KEYWORD * 20} jobs")
    
    # Ask user for confirmation
    response = input("\n Start scraping? (y/n): ").strip().lower()
    if response != 'y':
        print(" Scraping cancelled")
        return
    
    # Initialize scraper
    scraper = ProductionNaukriScraper(headless=True)  # Set to False to watch
    
    # Start scraping
    jobs_data = scraper.scrape_all_keywords()
    
    if jobs_data:
        # Save data
        scraper.save_data()
        
        # Show sample data
        print("\n Sample Job Data:")
        print("-" * 50)
        if len(jobs_data) > 0:
            sample_job = jobs_data[0]
            for key, value in sample_job.items():
                if key != 'description':  # Skip long description
                    print(f"{key}: {value}")
        
        print(f"\n Scraping completed successfully!")
        print(f" Files saved in: {RAW_DATA_DIR}")
    else:
        print("\nNo data was scraped")

if __name__ == "__main__":
    main()