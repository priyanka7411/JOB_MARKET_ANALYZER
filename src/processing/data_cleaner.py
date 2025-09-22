import pandas as pd
import polars as pl
import re
from pathlib import Path
from typing import List, Dict, Optional
import sys
from datetime import datetime
import logging
import ast

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import *
from config.logging_config import setup_logging

class JobDataCleaner:
    """Comprehensive job data cleaning for your specific CSV structure"""
    
    def __init__(self):
        self.logger = setup_logging()
        self.stats = {
            'total_records': 0,
            'valid_records': 0,
            'cleaned_records': 0,
            'duplicate_removals': 0,
            'skill_parsing_success': 0,
            'skill_parsing_failures': 0
        }
        
    def load_raw_data(self) -> pd.DataFrame:
        """Load the most recent scraped data"""
        csv_files = list(RAW_DATA_DIR.glob("naukri_jobs_*.csv"))
        if not csv_files:
            raise FileNotFoundError("No scraped data files found")
        
        latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
        self.logger.info(f"Loading data from: {latest_file.name}")
        
        df = pd.read_csv(latest_file)
        self.stats['total_records'] = len(df)
        self.logger.info(f"Loaded {len(df)} raw job records")
        
        # Display sample data info
        self.logger.info(f" Columns: {list(df.columns)}")
        
        return df
    
    def clean_and_validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate job data"""
        self.logger.info(" Starting data cleaning and validation...")
        
        df_clean = df.copy()
        
        # 1. Handle missing values
        self.logger.info(" Handling missing values...")
        
        # Fill missing essential fields
        df_clean['title'] = df_clean['title'].fillna('Unknown Position')
        df_clean['company'] = df_clean['company'].fillna('Unknown Company')
        df_clean['location'] = df_clean['location'].fillna('Unknown Location')
        df_clean['skills'] = df_clean['skills'].fillna('[]')
        
        # 2. Clean text fields
        self.logger.info("  Cleaning text fields...")
        
        text_columns = ['title', 'company', 'location', 'description']
        for col in text_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str)
                df_clean[col] = df_clean[col].str.strip()
                df_clean[col] = df_clean[col].replace('nan', None)
                # Remove extra whitespaces
                df_clean[col] = df_clean[col].str.replace(r'\s+', ' ', regex=True)
        
        # 3. Parse and clean skills
        self.logger.info(" Parsing skills data...")
        df_clean['skills_parsed'] = df_clean['skills'].apply(self.parse_skills)
        
        # 4. Clean location format
        self.logger.info(" Standardizing locations...")
        df_clean['location_clean'] = df_clean['location'].apply(self.clean_location)
        
        # 5. Parse company rating
        self.logger.info(" Parsing company ratings...")
        df_clean['company_rating_clean'] = df_clean['company_rating'].apply(self.parse_rating)
        
        # 6. Parse experience
        self.logger.info("  Parsing experience data...")
        exp_data = df_clean['experience'].apply(self.parse_experience)
        df_clean['experience_min'] = exp_data.apply(lambda x: x['min'])
        df_clean['experience_max'] = exp_data.apply(lambda x: x['max'])
        
        # 7. Determine job category and seniority
        self.logger.info("🎯 Determining job categories...")
        df_clean['job_category'] = df_clean.apply(self.determine_job_category, axis=1)
        df_clean['seniority_level'] = df_clean.apply(self.determine_seniority, axis=1)
        
        # 8. Check for remote work
        df_clean['remote_friendly'] = df_clean['description'].apply(self.check_remote_friendly)
        
        self.stats['valid_records'] = len(df_clean)
        self.logger.info(f" Processed {len(df_clean)} job records")
        
        return df_clean
    
    def parse_skills(self, skills_str: str) -> List[str]:
        """Parse skills from string representation of list"""
        if pd.isna(skills_str) or skills_str == '' or str(skills_str).lower() == 'nan':
            self.stats['skill_parsing_failures'] += 1
            return []
        
        try:
            # Parse the string representation of list: "['python', 'pandas']"
            skills_list = ast.literal_eval(str(skills_str))
            if isinstance(skills_list, list):
                # Clean and standardize skills
                cleaned_skills = []
                for skill in skills_list:
                    if isinstance(skill, str) and skill.strip():
                        # Standardize skill format
                        skill_clean = skill.strip().lower()
                        # Remove special characters but keep important ones
                        skill_clean = re.sub(r'[^\w\s\-\+\#\.]', '', skill_clean)
                        if len(skill_clean) >= 2:  # Minimum skill length
                            cleaned_skills.append(skill_clean)
                
                self.stats['skill_parsing_success'] += 1
                return list(set(cleaned_skills))  # Remove duplicates
            
        except Exception as e:
            self.logger.debug(f"Failed to parse skills: {skills_str} -> {e}")
            
        # Fallback: split by comma
        try:
            if ',' in str(skills_str):
                skills = [s.strip().lower() for s in str(skills_str).split(',')]
                self.stats['skill_parsing_success'] += 1
                return [s for s in skills if s and len(s) >= 2]
        except:
            pass
        
        self.stats['skill_parsing_failures'] += 1
        return []
    
    def clean_location(self, location: str) -> str:
        """Standardize location format"""
        if pd.isna(location) or str(location).lower() == 'nan':
            return 'Unknown'
        
        location = str(location).strip()
        
        # Remove extra location details and standardize
        location = re.sub(r'\s*,\s*India$', '', location)  # Remove ", India"
        location = re.sub(r'\s*,\s*Karnataka$', '', location)  # Remove ", Karnataka" 
        location = re.sub(r'\s*,\s*Maharashtra$', '', location)  # Remove ", Maharashtra"
        
        # Standardize common city names
        location_mapping = {
            'bengaluru': 'Bengaluru',
            'bangalore': 'Bengaluru', 
            'bangalore rural': 'Bengaluru',
            'mumbai': 'Mumbai',
            'pune': 'Pune',
            'hyderabad': 'Hyderabad',
            'chennai': 'Chennai',
            'delhi': 'Delhi',
            'ncr': 'Delhi NCR',
            'gurgaon': 'Gurgaon',
            'noida': 'Noida'
        }
        
        location_lower = location.lower()
        for key, value in location_mapping.items():
            if key in location_lower:
                return value
        
        return location.title()
    
    def parse_rating(self, rating: str) -> Optional[float]:
        """Parse company rating"""
        if pd.isna(rating) or str(rating).lower() in ['nan', '']:
            return None
        
        try:
            rating_val = float(rating)
            return rating_val if 0 <= rating_val <= 5 else None
        except:
            return None
    
    def parse_experience(self, exp_str: str) -> Dict[str, Optional[int]]:
        """Parse experience from strings like '3-7 Yrs', '4-9 Yrs'"""
        if pd.isna(exp_str) or str(exp_str).lower() == 'nan':
            return {'min': None, 'max': None}
        
        exp_str = str(exp_str).lower().strip()
        
        # Pattern matching for different experience formats
        patterns = [
            r'(\d+)-(\d+)\s*yrs?',  # "3-7 Yrs"
            r'(\d+)\s*to\s*(\d+)\s*yrs?',  # "3 to 7 years"
            r'(\d+)\+\s*yrs?',  # "5+ Yrs" 
            r'(\d+)\s*yrs?',  # "3 Yrs"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, exp_str)
            if match:
                groups = match.groups()
                if len(groups) == 2:
                    return {'min': int(groups[0]), 'max': int(groups[1])}
                elif '+' in exp_str:
                    return {'min': int(groups[0]), 'max': None}  # 5+ years
                else:
                    exp_val = int(groups[0])
                    return {'min': exp_val, 'max': exp_val}
        
        return {'min': None, 'max': None}
    
    def determine_job_category(self, row) -> str:
        """Determine job category from title and skills"""
        title = str(row.get('title', '')).lower()
        skills = row.get('skills_parsed', [])
        
        # Define category keywords
        categories = {
            'data_science': ['data scientist', 'machine learning', 'ml engineer', 'ai engineer', 'data analyst'],
            'software_engineer': ['software engineer', 'developer', 'programmer', 'sde'],
            'devops': ['devops', 'site reliability', 'infrastructure', 'cloud engineer'],
            'product_manager': ['product manager', 'product owner', 'pm'],
            'full_stack': ['full stack', 'fullstack'],
            'frontend': ['frontend', 'front-end', 'ui developer'],
            'backend': ['backend', 'back-end', 'api developer'],
            'mobile': ['mobile developer', 'android', 'ios', 'flutter', 'react native']
        }
        
        for category, keywords in categories.items():
            if any(keyword in title for keyword in keywords):
                return category
        
        # Check skills for category hints
        if any(skill in ['python', 'machine learning', 'tensorflow', 'pytorch'] for skill in skills):
            return 'data_science'
        elif any(skill in ['react', 'angular', 'vue'] for skill in skills):
            return 'frontend'
        elif any(skill in ['java', 'spring', 'node.js'] for skill in skills):
            return 'backend'
        
        return 'software_engineer'  # Default category
    
    def determine_seniority(self, row) -> str:
        """Determine seniority level"""
        title = str(row.get('title', '')).lower()
        exp_min = row.get('experience_min')
        
        # Check title for seniority indicators
        if any(word in title for word in ['senior', 'sr.', 'lead', 'principal', 'architect', 'manager']):
            return 'senior'
        elif any(word in title for word in ['junior', 'jr.', 'intern', 'trainee', 'fresher', 'associate']):
            return 'junior'
        
        # Use experience if available
        if exp_min is not None:
            if exp_min <= 2:
                return 'junior'
            elif exp_min >= 5:
                return 'senior'
            else:
                return 'mid'
        
        return 'mid'  # Default
    
    def check_remote_friendly(self, description: str) -> bool:
        """Check if job is remote friendly"""
        if pd.isna(description):
            return False
        
        desc_lower = str(description).lower()
        remote_indicators = ['remote', 'work from home', 'wfh', 'distributed team', 'anywhere']
        return any(indicator in desc_lower for indicator in remote_indicators)
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate job postings"""
        self.logger.info(" Removing duplicate jobs...")
        
        initial_count = len(df)
        
        # Remove duplicates based on title + company + location
        df_unique = df.drop_duplicates(
            subset=['title', 'company', 'location_clean'], 
            keep='first'
        )
        
        self.stats['duplicate_removals'] = initial_count - len(df_unique)
        self.logger.info(f" Removed {self.stats['duplicate_removals']} duplicates")
        
        return df_unique
    
    def save_cleaned_data(self, df: pd.DataFrame) -> str:
        """Save cleaned data to CSV"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = PROCESSED_DATA_DIR / f"cleaned_jobs_{timestamp}.csv"
        
        # Select final columns for output
        output_columns = [
            'job_id', 'title', 'company', 'location_clean', 'experience_min', 'experience_max',
            'salary', 'description', 'skills_parsed', 'job_url', 'company_rating_clean',
            'posted_date', 'scraped_at', 'search_keyword', 'page_number',
            'job_category', 'seniority_level', 'remote_friendly'
        ]
        
        # Only include columns that exist
        available_columns = [col for col in output_columns if col in df.columns]
        df_output = df[available_columns].copy()
        
        # Rename columns for consistency
        column_mapping = {
            'location_clean': 'location',
            'skills_parsed': 'skills_extracted', 
            'company_rating_clean': 'company_rating'
        }
        
        df_output = df_output.rename(columns=column_mapping)
        
        # Save to CSV
        df_output.to_csv(output_file, index=False)
        
        self.stats['cleaned_records'] = len(df_output)
        self.logger.info(f" Cleaned data saved: {output_file}")
        
        return str(output_file)
    
    def print_cleaning_stats(self):
        """Print comprehensive data cleaning statistics"""
        print("\n" + "="*60)
        print(" DATA CLEANING STATISTICS")
        print("="*60)
        print(f" Total raw records: {self.stats['total_records']}")
        print(f" Valid records processed: {self.stats['valid_records']}")
        print(f"  Duplicates removed: {self.stats['duplicate_removals']}")
        print(f"Final cleaned records: {self.stats['cleaned_records']}")
        print(f" Skill parsing success: {self.stats['skill_parsing_success']}")
        print(f" Skill parsing failures: {self.stats['skill_parsing_failures']}")
        
        if self.stats['total_records'] > 0:
            success_rate = (self.stats['cleaned_records'] / self.stats['total_records']) * 100
            skill_success_rate = (self.stats['skill_parsing_success'] / (self.stats['skill_parsing_success'] + self.stats['skill_parsing_failures'])) * 100 if (self.stats['skill_parsing_success'] + self.stats['skill_parsing_failures']) > 0 else 0
            
            print(f" Overall success rate: {success_rate:.1f}%")
            print(f" Skill parsing success rate: {skill_success_rate:.1f}%")
        
        print("="*60 + "\n")

def main():
    """Main data cleaning pipeline"""
    print(" Starting Job Data Cleaning Pipeline for Your CSV Structure")
    
    cleaner = JobDataCleaner()
    
    try:
        # Load raw data
        df = cleaner.load_raw_data()
        
        # Clean and validate
        df_clean = cleaner.clean_and_validate(df)
        
        # Remove duplicates
        df_unique = cleaner.remove_duplicates(df_clean)
        
        # Save cleaned data
        output_file = cleaner.save_cleaned_data(df_unique)
        
        # Print statistics
        cleaner.print_cleaning_stats()
        
        print(f" Data cleaning completed successfully!")
        print(f" Cleaned data saved: {Path(output_file).name}")
        
        # Show sample of cleaned data
        print(f"\n Sample of cleaned data:")
        sample_df = pd.read_csv(output_file)
        print(f"Shape: {sample_df.shape}")
        print(f"Columns: {list(sample_df.columns)}")
        
        if len(sample_df) > 0:
            print(f"\nFirst record:")
            first_record = sample_df.iloc[0]
            for col in ['title', 'company', 'location', 'job_category', 'seniority_level']:
                if col in first_record:
                    print(f"  {col}: {first_record[col]}")
        
        return output_file
        
    except Exception as e:
        print(f" Data cleaning failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()