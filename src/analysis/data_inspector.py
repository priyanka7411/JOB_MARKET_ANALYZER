import pandas as pd
import json
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import RAW_DATA_DIR

def inspect_scraped_data():
    """Analyze the scraped job data"""
    
    # Find the latest CSV file
    csv_files = list(RAW_DATA_DIR.glob("naukri_jobs_*.csv"))
    if not csv_files:
        print(" No scraped data files found")
        return
    
    latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
    print(f" Analyzing: {latest_file.name}")
    
    try:
        df = pd.read_csv(latest_file)
        
        print(f"\n DATASET OVERVIEW")
        print("=" * 50)
        print(f"Total jobs: {len(df)}")
        print(f"Columns: {len(df.columns)}")
        print(f"Data types:\n{df.dtypes}")
        
        print(f"\n JOB KEYWORDS DISTRIBUTION")
        print("=" * 50)
        if 'search_keyword' in df.columns:
            keyword_counts = df['search_keyword'].value_counts()
            print(keyword_counts)
        
        print(f"\n TOP LOCATIONS")
        print("=" * 50)
        if 'location' in df.columns:
            location_counts = df['location'].value_counts().head(10)
            print(location_counts)
        
        print(f"\n TOP COMPANIES")
        print("=" * 50)
        if 'company' in df.columns:
            company_counts = df['company'].value_counts().head(10)
            print(company_counts)
        
        print(f"\n MOST COMMON SKILLS")
        print("=" * 50)
        if 'skills' in df.columns:
            # Parse skills from string representation of list
            all_skills = []
            for skills_str in df['skills'].dropna():
                try:
                    skills_list = eval(skills_str) if isinstance(skills_str, str) else skills_str
                    if isinstance(skills_list, list):
                        all_skills.extend(skills_list)
                except:
                    pass
            
            if all_skills:
                skills_df = pd.Series(all_skills).value_counts().head(20)
                print(skills_df)
        
        print(f"\n DATA QUALITY CHECK")
        print("=" * 50)
        missing_data = df.isnull().sum()
        print("Missing values per column:")
        print(missing_data[missing_data > 0])
        
        print(f"\n Analysis complete!")
        
    except Exception as e:
        print(f" Error analyzing data: {e}")

if __name__ == "__main__":
    inspect_scraped_data()