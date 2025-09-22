import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
from datetime import datetime

# Add project root to path
# This is crucial for the script to find the config files
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config.settings import *

def generate_data_quality_report():
    """
    Generate a comprehensive data quality report from the latest NLP processed data.
    This function now returns the key dataframes and dictionaries.
    """
    
    print("GENERATING DATA QUALITY REPORT")
    print("=" * 60)
    
    # Load the latest NLP processed data
    csv_files = list(PROCESSED_DATA_DIR.glob("jobs_with_nlp_*.csv"))
    if not csv_files:
        print("No processed data files found")
        return {}, pd.DataFrame() # Return empty structures on failure
    
    latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
    df = pd.read_csv(latest_file)
    
    print(f"Analyzing: {latest_file.name}")
    print(f"Dataset shape: {df.shape}")
    
    # Basic statistics
    print(f"\nBASIC STATISTICS")
    print("-" * 30)
    print(f"Total records: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Data completeness analysis
    print(f"\nDATA COMPLETENESS ANALYSIS")
    print("-" * 30)
    missing_data = df.isnull().sum()
    missing_percentages = (missing_data / len(df)) * 100
    
    completeness_report = pd.DataFrame({
        'Column': missing_data.index,
        'Missing_Count': missing_data.values,
        'Missing_Percentage': missing_percentages.values,
        'Complete_Records': len(df) - missing_data.values
    }).sort_values('Missing_Percentage', ascending=False)
    
    print(completeness_report.head(10))
    
    # Critical fields analysis
    critical_fields = ['title', 'company', 'location', 'skills_all']
    print(f"\nCRITICAL FIELDS ANALYSIS")
    print("-" * 30)
    for field in critical_fields:
        if field in df.columns:
            complete_count = df[field].notna().sum()
            completeness = (complete_count / len(df)) * 100
            print(f"{field}: {complete_count}/{len(df)} ({completeness:.1f}% complete)")
    
    # Skills analysis
    skills_stats = pd.Series([0])
    if 'skills_all' in df.columns:
        print(f"\nSKILLS DATA QUALITY")
        print("-" * 30)
        
        skills_data = []
        for idx, row in df.iterrows():
            skills = row.get('skills_all', '[]')
            if isinstance(skills, str):
                try:
                    skills = eval(skills)
                except:
                    skills = []
            
            if isinstance(skills, list):
                skills_data.append(len(skills))
            else:
                skills_data.append(0)
        
        skills_stats = pd.Series(skills_data)
        print(f"Jobs with skills: {(skills_stats > 0).sum()}/{len(df)} ({((skills_stats > 0).sum()/len(df))*100:.1f}%)")
        print(f"Average skills per job: {skills_stats.mean():.1f}")
        print(f"Max skills in a job: {skills_stats.max()}")
        print(f"Jobs with 0 skills: {(skills_stats == 0).sum()}")
        print(f"Jobs with 5+ skills: {(skills_stats >= 5).sum()}")
    
    # Salary data quality
    salary_fields = ['salary_min', 'salary_max']
    if any(field in df.columns for field in salary_fields):
        print(f"\nSALARY DATA QUALITY")
        print("-" * 30)
        
        for field in salary_fields:
            if field in df.columns:
                valid_salaries = df[field].dropna()
                if len(valid_salaries) > 0:
                    print(f"{field}:")
                    print(f"  Valid entries: {len(valid_salaries)}/{len(df)} ({(len(valid_salaries)/len(df))*100:.1f}%)")
                    print(f"  Range: ₹{valid_salaries.min():.1f} - ₹{valid_salaries.max():.1f} Lakhs")
                    print(f"  Average: ₹{valid_salaries.mean():.1f} Lakhs")
    
    # Experience data quality
    exp_fields = ['experience_min', 'experience_max']
    if any(field in df.columns for field in exp_fields):
        print(f"\nEXPERIENCE DATA QUALITY")
        print("-" * 30)
        
        for field in exp_fields:
            if field in df.columns:
                valid_exp = df[field].dropna()
                if len(valid_exp) > 0:
                    print(f"{field}:")
                    print(f"  Valid entries: {len(valid_exp)}/{len(df)} ({(len(valid_exp)/len(df))*100:.1f}%)")
                    print(f"  Range: {valid_exp.min():.1f} - {valid_exp.max():.1f} years")
                    print(f"  Average: {valid_exp.mean():.1f} years")
    
    # Data consistency checks
    print(f"\nDATA CONSISTENCY CHECKS")
    print("-" * 30)
    
    duplicate_ids = 0
    if 'job_id' in df.columns:
        duplicate_ids = df['job_id'].duplicated().sum()
        print(f"Duplicate job IDs: {duplicate_ids}")
    
    invalid_ranges = 0
    if 'salary_min' in df.columns and 'salary_max' in df.columns:
        invalid_ranges = ((df['salary_max'] < df['salary_min']) & 
                         df['salary_min'].notna() & 
                         df['salary_max'].notna()).sum()
        print(f"Invalid salary ranges (max < min): {invalid_ranges}")
    
    invalid_exp_ranges = 0
    if 'experience_min' in df.columns and 'experience_max' in df.columns:
        invalid_exp_ranges = ((df['experience_max'] < df['experience_min']) & 
                             df['experience_min'].notna() & 
                             df['experience_max'].notna()).sum()
        print(f"Invalid experience ranges (max < min): {invalid_exp_ranges}")
    
    # Generate quality score
    print(f"\nOVERALL DATA QUALITY SCORE")
    print("-" * 30)
    
    quality_metrics = {
        'completeness_score': (df.notna().sum().sum() / (len(df) * len(df.columns))) * 100,
        'critical_fields_score': sum([df[field].notna().sum() for field in critical_fields if field in df.columns]) / (len(df) * len([f for f in critical_fields if f in df.columns])) * 100,
        'consistency_score': 100 - (duplicate_ids / len(df)) * 100 if 'job_id' in df.columns else 95,
        'jobs_with_skills_percentage': ((skills_stats > 0).sum()/len(df))*100,
        'invalid_salary_ranges': invalid_ranges,
        'invalid_experience_ranges': invalid_exp_ranges
    }
    
    overall_quality = np.mean(list(quality_metrics.values())[0:3]) # Calculate mean of key scores
    quality_metrics['overall_quality_score'] = overall_quality

    print(f"Completeness score: {quality_metrics['completeness_score']:.1f}%")
    print(f"Critical fields score: {quality_metrics['critical_fields_score']:.1f}%")
    print(f"Consistency score: {quality_metrics['consistency_score']:.1f}%")
    print(f"Overall quality score: {overall_quality:.1f}%")
    
    # Quality recommendations
    print(f"\nQUALITY IMPROVEMENT RECOMMENDATIONS")
    print("-" * 30)
    
    if quality_metrics['completeness_score'] < 80:
        print("• Consider improving data collection to reduce missing values")
    
    if quality_metrics['critical_fields_score'] < 90:
        print("• Focus on ensuring critical fields (title, company, skills) are complete")
    
    if duplicate_ids > 0:
        print("• Remove or consolidate duplicate job listings")
    
    high_missing_fields = completeness_report[completeness_report['Missing_Percentage'] > 50]['Column'].tolist()
    if high_missing_fields:
        print(f"• Consider removing or improving fields with >50% missing data: {', '.join(high_missing_fields)}")
    
    print(f"\nData quality report generation completed.")
    
    return quality_metrics, completeness_report

if __name__ == "__main__":
    # Define output directory and filenames
    output_dir = Path("/Users/priyankamalavade/Desktop/job-market-analyzer/data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_text_path = output_dir / f"data_quality_report_{timestamp}.txt"
    report_csv_path = output_dir / f"completeness_report_{timestamp}.csv"
    report_json_path = output_dir / f"quality_metrics_{timestamp}.json"
    
    # Redirect all print output to a text file
    original_stdout = sys.stdout
    with open(report_text_path, 'w') as f:
        sys.stdout = f
        quality_metrics, completeness_report = generate_data_quality_report()
        
    # Restore stdout
    sys.stdout = original_stdout
    
    # Save the structured data to files
    if not completeness_report.empty:
        completeness_report.to_csv(report_csv_path, index=False)
        
    if quality_metrics:
        # Convert NumPy types to native Python types for JSON serialization
        serializable_metrics = {
            key: value.item() if isinstance(value, np.generic) else value
            for key, value in quality_metrics.items()
        }
        
        with open(report_json_path, 'w') as json_file:
            json.dump(serializable_metrics, json_file, indent=4)
            
    # Final confirmation message to the user
    print("\n--- REPORT SAVED ---")
    print(f"Text report saved to: {report_text_path}")
    print(f"CSV report saved to: {report_csv_path}")
    print(f"JSON metrics saved to: {report_json_path}")
    print("--------------------")