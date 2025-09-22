import duckdb
import pandas as pd
import polars as pl
from pathlib import Path
import json
from datetime import datetime
import sys
from typing import List, Dict, Optional
import ast

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import *
from config.logging_config import setup_logging

class JobMarketDatabase:
    """DuckDB database manager for job market data with analytics"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(DATABASE_PATH)
        self.conn = None
        self.logger = setup_logging()
        self.connect()
        self.setup_tables()
        
    def connect(self):
        """Connect to DuckDB database"""
        try:
            self.conn = duckdb.connect(self.db_path)
            self.logger.info("Connected to DuckDB: {Path(self.db_path).name}")
        except Exception as e:
            self.logger.error("Database connection failed: {e}")
            raise
    
    def setup_tables(self):
        """Create database tables with optimized schema"""
        
        # Jobs table with comprehensive schema
        jobs_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {PROCESSED_JOBS_TABLE} (
            job_id VARCHAR PRIMARY KEY,
            title VARCHAR NOT NULL,
            company VARCHAR NOT NULL,
            location VARCHAR,
            experience_min INTEGER,
            experience_max INTEGER,
            salary VARCHAR,
            description TEXT,
            job_url VARCHAR,
            company_rating DOUBLE,
            posted_date VARCHAR,
            search_keyword VARCHAR,
            page_number INTEGER,
            job_category VARCHAR,
            seniority_level VARCHAR,
            remote_friendly BOOLEAN,
            total_skills_count INTEGER,
            sentiment_score DOUBLE,
            scraped_at TIMESTAMP,
            nlp_processed_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        # Skills table for normalized skill storage
        skills_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {SKILLS_TABLE} (
            skill_id INTEGER PRIMARY KEY,
            job_id VARCHAR,
            skill_name VARCHAR NOT NULL,
            skill_category VARCHAR,
            extraction_method VARCHAR DEFAULT 'combined',
            confidence_score DOUBLE DEFAULT 1.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (job_id) REFERENCES {PROCESSED_JOBS_TABLE}(job_id)
        );
        """
        
        # Companies table for company analytics
        companies_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {COMPANIES_TABLE} (
            company_id INTEGER PRIMARY KEY,
            company_name VARCHAR UNIQUE NOT NULL,
            total_jobs INTEGER DEFAULT 0,
            avg_rating DOUBLE,
            locations_json VARCHAR,
            job_categories_json VARCHAR,
            top_skills_json VARCHAR,
            avg_skills_per_job DOUBLE,
            remote_jobs_count INTEGER DEFAULT 0,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        # Execute table creation
        try:
            self.conn.execute(jobs_table_sql)
            self.conn.execute(skills_table_sql)
            self.conn.execute(companies_table_sql)
            self.logger.info("Database tables created/verified")
        except Exception as e:
            self.logger.error(f"Table creation failed: {e}")
            raise
    
    def parse_json_column(self, value):
        """Parse JSON-like columns safely"""
        if pd.isna(value) or value == '' or str(value).lower() == 'nan':
            return {}
        
        if isinstance(value, dict):
            return value
        
        if isinstance(value, str):
            try:
                # Try JSON parsing first
                parsed = json.loads(value)
                return parsed if isinstance(parsed, dict) else {}
            except:
                try:
                    # Try ast.literal_eval for Python dict strings
                    parsed = ast.literal_eval(value)
                    return parsed if isinstance(parsed, dict) else {}
                except:
                    return {}
        
        return {}
    
    def parse_list_column(self, value):
        """Parse list-like columns safely"""
        if pd.isna(value) or value == '' or str(value).lower() == 'nan':
            return []
        
        if isinstance(value, list):
            return value
        
        if isinstance(value, str):
            try:
                # Try JSON parsing first
                parsed = json.loads(value)
                return parsed if isinstance(parsed, list) else []
            except:
                try:
                    # Try ast.literal_eval for Python list strings
                    parsed = ast.literal_eval(value)
                    return parsed if isinstance(parsed, list) else []
                except:
                    # Split by comma as fallback
                    return [item.strip() for item in value.split(',') if item.strip()]
        
        return []
    
    def insert_jobs_bulk(self, df: pd.DataFrame) -> int:
        """Insert jobs data in bulk"""
        try:
            self.logger.info("Preparing job data for database insertion...")
            
            # Prepare data for insertion
            jobs_data = df.copy()
            
            # Handle missing values and data types
            jobs_data = jobs_data.fillna('')
            
            # Parse job requirements if available
            sentiment_scores = []
            if 'job_requirements' in jobs_data.columns:
                for req in jobs_data['job_requirements']:
                    req_dict = self.parse_json_column(req)
                    sentiment_scores.append(req_dict.get('sentiment_score', 0.0))
                jobs_data['sentiment_score'] = sentiment_scores
            else:
                jobs_data['sentiment_score'] = 0.0
            
            # Convert boolean columns
            if 'remote_friendly' in jobs_data.columns:
                jobs_data['remote_friendly'] = jobs_data['remote_friendly'].astype(bool)
            else:
                jobs_data['remote_friendly'] = False
            
            # Select and prepare columns for insertion
            insert_columns = {
                'job_id': 'job_id',
                'title': 'title', 
                'company': 'company',
                'location': 'location',
                'experience_min': 'experience_min',
                'experience_max': 'experience_max',
                'salary': 'salary',
                'description': 'description',
                'job_url': 'job_url',
                'company_rating': 'company_rating',
                'posted_date': 'posted_date',
                'search_keyword': 'search_keyword',
                'page_number': 'page_number',
                'job_category': 'job_category',
                'seniority_level': 'seniority_level', 
                'remote_friendly': 'remote_friendly',
                'total_skills_count': 'total_skills_count',
                'sentiment_score': 'sentiment_score',
                'scraped_at': 'scraped_at',
                'nlp_processed_at': 'nlp_processed_at'
            }
            
            # Create insertion DataFrame with only available columns
            insert_df = pd.DataFrame()
            for db_col, df_col in insert_columns.items():
                if df_col in jobs_data.columns:
                    insert_df[db_col] = jobs_data[df_col]
                else:
                    # Set default values for missing columns
                    if db_col in ['experience_min', 'experience_max', 'page_number', 'total_skills_count']:
                        insert_df[db_col] = 0
                    elif db_col in ['company_rating', 'sentiment_score']:
                        insert_df[db_col] = 0.0
                    elif db_col == 'remote_friendly':
                        insert_df[db_col] = False
                    else:
                        insert_df[db_col] = ''
            
            # Clean up data types
            numeric_columns = ['experience_min', 'experience_max', 'page_number', 'total_skills_count']
            for col in numeric_columns:
                insert_df[col] = pd.to_numeric(insert_df[col], errors='coerce').fillna(0).astype(int)
            
            float_columns = ['company_rating', 'sentiment_score']
            for col in float_columns:
                insert_df[col] = pd.to_numeric(insert_df[col], errors='coerce').fillna(0.0)
            
            # Clear existing data and insert new data
            self.conn.execute(f"DELETE FROM {PROCESSED_JOBS_TABLE}")
            
            # Register DataFrame and insert
            self.conn.register('jobs_insert_df', insert_df)
            
            insert_sql = f"""
            INSERT INTO {PROCESSED_JOBS_TABLE} (
                job_id, title, company, location, experience_min, experience_max,
                salary, description, job_url, company_rating, posted_date,
                search_keyword, page_number, job_category, seniority_level,
                remote_friendly, total_skills_count, sentiment_score,
                scraped_at, nlp_processed_at
            )
            SELECT 
                job_id, title, company, location, experience_min, experience_max,
                salary, description, job_url, company_rating, posted_date,
                search_keyword, page_number, job_category, seniority_level,
                remote_friendly, total_skills_count, sentiment_score,
                scraped_at, nlp_processed_at
            FROM jobs_insert_df
            """
            
            self.conn.execute(insert_sql)
            inserted_count = len(insert_df)
            
            self.logger.info(f"Inserted {inserted_count} jobs into database")
            return inserted_count
            
        except Exception as e:
            self.logger.error(f"Bulk job insertion failed: {e}")
            raise
    
    def insert_skills_bulk(self, df: pd.DataFrame) -> int:
        """Insert skills data in bulk from NLP processed data"""
        try:
            self.logger.info("Preparing skills data for database insertion...")
            
            skills_records = []
            skill_id_counter = 0
            
            for idx, row in df.iterrows():
                job_id = row.get('job_id', f'job_{idx}')
                
                # Get all skills
                skills_all = self.parse_list_column(row.get('skills_all', []))
                
                # Get categorized skills for category mapping
                skills_categorized = self.parse_json_column(row.get('skills_categorized', {}))
                
                # Get NLP extracted skills to mark extraction method
                nlp_skills = set(self.parse_list_column(row.get('skills_nlp_extracted', [])))
                
                # Create skill records
                for skill in skills_all:
                    if not skill or str(skill).strip() == '':
                        continue
                        
                    skill_clean = str(skill).strip().lower()
                    
                    # Find category for skill
                    skill_category = 'other'
                    for category, cat_skills in skills_categorized.items():
                        if isinstance(cat_skills, list) and skill_clean in [s.lower() for s in cat_skills]:
                            skill_category = category
                            break
                    
                    # Determine extraction method
                    extraction_method = 'nlp_enhanced' if skill_clean in nlp_skills else 'scraped'
                    
                    skills_records.append({
                        'skill_id': skill_id_counter,
                        'job_id': job_id,
                        'skill_name': skill_clean,
                        'skill_category': skill_category,
                        'extraction_method': extraction_method,
                        'confidence_score': 0.9 if extraction_method == 'nlp_enhanced' else 0.8
                    })
                    skill_id_counter += 1
            
            if skills_records:
                skills_df = pd.DataFrame(skills_records)
                
                # Clear existing skills data
                self.conn.execute(f"DELETE FROM {SKILLS_TABLE}")
                
                # Register and insert
                self.conn.register('skills_insert_df', skills_df)
                
                insert_sql = f"""
                INSERT INTO {SKILLS_TABLE} (skill_id, job_id, skill_name, skill_category, extraction_method, confidence_score)
                SELECT skill_id, job_id, skill_name, skill_category, extraction_method, confidence_score 
                FROM skills_insert_df
                """
                
                self.conn.execute(insert_sql)
                
                self.logger.info(f"Inserted {len(skills_records)} skills into database")
                return len(skills_records)
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Skills insertion failed: {e}")
            raise
    
    def update_company_profiles(self) -> int:
        """Update company profiles based on job data"""
        try:
            self.logger.info("Updating company profiles...")
            
            # Clear existing company data
            self.conn.execute(f"DELETE FROM {COMPANIES_TABLE}")
            
            # Get company statistics
            company_stats_sql = f"""
            WITH company_stats AS (
                SELECT 
                    company,
                    COUNT(*) as total_jobs,
                    AVG(company_rating) as avg_rating,
                    COUNT(DISTINCT location) as location_count,
                    COUNT(DISTINCT job_category) as category_count,
                    AVG(total_skills_count) as avg_skills_per_job,
                    SUM(CASE WHEN remote_friendly THEN 1 ELSE 0 END) as remote_jobs_count,
                    AVG(sentiment_score) as avg_sentiment
                FROM {PROCESSED_JOBS_TABLE}
                WHERE company IS NOT NULL AND company != ''
                GROUP BY company
            )
            SELECT * FROM company_stats
            ORDER BY total_jobs DESC
            """
            
            company_df = self.conn.execute(company_stats_sql).df()
            
            # Get additional data for each company
            enhanced_companies = []
            company_id_counter = 0
            
            for idx, row in company_df.iterrows():
                company = row['company']
                
                # Get locations for this company
                locations_sql = f"""
                SELECT DISTINCT location, COUNT(*) as job_count
                FROM {PROCESSED_JOBS_TABLE} 
                WHERE company = ? AND location IS NOT NULL AND location != ''
                GROUP BY location
                ORDER BY job_count DESC
                LIMIT 10
                """
                locations_df = self.conn.execute(locations_sql, [company]).df()
                locations_data = locations_df.to_dict('records') if not locations_df.empty else []
                
                # Get job categories for this company
                categories_sql = f"""
                SELECT DISTINCT job_category, COUNT(*) as job_count
                FROM {PROCESSED_JOBS_TABLE} 
                WHERE company = ? AND job_category IS NOT NULL AND job_category != ''
                GROUP BY job_category
                ORDER BY job_count DESC
                """
                categories_df = self.conn.execute(categories_sql, [company]).df()
                categories_data = categories_df.to_dict('records') if not categories_df.empty else []
                
                # Get top skills for this company
                skills_sql = f"""
                SELECT s.skill_name, COUNT(*) as skill_count
                FROM {SKILLS_TABLE} s
                JOIN {PROCESSED_JOBS_TABLE} j ON s.job_id = j.job_id
                WHERE j.company = ?
                GROUP BY s.skill_name
                ORDER BY skill_count DESC
                LIMIT 15
                """
                skills_df = self.conn.execute(skills_sql, [company]).df()
                top_skills = skills_df.to_dict('records') if not skills_df.empty else []
                
                enhanced_companies.append({
                    'company_id': company_id_counter,
                    'company_name': company,
                    'total_jobs': int(row['total_jobs']),
                    'avg_rating': float(row['avg_rating']) if pd.notna(row['avg_rating']) else None,
                    'locations_json': json.dumps(locations_data),
                    'job_categories_json': json.dumps(categories_data),
                    'top_skills_json': json.dumps(top_skills),
                    'avg_skills_per_job': float(row['avg_skills_per_job']) if pd.notna(row['avg_skills_per_job']) else 0.0,
                    'remote_jobs_count': int(row['remote_jobs_count'])
                })
                company_id_counter += 1
            
            if enhanced_companies:
                companies_df = pd.DataFrame(enhanced_companies)
                
                # Register and insert
                self.conn.register('companies_insert_df', companies_df)
                
                insert_sql = f"""
                INSERT INTO {COMPANIES_TABLE} (
                    company_id, company_name, total_jobs, avg_rating, locations_json, 
                    job_categories_json, top_skills_json, avg_skills_per_job, remote_jobs_count
                )
                SELECT 
                    company_id, company_name, total_jobs, avg_rating, locations_json,
                    job_categories_json, top_skills_json, avg_skills_per_job, remote_jobs_count
                FROM companies_insert_df
                """
                
                self.conn.execute(insert_sql)
                
                self.logger.info(f"Updated {len(enhanced_companies)} company profiles")
                return len(enhanced_companies)
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Company profile update failed: {e}")
            raise
    
    def get_skill_demand_analysis(self) -> pd.DataFrame:
        """Get comprehensive skill demand analysis"""
        try:
            analysis_sql = f"""
            WITH skill_stats AS (
                SELECT 
                    s.skill_name,
                    s.skill_category,
                    COUNT(*) as job_count,
                    COUNT(DISTINCT s.job_id) as unique_jobs,
                    COUNT(DISTINCT j.company) as company_count,
                    COUNT(DISTINCT j.location) as location_count,
                    AVG(j.company_rating) as avg_company_rating,
                    AVG(j.total_skills_count) as avg_skills_in_job,
                    AVG(j.sentiment_score) as avg_job_sentiment,
                    SUM(CASE WHEN j.remote_friendly THEN 1 ELSE 0 END) as remote_jobs,
                    MAX(s.confidence_score) as max_confidence
                FROM {SKILLS_TABLE} s
                JOIN {PROCESSED_JOBS_TABLE} j ON s.job_id = j.job_id
                GROUP BY s.skill_name, s.skill_category
                HAVING job_count >= 2
            ),
            total_jobs AS (
                SELECT COUNT(*) as total FROM {PROCESSED_JOBS_TABLE}
            )
            SELECT 
                skill_name,
                skill_category,
                job_count,
                unique_jobs,
                company_count,
                location_count,
                ROUND(avg_company_rating, 2) as avg_company_rating,
                ROUND(avg_skills_in_job, 1) as avg_skills_in_job,
                ROUND(avg_job_sentiment, 2) as avg_job_sentiment,
                remote_jobs,
                ROUND(job_count * 100.0 / total.total, 2) as demand_percentage,
                ROUND(remote_jobs * 100.0 / job_count, 1) as remote_percentage,
                max_confidence
            FROM skill_stats, total_jobs total
            ORDER BY job_count DESC, demand_percentage DESC
            LIMIT 100
            """
            
            result_df = self.conn.execute(analysis_sql).df()
            self.logger.info(f"Generated skill demand analysis for {len(result_df)} skills")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Skill analysis failed: {e}")
            return pd.DataFrame()
    
    def get_location_analysis(self) -> pd.DataFrame:
        """Get job market analysis by location"""
        try:
            location_sql = f"""
            SELECT 
                location,
                COUNT(*) as job_count,
                COUNT(DISTINCT company) as company_count,
                COUNT(DISTINCT job_category) as category_count,
                AVG(company_rating) as avg_company_rating,
                AVG(total_skills_count) as avg_skills_per_job,
                SUM(CASE WHEN remote_friendly THEN 1 ELSE 0 END) as remote_jobs,
                ROUND(AVG(sentiment_score), 2) as avg_sentiment,
                COUNT(DISTINCT search_keyword) as job_type_diversity
            FROM {PROCESSED_JOBS_TABLE}
            WHERE location IS NOT NULL AND location != '' AND location != 'Unknown'
            GROUP BY location
            HAVING job_count >= 3
            ORDER BY job_count DESC
            LIMIT 50
            """
            
            result_df = self.conn.execute(location_sql).df()
            
            # Add remote job percentage
            if not result_df.empty:
                result_df['remote_job_percentage'] = (result_df['remote_jobs'] / result_df['job_count'] * 100).round(1)
            
            self.logger.info(f"Generated location analysis for {len(result_df)} locations")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Location analysis failed: {e}")
            return pd.DataFrame()
    
    def get_company_analysis(self) -> pd.DataFrame:
        """Get company analysis from database"""
        try:
            company_sql = f"""
            SELECT 
                company_name,
                total_jobs,
                avg_rating,
                avg_skills_per_job,
                remote_jobs_count,
                ROUND(remote_jobs_count * 100.0 / total_jobs, 1) as remote_percentage,
                locations_json,
                job_categories_json,
                top_skills_json
            FROM {COMPANIES_TABLE}
            ORDER BY total_jobs DESC
            LIMIT 100
            """
            
            result_df = self.conn.execute(company_sql).df()
            self.logger.info(f"Generated company analysis for {len(result_df)} companies")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Company analysis failed: {e}")
            return pd.DataFrame()
    
    def export_analytics_data(self) -> Dict[str, str]:
        """Export key analytics data to CSV files"""
        export_files = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Export skill demand analysis
            skill_df = self.get_skill_demand_analysis()
            if not skill_df.empty:
                skill_file = PROCESSED_DATA_DIR / f"skill_demand_analysis_{timestamp}.csv"
                skill_df.to_csv(skill_file, index=False)
                export_files['skills'] = str(skill_file)
                
            # Export location analysis
            location_df = self.get_location_analysis()
            if not location_df.empty:
                location_file = PROCESSED_DATA_DIR / f"location_analysis_{timestamp}.csv"
                location_df.to_csv(location_file, index=False)
                export_files['locations'] = str(location_file)
                
            # Export company analysis
            company_df = self.get_company_analysis()
            if not company_df.empty:
                company_file = PROCESSED_DATA_DIR / f"company_analysis_{timestamp}.csv"
                company_df.to_csv(company_file, index=False)
                export_files['companies'] = str(company_file)
            
            self.logger.info(f"Exported {len(export_files)} analytics files")
            
        except Exception as e:
            self.logger.error(f"Analytics export failed: {e}")
            
        return export_files
    
    def get_database_stats(self) -> Dict:
        """Get comprehensive database statistics"""
        try:
            stats = {}
            
            # Jobs table stats
            jobs_count = self.conn.execute(f"SELECT COUNT(*) FROM {PROCESSED_JOBS_TABLE}").fetchone()[0]
            stats['total_jobs'] = jobs_count
            
            # Skills table stats  
            skills_count = self.conn.execute(f"SELECT COUNT(*) FROM {SKILLS_TABLE}").fetchone()[0]
            unique_skills = self.conn.execute(f"SELECT COUNT(DISTINCT skill_name) FROM {SKILLS_TABLE}").fetchone()[0]
            stats['total_skills'] = skills_count
            stats['unique_skills'] = unique_skills
            
            # Companies table stats
            companies_count = self.conn.execute(f"SELECT COUNT(*) FROM {COMPANIES_TABLE}").fetchone()[0]
            stats['total_companies'] = companies_count
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Database stats failed: {e}")
            return {}
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.logger.info("Database connection closed")

def main():
    """Main database integration pipeline"""
    print("Starting DuckDB Database Integration Pipeline")
    print("=" * 60)
    
    try:
        # Find the latest NLP processed file
        csv_files = list(PROCESSED_DATA_DIR.glob("jobs_with_nlp_*.csv"))
        if not csv_files:
            print("No NLP processed files found. Run skill extraction first.")
            print("Run: python src/processing/skill_extractor.py")
            return
        
        latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
        print(f"Loading NLP data: {latest_file.name}")
        
        df = pd.read_csv(latest_file)
        print(f"Loaded {len(df)} processed job records")
        print(f"Available columns: {len(df.columns)}")
        
        # Initialize database
        print(f"\nInitializing database connection...")
        db = JobMarketDatabase()
        
        # New: Clear skills table first to avoid foreign key constraint error
        print(f"\nClearing skills data...")
        db.conn.execute(f"DELETE FROM {SKILLS_TABLE}")
        
        # Now, insert jobs data
        print(f"\nInserting job data...")
        jobs_inserted = db.insert_jobs_bulk(df)
        
        # Then, insert skills data
        print(f"\nInserting skills data...")
        skills_inserted = db.insert_skills_bulk(df)
        
        # Update company profiles
        print(f"\nUpdating company profiles...")
        companies_updated = db.update_company_profiles()
        
        # Export analytics
        print(f"\nGenerating analytics...")
        export_files = db.export_analytics_data()
        
        # Get database statistics
        db_stats = db.get_database_stats()
        
        # Print comprehensive results
        print("\n" + "="*60)
        print("DATABASE INTEGRATION RESULTS")
        print("="*60)
        print(f"Jobs inserted: {jobs_inserted}")
        print(f"Skills inserted: {skills_inserted}")
        print(f"Companies updated: {companies_updated}")
        print(f"Analytics files generated: {len(export_files)}")
        
        print(f"\nDATABASE STATISTICS:")
        print(f"  Total jobs: {db_stats.get('total_jobs', 0)}")
        print(f"  Total skills: {db_stats.get('total_skills', 0)}")
        print(f"  Unique skills: {db_stats.get('unique_skills', 0)}")
        print(f"  Companies: {db_stats.get('total_companies', 0)}")
        
        print(f"\nANALYTICS FILES GENERATED:")
        for data_type, file_path in export_files.items():
            file_name = Path(file_path).name
            print(f"  {data_type.title()}: {file_name}")
        
        print(f"\nDatabase integration completed successfully!")
        print(f"Database location: {DATABASE_PATH}")
        print(f"Analytics files: {PROCESSED_DATA_DIR}")
        
        # Quick preview of skill analysis
        if export_files.get('skills'):
            print(f"\nTOP 10 SKILLS PREVIEW:")
            skills_df = pd.read_csv(export_files['skills'])
            top_skills = skills_df.head(10)
            for idx, row in top_skills.iterrows():
                print(f"  {idx+1:2d}. {row['skill_name']}: {row['job_count']} jobs ({row['demand_percentage']}%)")
        
        db.close()
        
    except Exception as e:
        print(f"Database integration failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()