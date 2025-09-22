import spacy
import pandas as pd
from collections import Counter
import re
from typing import List, Dict, Set, Tuple
from pathlib import Path
import sys
import json
from transformers import pipeline
import logging
from datetime import datetime
import ast

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import *
from config.logging_config import setup_logging

class AdvancedSkillExtractor:
    """Advanced NLP-powered skill extraction and categorization for your job data"""
    
    def __init__(self):
        self.logger = setup_logging()
        self.nlp = None
        self.sentiment_analyzer = None
        self.known_skills = set()
        self.load_models()
        self.setup_skill_database()
        
    def load_models(self):
        """Load NLP models with error handling"""
        try:
            self.logger.info(" Loading spaCy model...")
            self.nlp = spacy.load("en_core_web_sm")
            
            self.logger.info("Loading sentiment analyzer...")
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis", 
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
            
            self.logger.info(" NLP models loaded successfully")
            
        except Exception as e:
            self.logger.error(f" Error loading models: {e}")
            self.logger.info(" Installing required models...")
            
            # Try to download spaCy model if missing
            import subprocess
            try:
                subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
                self.nlp = spacy.load("en_core_web_sm")
                self.logger.info(" spaCy model installed and loaded")
            except:
                self.logger.error(" Failed to install spaCy model. Using basic extraction only.")
                self.nlp = None
            
            # Sentiment analyzer with fallback
            try:
                self.sentiment_analyzer = pipeline("sentiment-analysis")
            except:
                self.logger.warning(" Sentiment analyzer not available. Skipping sentiment analysis.")
                self.sentiment_analyzer = None
    
    def setup_skill_database(self):
        """Create comprehensive skill database"""
        
        # Enhanced skill categories based on your actual data
        self.skill_categories = {
            'programming_languages': {
                'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust', 
                'scala', 'kotlin', 'swift', 'php', 'ruby', 'r', 'matlab', 'sql', 'c',
                'perl', 'shell scripting', 'bash'
            },
            'machine_learning_ai': {
                'machine learning', 'deep learning', 'neural networks', 'tensorflow', 
                'pytorch', 'keras', 'scikit-learn', 'pandas', 'numpy', 'artificial intelligence',
                'computer vision', 'natural language processing', 'nlp', 'opencv', 'transformers',
                'bert', 'gpt', 'llm', 'reinforcement learning', 'ensemble methods',
                'random forest', 'svm', 'linear regression', 'logistic regression', 'clustering',
                'classification', 'regression', 'predictive modeling', 'statistical modeling',
                'data mining', 'pattern recognition', 'feature engineering', 'model deployment',
                'mlops', 'model optimization', 'hyperparameter tuning', 'cross validation',
                'time series', 'anomaly detection', 'recommendation systems'
            },
            'data_science_analytics': {
                'data science', 'data analysis', 'data analytics', 'statistics', 'statistical analysis',
                'data visualization', 'business intelligence', 'predictive analytics', 
                'descriptive analytics', 'prescriptive analytics', 'data modeling', 'etl',
                'data warehousing', 'big data', 'data pipeline', 'data engineering',
                'apache spark', 'hadoop', 'hive', 'pig', 'kafka', 'airflow', 'luigi',
                'data governance', 'data quality', 'a/b testing', 'experimentation',
                'hypothesis testing', 'regression analysis', 'correlation analysis'
            },
            'cloud_platforms': {
                'aws', 'amazon web services', 'azure', 'microsoft azure', 'gcp', 
                'google cloud platform', 'google cloud', 'cloud computing', 'serverless',
                'lambda', 'ec2', 's3', 'rds', 'dynamodb', 'cloudformation', 'terraform',
                'ansible', 'chef', 'puppet', 'kubernetes', 'docker', 'containerization',
                'microservices', 'service mesh', 'istio', 'cloud architecture', 'cloud security',
                'cloud migration', 'multi-cloud', 'hybrid cloud', 'infrastructure as code',
                'aws sagemaker', 'azure ml', 'google ai platform', 'vertex ai'
            },
            'web_frameworks': {
                'react', 'angular', 'vue', 'vue.js', 'node.js', 'express', 'django', 
                'flask', 'spring', 'spring boot', 'laravel', 'symfony', 'ruby on rails',
                'asp.net', '.net', 'fastapi', 'tornado', 'bottle', 'streamlit', 'dash',
                'next.js', 'nuxt.js', 'svelte', 'ember.js', 'backbone.js', 'jquery',
                'bootstrap', 'material-ui', 'ant design', 'semantic ui'
            },
            'databases': {
                'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'cassandra',
                'oracle', 'sqlite', 'dynamodb', 'neo4j', 'graph database', 'nosql',
                'relational database', 'acid', 'sql server', 'mariadb', 'couchdb',
                'influxdb', 'time series database', 'data modeling', 'database design',
                'database optimization', 'indexing', 'sharding', 'replication'
            },
            'devops_tools': {
                'devops', 'ci/cd', 'jenkins', 'gitlab ci', 'github actions', 'travis ci',
                'circleci', 'docker', 'kubernetes', 'helm', 'prometheus', 'grafana',
                'elk stack', 'elasticsearch', 'logstash', 'kibana', 'monitoring', 
                'logging', 'infrastructure monitoring', 'application monitoring',
                'load balancing', 'reverse proxy', 'nginx', 'apache', 'linux',
                'bash scripting', 'automation', 'configuration management'
            },
            'soft_skills': {
                'communication', 'leadership', 'teamwork', 'problem solving',
                'project management', 'agile', 'scrum', 'kanban', 'waterfall',
                'stakeholder management', 'collaboration', 'mentoring', 'coaching',
                'presentation skills', 'analytical thinking', 'critical thinking',
                'creativity', 'innovation', 'adaptability', 'time management'
            }
        }
        
        # Create flat set of all known skills
        self.known_skills = set()
        for category, skills in self.skill_categories.items():
            self.known_skills.update(skills)
        
        # Add variations and abbreviations
        skill_variations = {
            'artificial intelligence': ['ai', 'a.i.'],
            'machine learning': ['ml', 'm.l.'],
            'deep learning': ['dl', 'd.l.'],
            'natural language processing': ['nlp', 'n.l.p.'],
            'computer vision': ['cv', 'c.v.'],
            'amazon web services': ['aws'],
            'google cloud platform': ['gcp'],
            'microsoft azure': ['azure'],
            'javascript': ['js'],
            'typescript': ['ts'],
            'python': ['py'],
            'react.js': ['react', 'reactjs'],
            'node.js': ['nodejs', 'node'],
            'vue.js': ['vue', 'vuejs'],
            'postgresql': ['postgres'],
            'mongodb': ['mongo'],
            'elasticsearch': ['elastic']
        }
        
        # Add variations to known skills
        for main_skill, variations in skill_variations.items():
            for variation in variations:
                self.known_skills.add(variation)
        
        self.logger.info(f" Skill database created with {len(self.known_skills)} known skills")
    
    def extract_skills_from_text(self, text: str) -> List[str]:
        """Extract skills using multiple NLP techniques"""
        if not text or pd.isna(text):
            return []
        
        text_lower = str(text).lower()
        extracted_skills = set()
        
        # Method 1: Direct pattern matching with word boundaries
        for skill in self.known_skills:
            # Create regex pattern with word boundaries
            pattern = r'\b' + re.escape(skill.lower()) + r'\b'
            if re.search(pattern, text_lower):
                extracted_skills.add(skill)
        
        # Method 2: spaCy NLP processing (if available)
        if self.nlp:
            try:
                doc = self.nlp(text[:1000])  # Limit text length for performance
                
                # Extract entities that might be technologies
                for ent in doc.ents:
                    if ent.label_ in ["ORG", "PRODUCT", "GPE"]:
                        ent_text = ent.text.lower().strip()
                        # Check if entity looks like a technology
                        if (len(ent_text) >= 2 and 
                            any(tech_word in ent_text for tech_word in 
                                ['tech', 'soft', 'data', 'cloud', 'web', 'dev', 'ai', 'ml'])):
                            extracted_skills.add(ent_text)
                
                # Extract noun phrases that might be skills
                for chunk in doc.noun_chunks:
                    chunk_text = chunk.text.lower().strip()
                    if (2 <= len(chunk_text) <= 25 and 
                        any(word in chunk_text for word in 
                            ['learning', 'analysis', 'development', 'engineering', 'science'])):
                        extracted_skills.add(chunk_text)
                        
            except Exception as e:
                self.logger.debug(f"spaCy processing error: {e}")
        
        # Method 3: Common abbreviations and acronyms
        abbreviations = {
            r'\bml\b': 'machine learning',
            r'\bai\b': 'artificial intelligence',
            r'\bnlp\b': 'natural language processing',
            r'\bcv\b': 'computer vision',
            r'\bdl\b': 'deep learning',
            r'\baws\b': 'amazon web services',
            r'\bgcp\b': 'google cloud platform',
            r'\bjs\b': 'javascript',
            r'\bts\b': 'typescript',
            r'\bapi\b': 'api development',
            r'\bui\b': 'user interface',
            r'\bux\b': 'user experience'
        }
        
        for pattern, skill in abbreviations.items():
            if re.search(pattern, text_lower):
                extracted_skills.add(skill)
        
        # Method 4: Technology stack patterns
        tech_patterns = [
            r'(react|angular|vue)\.?js',
            r'node\.?js',
            r'spring\s+boot',
            r'asp\.?net',
            r'\.net\s+core',
            r'ruby\s+on\s+rails',
            r'django\s+rest\s+framework',
            r'express\.?js'
        ]
        
        for pattern in tech_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                skill = match.group().replace('.', '').strip()
                extracted_skills.add(skill)
        
        # Clean and filter skills
        final_skills = []
        for skill in extracted_skills:
            skill_clean = skill.strip().lower()
            if (len(skill_clean) >= 2 and 
                len(skill_clean) <= 30 and
                not skill_clean.isdigit() and
                skill_clean not in ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for']):
                final_skills.append(skill_clean)
        
        return list(set(final_skills))  # Remove duplicates
    
    def categorize_skills(self, skills: List[str]) -> Dict[str, List[str]]:
        """Categorize extracted skills into predefined categories"""
        categorized = {category: [] for category in self.skill_categories.keys()}
        uncategorized = []
        
        for skill in skills:
            skill_lower = skill.lower()
            categorized_flag = False
            
            # Check each category
            for category, category_skills in self.skill_categories.items():
                for category_skill in category_skills:
                    if (skill_lower == category_skill.lower() or 
                        category_skill.lower() in skill_lower or
                        skill_lower in category_skill.lower()):
                        categorized[category].append(skill)
                        categorized_flag = True
                        break
                
                if categorized_flag:
                    break
            
            if not categorized_flag:
                uncategorized.append(skill)
        
        # Add uncategorized skills if any
        if uncategorized:
            categorized['other_technical'] = uncategorized
        
        # Remove empty categories
        return {k: v for k, v in categorized.items() if v}
    
    def extract_job_requirements(self, description: str) -> Dict:
        """Extract structured requirements from job description"""
        if not description or pd.isna(description):
            return {
                'required_skills': [],
                'preferred_skills': [],
                'education': [],
                'certifications': [],
                'experience_level': None,
                'job_type': 'onsite',
                'sentiment_score': 0.0
            }
        
        desc_str = str(description)
        desc_lower = desc_str.lower()
        requirements = {
            'required_skills': [],
            'preferred_skills': [],
            'education': [],
            'certifications': [],
            'experience_level': None,
            'job_type': 'onsite',
            'sentiment_score': 0.0
        }
        
        try:
            # Analyze sentiment if analyzer is available
            if self.sentiment_analyzer:
                # Truncate description for transformer model
                desc_truncated = desc_str[:512]
                sentiment_result = self.sentiment_analyzer(desc_truncated)
                if sentiment_result and len(sentiment_result) > 0:
                    score = sentiment_result[0]['score']
                    label = sentiment_result[0]['label']
                    requirements['sentiment_score'] = score if label == 'POSITIVE' else -score
                    
        except Exception as e:
            self.logger.debug(f"Sentiment analysis error: {e}")
        
        # Extract education requirements
        education_patterns = [
            r'\b(bachelor|master|phd|doctorate|mba|b\.tech|m\.tech|bca|mca|b\.e\.|m\.e\.)\b',
            r'\b(computer science|engineering|mathematics|statistics|data science)\b',
            r'\b(degree|graduation|post.graduation)\b'
        ]
        
        for pattern in education_patterns:
            matches = re.findall(pattern, desc_lower)
            requirements['education'].extend(matches)
        
        # Extract certifications
        cert_patterns = [
            r'\b(aws|azure|google cloud|gcp)\s+(certified|certification)\b',
            r'\bcertification\s+in\s+[\w\s]+',
            r'\b(pmp|cissp|cisa|cism|ceh|ciscp)\b',
            r'\b(oracle|microsoft|cisco|comptia)\s+certified\b'
        ]
        
        for pattern in cert_patterns:
            matches = re.findall(pattern, desc_lower)
            for match in matches:
                cert = match if isinstance(match, str) else ' '.join(match)
                requirements['certifications'].append(cert)
        
        # Determine job type
        remote_indicators = ['remote', 'work from home', 'wfh', 'distributed', 'anywhere']
        hybrid_indicators = ['hybrid', 'flexible', 'part remote']
        
        if any(indicator in desc_lower for indicator in remote_indicators):
            requirements['job_type'] = 'remote'
        elif any(indicator in desc_lower for indicator in hybrid_indicators):
            requirements['job_type'] = 'hybrid'
        
        # Extract experience level hints
        if any(word in desc_lower for word in ['senior', 'lead', 'principal', 'architect']):
            requirements['experience_level'] = 'senior'
        elif any(word in desc_lower for word in ['junior', 'entry', 'fresher', 'graduate']):
            requirements['experience_level'] = 'junior'
        else:
            requirements['experience_level'] = 'mid'
        
        return requirements
    
    def process_job_descriptions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process all job descriptions with NLP"""
        self.logger.info(" Starting skill extraction from job descriptions...")
        
        results = []
        total_jobs = len(df)
        
        for idx, row in df.iterrows():
            try:
                # Combine multiple text fields for analysis
                all_text = ""
                
                # Add description
                if 'description' in row and pd.notna(row['description']):
                    all_text += str(row['description']) + " "
                
                # Add title for additional context
                if 'title' in row and pd.notna(row['title']):
                    all_text += str(row['title']) + " "
                
                # Get existing skills from your scraper
                existing_skills = []
                skills_column = None
                for col in ['skills_extracted', 'skills', 'skills_parsed']:
                    if col in row and pd.notna(row[col]):
                        skills_column = col
                        skills_data = row[col]
                        break
                
                if skills_data is not None:
                    if isinstance(skills_data, str):
                        try:
                            # Parse string representation of list
                            if skills_data.startswith('['):
                                existing_skills = ast.literal_eval(skills_data)
                            else:
                                existing_skills = [s.strip() for s in skills_data.split(',')]
                        except:
                            existing_skills = []
                    elif isinstance(skills_data, list):
                        existing_skills = skills_data
                
                # Clean existing skills
                if isinstance(existing_skills, list):
                    existing_skills = [s.lower().strip() for s in existing_skills if s and str(s).strip()]
                else:
                    existing_skills = []
                
                # Extract new skills using NLP
                nlp_extracted_skills = self.extract_skills_from_text(all_text)
                
                # Combine and deduplicate skills
                all_skills = list(set(existing_skills + nlp_extracted_skills))
                
                # Categorize skills
                categorized_skills = self.categorize_skills(all_skills)
                
                # Extract job requirements
                requirements = self.extract_job_requirements(row.get('description', ''))
                
                # Create result record
                result = row.to_dict()
                result['skills_nlp_extracted'] = nlp_extracted_skills
                result['skills_all'] = all_skills
                result['skills_categorized'] = categorized_skills
                result['job_requirements'] = requirements
                result['total_skills_count'] = len(all_skills)
                result['nlp_processed_at'] = datetime.now().isoformat()
                
                results.append(result)
                
                # Progress update
                if (idx + 1) % 50 == 0 or (idx + 1) == total_jobs:
                    progress = ((idx + 1) / total_jobs) * 100
                    self.logger.info(f"Processed {idx + 1}/{total_jobs} job descriptions ({progress:.1f}%)")
                    
            except Exception as e:
                self.logger.error(f"Error processing job {idx}: {e}")
                # Add row with minimal processing to not lose data
                result = row.to_dict()
                result['skills_nlp_extracted'] = []
                result['skills_all'] = existing_skills if 'existing_skills' in locals() else []
                result['skills_categorized'] = {}
                result['job_requirements'] = {}
                result['total_skills_count'] = 0
                results.append(result)
        
        processed_df = pd.DataFrame(results)
        self.logger.info(f" Skill extraction completed for {len(processed_df)} jobs")
        
        return processed_df
    
    def generate_skill_insights(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive skill market insights"""
        self.logger.info(" Generating skill market insights...")
        
        insights = {
            'total_jobs_analyzed': len(df),
            'unique_skills': set(),
            'skill_frequency': {},
            'skills_by_category': {},
            'skills_by_location': {},
            'trending_skills': [],
            'rare_skills': [],
            'nlp_extraction_stats': {}
        }
        
        # Collect all skills
        all_skills_list = []
        nlp_skills_list = []
        
        for idx, row in df.iterrows():
            # All skills
            skills = row.get('skills_all', [])
            if isinstance(skills, list):
                all_skills_list.extend([s.lower() for s in skills if s])
                insights['unique_skills'].update([s.lower() for s in skills if s])
            
            # NLP extracted skills
            nlp_skills = row.get('skills_nlp_extracted', [])
            if isinstance(nlp_skills, list):
                nlp_skills_list.extend([s.lower() for s in nlp_skills if s])
        
        # Skill frequency analysis
        skill_counter = Counter(all_skills_list)
        insights['skill_frequency'] = dict(skill_counter.most_common(50))
        
        # NLP extraction statistics
        insights['nlp_extraction_stats'] = {
            'total_nlp_skills_extracted': len(nlp_skills_list),
            'unique_nlp_skills': len(set(nlp_skills_list)),
            'avg_nlp_skills_per_job': len(nlp_skills_list) / len(df) if len(df) > 0 else 0
        }
        
        # Trending vs rare skills
        total_jobs = len(df)
        insights['trending_skills'] = [
            skill for skill, count in skill_counter.items() 
            if count >= max(3, total_jobs * 0.05)  # Appears in 5%+ of jobs or at least 3 jobs
        ]
        
        insights['rare_skills'] = [
            skill for skill, count in skill_counter.items() 
            if count <= 2  # Appears in 2 or fewer jobs
        ]
        
        # Skills by category analysis
        category_counters = {}
        for category in self.skill_categories.keys():
            category_counters[category] = Counter()
        category_counters['other_technical'] = Counter()
        
        for idx, row in df.iterrows():
            categorized = row.get('skills_categorized', {})
            if isinstance(categorized, str):
                try:
                    categorized = ast.literal_eval(categorized)
                except:
                    categorized = {}
            
            if isinstance(categorized, dict):
                for category, skills in categorized.items():
                    if isinstance(skills, list) and category in category_counters:
                        category_counters[category].update([s.lower() for s in skills])
        
        insights['skills_by_category'] = {
            category: dict(counter.most_common(15))
            for category, counter in category_counters.items()
            if counter
        }
        
        # Skills by location
        location_skills = {}
        for idx, row in df.iterrows():
            location = row.get('location', 'Unknown')
            if location and location != 'Unknown':
                skills = row.get('skills_all', [])
                
                if isinstance(skills, list):
                    if location not in location_skills:
                        location_skills[location] = Counter()
                    location_skills[location].update([s.lower() for s in skills if s])
        
        insights['skills_by_location'] = {
            loc: dict(counter.most_common(10))
            for loc, counter in location_skills.items()
            if counter
        }
        
        self.logger.info(f" Generated insights for {len(insights['unique_skills'])} unique skills")
        return insights
    
    def save_nlp_results(self, df: pd.DataFrame, insights: Dict) -> Tuple[str, str]:
        """Save NLP processed data and insights"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save processed jobs with NLP data
        jobs_file = PROCESSED_DATA_DIR / f"jobs_with_nlp_{timestamp}.csv"
        
        # Convert complex columns to JSON strings for CSV storage
        df_save = df.copy()
        json_columns = ['skills_categorized', 'job_requirements']
        for col in json_columns:
            if col in df_save.columns:
                df_save[col] = df_save[col].apply(
                    lambda x: json.dumps(x) if isinstance(x, (dict, list)) else str(x)
                )
        
        df_save.to_csv(jobs_file, index=False)
        
        # Save insights as JSON
        insights_file = PROCESSED_DATA_DIR / f"skill_insights_{timestamp}.json"
        
        # Convert sets to lists for JSON serialization
        insights_serializable = {}
        for key, value in insights.items():
            if isinstance(value, set):
                insights_serializable[key] = list(value)
            else:
                insights_serializable[key] = value
        
        with open(insights_file, 'w') as f:
            json.dump(insights_serializable, f, indent=2, default=str)
        
        self.logger.info(f" NLP analysis saved: {jobs_file.name}")
        self.logger.info(f" Insights saved: {insights_file.name}")
        
        return str(jobs_file), str(insights_file)

def main():
    """Main NLP skill extraction pipeline"""
    print(" Starting Advanced NLP Skill Extraction Pipeline")
    print("=" * 60)
    
    try:
        # Find the latest cleaned data file
        csv_files = list(PROCESSED_DATA_DIR.glob("cleaned_jobs_*.csv"))
        if not csv_files:
            print(" No cleaned data files found. Run data cleaning first.")
            print(" Run: python src/processing/data_cleaner.py")
            return
        
        latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
        print(f" Loading cleaned data: {latest_file.name}")
        
        df = pd.read_csv(latest_file)
        print(f" Loaded {len(df)} cleaned job records")
        print(f" Columns available: {list(df.columns)}")
        
        # Initialize skill extractor
        print(f"\n Initializing NLP models...")
        extractor = AdvancedSkillExtractor()
        
        # Process job descriptions
        print(f"\n Processing job descriptions with NLP...")
        processed_df = extractor.process_job_descriptions(df)
        
        # Generate comprehensive insights
        print(f"\n Generating market insights...")
        insights = extractor.generate_skill_insights(processed_df)
        
        # Save results
        print(f"\n Saving NLP results...")
        jobs_file, insights_file = extractor.save_nlp_results(processed_df, insights)
        
        # Print comprehensive results
        print("\n" + "="*60)
        print(" NLP SKILL EXTRACTION RESULTS")
        print("="*60)
        print(f" Total jobs analyzed: {insights['total_jobs_analyzed']}")
        print(f" Total unique skills found: {len(insights['unique_skills'])}")
        print(f" NLP skills extracted: {insights['nlp_extraction_stats']['total_nlp_skills_extracted']}")
        print(f" Unique NLP skills: {insights['nlp_extraction_stats']['unique_nlp_skills']}")
        print(f" Avg NLP skills per job: {insights['nlp_extraction_stats']['avg_nlp_skills_per_job']:.1f}")
        print(f" Trending skills: {len(insights['trending_skills'])}")
        print(f" Rare/specialized skills: {len(insights['rare_skills'])}")
        print(f" Locations analyzed: {len(insights['skills_by_location'])}")
        
        print(f"\n TOP 15 MOST DEMANDED SKILLS:")
        for i, (skill, count) in enumerate(list(insights['skill_frequency'].items())[:15], 1):
            percentage = (count / insights['total_jobs_analyzed']) * 100
            print(f"  {i:2d}. {skill}: {count} jobs ({percentage:.1f}%)")
        
        print(f"\n TRENDING SKILLS (High Demand):")
        trending_display = insights['trending_skills'][:10]
        print(f"  {', '.join(trending_display)}")
        
        print(f"\n SKILLS BY CATEGORY:")
        for category, skills_dict in insights['skills_by_category'].items():
            if skills_dict:  # Only show categories with skills
                category_name = category.replace('_', ' ').title()
                top_skills = list(skills_dict.keys())[:5]
                print(f"  {category_name}: {', '.join(top_skills)}")
        
        print(f"\n TOP JOB MARKETS:")
        location_totals = {
            loc: sum(skills.values()) 
            for loc, skills in insights['skills_by_location'].items()
        }
        top_locations = sorted(location_totals.items(), key=lambda x: x[1], reverse=True)[:8]
        for loc, total in top_locations:
            print(f"  {loc}: {total} skill mentions")
        
        print("\n NLP processing completed successfully!")
        print(" Data and insights saved to:")
        print(f" - Processed Jobs: {jobs_file}")
        print(f" - Skill Insights: {insights_file}")
        print("="*60)
        print("Pipeline execution finished.")

    except Exception as e:
        print(f"\n An unexpected error occurred: {e}")
        print("Please check the logs for more details.")

if __name__ == "__main__":
    main()