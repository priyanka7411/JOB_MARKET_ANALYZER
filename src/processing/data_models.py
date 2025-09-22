from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict
from datetime import datetime
import re
import ast

class JobData(BaseModel):
    """Pydantic model matching your actual CSV structure"""
    
    job_id: Optional[str] = None
    title: Optional[str] = None
    company: Optional[str] = None
    location: Optional[str] = None
    experience: Optional[str] = None
    salary: Optional[str] = None
    description: Optional[str] = None
    skills: Optional[str] = None  # Your skills are stored as string representation of list
    job_url: Optional[str] = None
    company_rating: Optional[str] = None
    posted_date: Optional[str] = None
    scraped_at: Optional[str] = None
    search_keyword: Optional[str] = None
    page_number: Optional[int] = None
    
    @validator('skills')
    def parse_skills(cls, v):
        """Parse skills from string representation of list"""
        if v is None or v == '' or str(v).lower() in ['nan', 'null']:
            return []
        
        if isinstance(v, str):
            try:
                # Parse the string representation of list
                skills_list = ast.literal_eval(v)
                if isinstance(skills_list, list):
                    return [skill.strip().lower() for skill in skills_list if skill.strip()]
            except:
                # If parsing fails, split by comma
                return [skill.strip().lower() for skill in v.split(',') if skill.strip()]
        
        return []
    
    @validator('title', 'company', 'location', 'description')
    def clean_text_fields(cls, v):
        """Clean text fields"""
        if v is None or str(v).lower() in ['nan', 'null', '']:
            return None
        return ' '.join(str(v).split()).strip()
    
    @validator('company_rating')
    def parse_rating(cls, v):
        """Parse company rating"""
        if v is None or str(v).lower() in ['nan', 'null', '']:
            return None
        try:
            return float(v)
        except:
            return None

class ProcessedJob(BaseModel):
    """Processed job model for your data"""
    
    job_id: str
    title: str = "Unknown"
    company: str = "Unknown" 
    location: str = "Unknown"
    experience_min: Optional[int] = None
    experience_max: Optional[int] = None
    salary_min: Optional[float] = None
    salary_max: Optional[float] = None
    description_clean: Optional[str] = None
    skills_extracted: List[str] = Field(default_factory=list)
    seniority_level: Optional[str] = "unknown"
    remote_friendly: bool = False
    scraped_at: datetime = Field(default_factory=datetime.now)
    processed_at: datetime = Field(default_factory=datetime.now)