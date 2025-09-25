import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional
import io
import base64

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import *

class ExecutiveReportGenerator:
    """Generate executive-level reports and insights"""
    
    def __init__(self, jobs_df: pd.DataFrame, skill_demand_df: pd.DataFrame, 
                 location_df: pd.DataFrame, company_df: pd.DataFrame):
        self.jobs_df = jobs_df if jobs_df is not None else pd.DataFrame()
        self.skill_demand_df = skill_demand_df if skill_demand_df is not None else pd.DataFrame()
        self.location_df = location_df if location_df is not None else pd.DataFrame()
        self.company_df = company_df if company_df is not None else pd.DataFrame()
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    def generate_executive_summary(self) -> Dict:
        """Generate high-level executive insights"""
        summary = {
            'overview': {},
            'key_insights': [],
            'market_trends': {},
            'recommendations': [],
            'kpis': {}
        }
        
        # Overview metrics with safe handling
        summary['overview'] = {
            'total_jobs': len(self.jobs_df) if not self.jobs_df.empty else 0,
            'unique_skills': len(self.skill_demand_df) if not self.skill_demand_df.empty else 0,
            'active_companies': (len(self.company_df) if not self.company_df.empty 
                               else (self.jobs_df['company'].nunique() if 'company' in self.jobs_df.columns and not self.jobs_df.empty else 0)),
            'job_markets': (len(self.location_df) if not self.location_df.empty 
                          else (self.jobs_df['location'].nunique() if 'location' in self.jobs_df.columns and not self.jobs_df.empty else 0)),
            'analysis_period': self.timestamp
        }
        
        # Key insights with error handling
        try:
            if not self.skill_demand_df.empty and 'skill_name' in self.skill_demand_df.columns:
                top_skill = self.skill_demand_df.iloc[0]
                skill_name = str(top_skill['skill_name']).title()
                demand_pct = top_skill.get('demand_percentage', 0)
                company_count = top_skill.get('company_count', 0)
                
                summary['key_insights'].extend([
                    f"🔥 Most in-demand skill: {skill_name} appears in {demand_pct:.1f}% of jobs",
                    f"🏢 {company_count} companies are actively hiring for {skill_name}",
                    f"📈 Top 5 skills account for {self.skill_demand_df.head(5)['demand_percentage'].sum():.1f}% of total demand"
                ])
        except Exception as e:
            summary['key_insights'].append("Skills analysis data unavailable")
        
        try:
            if not self.location_df.empty and 'location' in self.location_df.columns:
                top_location = self.location_df.iloc[0]
                location_name = str(top_location['location'])
                job_count = top_location.get('job_count', 0)
                
                summary['key_insights'].extend([
                    f"📍 {location_name} leads job market with {job_count} positions"
                ])
                
                if 'avg_company_rating' in self.location_df.columns:
                    avg_rating = self.location_df.head(5)['avg_company_rating'].mean()
                    if not pd.isna(avg_rating):
                        summary['key_insights'].append(f"⭐ Average company rating in top markets: {avg_rating:.2f}/5.0")
        except Exception as e:
            summary['key_insights'].append("Location analysis data unavailable")
        
        # Market trends with safe column checking
        try:
            if not self.jobs_df.empty:
                if 'remote_friendly' in self.jobs_df.columns:
                    remote_pct = (self.jobs_df['remote_friendly'].sum() / len(self.jobs_df) * 100)
                    summary['market_trends']['remote_work'] = f"{remote_pct:.1f}% of jobs offer remote work options"
                
                if 'total_skills_count' in self.jobs_df.columns:
                    avg_skills = self.jobs_df['total_skills_count'].mean()
                    if not pd.isna(avg_skills):
                        summary['market_trends']['skill_complexity'] = f"Jobs require average of {avg_skills:.1f} skills, indicating increasing complexity"
        except Exception as e:
            summary['market_trends']['data_quality'] = "Market trends analysis limited due to data constraints"
        
        # Strategic recommendations
        try:
            if not self.skill_demand_df.empty and 'skill_name' in self.skill_demand_df.columns:
                trending_skills = self.skill_demand_df.head(3)['skill_name'].tolist()
                trending_skills_formatted = [str(s).title() for s in trending_skills]
                
                summary['recommendations'].extend([
                    f"💡 Focus recruitment on high-demand skills: {', '.join(trending_skills_formatted)}",
                    "📚 Invest in upskilling programs for emerging technologies",
                    "🌐 Consider remote-first strategies to access broader talent pool"
                ])
            else:
                summary['recommendations'].extend([
                    "📊 Collect more comprehensive skill data for better insights",
                    "🔍 Analyze job market trends to identify opportunities",
                    "💼 Focus on building a diverse talent pipeline"
                ])
        except Exception as e:
            summary['recommendations'].append("Strategic recommendations require additional data analysis")
        
        # KPIs
        summary['kpis'] = {
            'market_coverage': f"{summary['overview']['job_markets']} cities analyzed",
            'skill_diversity': f"{summary['overview']['unique_skills']} distinct skills identified",
            'company_participation': f"{summary['overview']['active_companies']} companies actively hiring",
            'data_freshness': "Updated within last 24 hours"
        }
        
        return summary
    
    def create_executive_dashboard(self) -> List[go.Figure]:
        """Create executive-level charts with error handling"""
        figures = []
        
        try:
            # 1. Market Overview KPI Chart
            if not self.skill_demand_df.empty and not self.location_df.empty:
                kpi_data = {
                    'Metric': ['Skills Analyzed', 'Job Markets', 'Active Companies', 'Job Postings'],
                    'Value': [
                        len(self.skill_demand_df),
                        len(self.location_df), 
                        self.jobs_df['company'].nunique() if 'company' in self.jobs_df.columns and not self.jobs_df.empty else 0,
                        len(self.jobs_df)
                    ],
                    'Category': ['Skills', 'Markets', 'Companies', 'Jobs']
                }
                
                fig_kpi = px.bar(
                    pd.DataFrame(kpi_data),
                    x='Metric',
                    y='Value',
                    color='Category',
                    title='📊 Market Intelligence Overview',
                    color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
                )
                fig_kpi.update_layout(height=400, showlegend=False)
                figures.append(fig_kpi)
        except Exception as e:
            # Create a placeholder chart
            fig_placeholder = go.Figure()
            fig_placeholder.add_annotation(text=f"Market overview chart unavailable: {str(e)}")
            figures.append(fig_placeholder)
        
        try:
            # 2. Top Skills Executive View
            if not self.skill_demand_df.empty and all(col in self.skill_demand_df.columns 
                                                    for col in ['skill_name', 'demand_percentage']):
                top_10_skills = self.skill_demand_df.head(10)
                
                fig_skills = px.bar(
                    top_10_skills,
                    x='demand_percentage',
                    y='skill_name',
                    title='🚀 Top 10 Strategic Skills (Market Demand %)',
                    color='company_count' if 'company_count' in top_10_skills.columns else None,
                    color_continuous_scale='viridis',
                    labels={'demand_percentage': 'Market Demand (%)', 'skill_name': 'Skills'},
                    orientation='h'
                )
                fig_skills.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
                figures.append(fig_skills)
        except Exception as e:
            # Create a placeholder chart
            fig_placeholder = go.Figure()
            fig_placeholder.add_annotation(text=f"Skills chart unavailable: {str(e)}")
            figures.append(fig_placeholder)
        
        try:
            # 3. Geographic Market Distribution
            if not self.location_df.empty and all(col in self.location_df.columns 
                                                for col in ['location', 'job_count']):
                top_locations = self.location_df.head(8)
                
                fig_geo = px.treemap(
                    top_locations,
                    path=['location'],
                    values='job_count',
                    color='avg_company_rating' if 'avg_company_rating' in top_locations.columns else None,
                    title='🗺️ Job Market Distribution by Location',
                    color_continuous_scale='RdYlGn'
                )
                fig_geo.update_layout(height=400)
                figures.append(fig_geo)
        except Exception as e:
            # Create a placeholder chart
            fig_placeholder = go.Figure()
            fig_placeholder.add_annotation(text=f"Geographic chart unavailable: {str(e)}")
            figures.append(fig_placeholder)
        
        try:
            # 4. Company Hiring Activity
            if not self.company_df.empty and all(col in self.company_df.columns 
                                               for col in ['total_jobs', 'company_name']):
                top_companies = self.company_df.head(10)
                
                fig_companies = px.scatter(
                    top_companies,
                    x='total_jobs',
                    y='avg_rating' if 'avg_rating' in top_companies.columns else 'total_jobs',
                    size='avg_skills_per_job' if 'avg_skills_per_job' in top_companies.columns else None,
                    color='remote_percentage' if 'remote_percentage' in top_companies.columns else None,
                    hover_name='company_name',
                    title='🏢 Top Companies: Hiring Activity vs Rating',
                    labels={'total_jobs': 'Jobs Posted', 'avg_rating': 'Company Rating'},
                    color_continuous_scale='blues'
                )
                fig_companies.update_layout(height=500)
                figures.append(fig_companies)
        except Exception as e:
            # Create a placeholder chart
            fig_placeholder = go.Figure()
            fig_placeholder.add_annotation(text=f"Company chart unavailable: {str(e)}")
            figures.append(fig_placeholder)
        
        return figures
    
    def generate_pdf_report(self, summary: Dict, figures: List[go.Figure]) -> bytes:
        """Generate PDF report with fallback to text"""
        try:
            # Try to import reportlab for PDF generation
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
            
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4)
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            title = Paragraph("Job Market Intelligence Report", styles['Title'])
            story.append(title)
            story.append(Spacer(1, 20))
            
            # Executive Summary
            exec_title = Paragraph("Executive Summary", styles['Heading1'])
            story.append(exec_title)
            
            overview_text = f"""
            <b>Market Analysis Overview</b><br/>
            • Total Jobs Analyzed: {summary['overview']['total_jobs']:,}<br/>
            • Unique Skills Identified: {summary['overview']['unique_skills']:,}<br/>
            • Active Companies: {summary['overview']['active_companies']:,}<br/>
            • Job Markets: {summary['overview']['job_markets']:,}<br/>
            • Analysis Date: {summary['overview']['analysis_period']}<br/>
            """
            
            story.append(Paragraph(overview_text, styles['Normal']))
            story.append(Spacer(1, 20))
            
            # Key Insights
            if summary['key_insights']:
                insights_title = Paragraph("Key Market Insights", styles['Heading2'])
                story.append(insights_title)
                
                for insight in summary['key_insights']:
                    # Clean the insight text for PDF
                    clean_insight = insight.replace('🔥', '•').replace('🏢', '•').replace('📈', '•').replace('📍', '•').replace('⭐', '•')
                    story.append(Paragraph(f"• {clean_insight}", styles['Normal']))
                
                story.append(Spacer(1, 20))
            
            # Recommendations
            if summary['recommendations']:
                rec_title = Paragraph("Strategic Recommendations", styles['Heading2'])
                story.append(rec_title)
                
                for rec in summary['recommendations']:
                    # Clean the recommendation text for PDF
                    clean_rec = rec.replace('💡', '•').replace('📚', '•').replace('🌐', '•').replace('📊', '•').replace('🔍', '•').replace('💼', '•')
                    story.append(Paragraph(f"• {clean_rec}", styles['Normal']))
            
            doc.build(story)
            buffer.seek(0)
            return buffer.getvalue()
            
        except ImportError:
            # Fallback: create text report
            report_text = f"""
JOB MARKET INTELLIGENCE REPORT
Generated: {self.timestamp}

EXECUTIVE SUMMARY
================
Total Jobs Analyzed: {summary['overview']['total_jobs']:,}
Unique Skills: {summary['overview']['unique_skills']:,}
Active Companies: {summary['overview']['active_companies']:,}
Job Markets: {summary['overview']['job_markets']:,}

KEY INSIGHTS
============
""" + '\n'.join([f"• {insight}" for insight in summary['key_insights']]) + f"""

RECOMMENDATIONS
===============
""" + '\n'.join([f"• {rec}" for rec in summary['recommendations']])
            
            return report_text.encode('utf-8')
        
        except Exception as e:
            # Ultimate fallback: basic text report
            fallback_text = f"""
JOB MARKET REPORT
Generated: {self.timestamp}

Report generation encountered an error: {str(e)}

BASIC SUMMARY:
Jobs Analyzed: {summary['overview']['total_jobs']}
Skills Found: {summary['overview']['unique_skills']}
Companies: {summary['overview']['active_companies']}
Locations: {summary['overview']['job_markets']}
            """
            return fallback_text.encode('utf-8')

def create_chart_export_buttons(fig: go.Figure, chart_name: str):
    """Create export buttons for charts with error handling"""
    if fig is None:
        st.error("Chart not available for export")
        return
        
    col1, col2, col3 = st.columns(3)
    
    try:
        with col1:
            if st.button(f"📊 PNG", key=f"png_{chart_name}"):
                try:
                    img_bytes = fig.to_image(format="png", width=1200, height=800)
                    st.download_button(
                        label="Download PNG",
                        data=img_bytes,
                        file_name=f"{chart_name}_{datetime.now().strftime('%Y%m%d')}.png",
                        mime="image/png",
                        key=f"download_png_{chart_name}"
                    )
                except Exception as e:
                    st.error(f"PNG export failed: {e}")
        
        with col2:
            if st.button(f"📈 SVG", key=f"svg_{chart_name}"):
                try:
                    img_bytes = fig.to_image(format="svg")
                    st.download_button(
                        label="Download SVG", 
                        data=img_bytes,
                        file_name=f"{chart_name}_{datetime.now().strftime('%Y%m%d')}.svg",
                        mime="image/svg+xml",
                        key=f"download_svg_{chart_name}"
                    )
                except Exception as e:
                    st.error(f"SVG export failed: {e}")
        
        with col3:
            if st.button(f"📋 HTML", key=f"html_{chart_name}"):
                try:
                    html_str = fig.to_html(include_plotlyjs=True)
                    st.download_button(
                        label="Download HTML",
                        data=html_str,
                        file_name=f"{chart_name}_{datetime.now().strftime('%Y%m%d')}.html",
                        mime="text/html",
                        key=f"download_html_{chart_name}"
                    )
                except Exception as e:
                    st.error(f"HTML export failed: {e}")
                    
    except Exception as e:
        st.error(f"Export buttons failed: {e}")

# Test function
def test_report_generator():
    """Test the report generator with sample data"""
    try:
        # Create sample data
        jobs_df = pd.DataFrame({
            'company': ['Company A', 'Company B', 'Company C'],
            'location': ['City 1', 'City 2', 'City 1'],
            'remote_friendly': [True, False, True],
            'total_skills_count': [5, 3, 7]
        })
        
        skill_df = pd.DataFrame({
            'skill_name': ['python', 'java', 'javascript'],
            'demand_percentage': [25.0, 20.0, 15.0],
            'company_count': [10, 8, 6],
            'job_count': [50, 40, 30]
        })
        
        location_df = pd.DataFrame({
            'location': ['City 1', 'City 2'],
            'job_count': [100, 80],
            'avg_company_rating': [4.2, 3.8]
        })
        
        company_df = pd.DataFrame({
            'company_name': ['Company A', 'Company B'],
            'total_jobs': [50, 30],
            'avg_rating': [4.5, 4.0],
            'avg_skills_per_job': [5.5, 4.0],
            'remote_percentage': [60, 20]
        })
        
        # Test report generator
        report_gen = ExecutiveReportGenerator(jobs_df, skill_df, location_df, company_df)
        summary = report_gen.generate_executive_summary()
        figures = report_gen.create_executive_dashboard()
        
        print("✅ Report generator test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Report generator test failed: {e}")
        return False

if __name__ == "__main__":
    test_report_generator()