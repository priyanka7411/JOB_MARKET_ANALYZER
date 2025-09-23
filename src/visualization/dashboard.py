import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
import sys
from datetime import datetime
import numpy as np
from collections import Counter
from advanced_charts import AdvancedJobMarketCharts

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import *

# Page configuration
st.set_page_config(
    page_title="Job Market Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .metric-container {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .insight-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    
    .skill-tag {
        background: #e1f5fe;
        padding: 0.3rem 0.6rem;
        border-radius: 15px;
        margin: 0.2rem;
        display: inline-block;
        font-size: 0.9rem;
        color: #01579b;
    }
</style>
""", unsafe_allow_html=True)

class JobMarketDashboard:
    """Professional Job Market Intelligence Dashboard"""
    
    def __init__(self):
        self.load_data()
        
    def load_data(self):
        """Load all processed data for dashboard"""
        try:
            # Load main NLP processed data
            nlp_files = list(PROCESSED_DATA_DIR.glob("jobs_with_nlp_*.csv"))
            if nlp_files:
                latest_nlp = max(nlp_files, key=lambda x: x.stat().st_mtime)
                self.jobs_df = pd.read_csv(latest_nlp)
                
                # Parse JSON columns
                if 'skills_categorized' in self.jobs_df.columns:
                    self.jobs_df['skills_categorized'] = self.jobs_df['skills_categorized'].apply(self.safe_json_parse)
                if 'skills_all' in self.jobs_df.columns:
                    self.jobs_df['skills_all'] = self.jobs_df['skills_all'].apply(self.safe_list_parse)
                if 'skills_nlp_extracted' in self.jobs_df.columns:
                    self.jobs_df['skills_nlp_extracted'] = self.jobs_df['skills_nlp_extracted'].apply(self.safe_list_parse)
            
            # Load analytics data
            self.skill_demand_df = self.load_analytics_file("skill_demand_analysis_*.csv")
            self.location_analysis_df = self.load_analytics_file("location_analysis_*.csv") 
            self.company_analysis_df = self.load_analytics_file("company_analysis_*.csv")
            
            # Load insights
            insight_files = list(PROCESSED_DATA_DIR.glob("skill_insights_*.json"))
            if insight_files:
                latest_insights = max(insight_files, key=lambda x: x.stat().st_mtime)
                with open(latest_insights, 'r') as f:
                    self.insights = json.load(f)
            else:
                self.insights = {}
                
        except Exception as e:
            st.error(f"Error loading data: {e}")
            self.jobs_df = pd.DataFrame()
            self.skill_demand_df = pd.DataFrame()
            self.location_analysis_df = pd.DataFrame()
            self.company_analysis_df = pd.DataFrame()
            self.insights = {}
    
    def load_analytics_file(self, pattern: str) -> pd.DataFrame:
        """Load latest analytics file matching pattern"""
        files = list(PROCESSED_DATA_DIR.glob(pattern))
        if files:
            latest_file = max(files, key=lambda x: x.stat().st_mtime)
            return pd.read_csv(latest_file)
        return pd.DataFrame()
    
    def safe_json_parse(self, value):
        """Safely parse JSON strings"""
        if pd.isna(value) or value == '':
            return {}
        try:
            if isinstance(value, str):
                return json.loads(value) if value.startswith('{') else {}
            return value if isinstance(value, dict) else {}
        except:
            return {}
    
    def safe_list_parse(self, value):
        """Safely parse list strings"""
        if pd.isna(value) or value == '':
            return []
        try:
            if isinstance(value, str):
                return json.loads(value) if value.startswith('[') else []
            return value if isinstance(value, list) else []
        except:
            return []
    
    def render_header(self):
        """Render dashboard header with key metrics"""
        st.markdown('<h1 class="main-header">Job Market Intelligence Dashboard</h1>', unsafe_allow_html=True)
        st.markdown("*Real-time insights from India's tech job market powered by AI and NLP*")
        
        # Key metrics row
        if not self.jobs_df.empty:
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric(
                    label="Total Jobs Analyzed",
                    value=f"{len(self.jobs_df):,}",
                    delta="From Naukri.com"
                )
            
            with col2:
                unique_skills = len(self.insights.get('unique_skills', []))
                st.metric(
                    label="Unique Skills Found", 
                    value=f"{unique_skills:,}",
                    delta="NLP Extracted"
                )
            
            with col3:
                companies = self.jobs_df['company'].nunique() if 'company' in self.jobs_df.columns else 0
                st.metric(
                    label="Companies",
                    value=f"{companies:,}",
                    delta="Hiring Now"
                )
            
            with col4:
                locations = self.jobs_df['location'].nunique() if 'location' in self.jobs_df.columns else 0
                st.metric(
                    label="Job Markets",
                    value=f"{locations:,}",
                    delta="Cities"
                )
            
            with col5:
                trending_count = len(self.insights.get('trending_skills', []))
                st.metric(
                    label="Trending Skills",
                    value=f"{trending_count:,}",
                    delta="High Demand"
                )
    
    def render_skill_analytics(self):
        """Render comprehensive skill analytics"""
        st.header("Skill Demand Analytics")
        
        if self.skill_demand_df.empty:
            st.warning("No skill demand data available")
            return
        
        # Skill demand overview
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Top skills chart
            top_skills = self.skill_demand_df.head(20)
            
            fig_skills = px.bar(
                top_skills,
                x='demand_percentage',
                y='skill_name',
                title='Top 20 Most In-Demand Skills',
                labels={'demand_percentage': 'Job Demand (%)', 'skill_name': 'Skill'},
                color='demand_percentage',
                color_continuous_scale='viridis',
                orientation='h'
            )
            fig_skills.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_skills, use_container_width=True)
        
        with col2:
            # Skill categories pie chart
            if 'skill_category' in top_skills.columns:
                category_counts = top_skills['skill_category'].value_counts()
                
                fig_categories = px.pie(
                    values=category_counts.values,
                    names=category_counts.index,
                    title='Skills by Category',
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_categories.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_categories, use_container_width=True)
        
        # Skills matrix analysis
        st.subheader("Skills Market Matrix")
        
        if len(self.skill_demand_df) > 10:
            # Create skills matrix: Demand vs Company Diversity
            matrix_data = self.skill_demand_df.head(50).copy()
            
            fig_matrix = px.scatter(
                matrix_data,
                x='company_count',
                y='demand_percentage', 
                size='job_count',
                color='skill_category',
                hover_name='skill_name',
                title='Skills Market Positioning (Size = Job Count)',
                labels={
                    'company_count': 'Company Diversity',
                    'demand_percentage': 'Market Demand (%)'
                }
            )
            
            fig_matrix.add_hline(y=matrix_data['demand_percentage'].median(), 
                                 line_dash="dash", annotation_text="Median Demand")
            fig_matrix.add_vline(x=matrix_data['company_count'].median(), 
                                 line_dash="dash", annotation_text="Median Company Diversity")
            
            st.plotly_chart(fig_matrix, use_container_width=True)
        
        # Skills insights
        self.render_skills_insights()
    
    def render_skills_insights(self):
        """Render actionable skill insights"""
        st.subheader("Actionable Skill Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="insight-box">
                <h4>High-Growth Skills</h4>
                <p>Skills with highest market penetration and company adoption</p>
            </div>
            """, unsafe_allow_html=True)
            
            if not self.skill_demand_df.empty:
                high_growth = self.skill_demand_df.head(5)['skill_name'].tolist()
                for skill in high_growth:
                    st.markdown(f'<span class="skill-tag">{skill}</span>', unsafe_allow_html=True)
    
        with col2:
            st.markdown("""
            <div class="insight-box">
                <h4>Niche Specializations</h4>
                <p>Lower volume but high-value specialized skills</p>
            </div>
            """, unsafe_allow_html=True)
            
            if not self.skill_demand_df.empty:
                # Find skills with low job count but high company diversity
                niche_skills = self.skill_demand_df[
                    (self.skill_demand_df['job_count'] <= 10) & 
                    (self.skill_demand_df['company_count'] >= 5)
                ]['skill_name'].head(5).tolist()
                
                for skill in niche_skills:
                    st.markdown(f'<span class="skill-tag">{skill}</span>', unsafe_allow_html=True)
    
        with col3:
            st.markdown("""
            <div class="insight-box">
                <h4>Enterprise Skills</h4>
                <p>Skills preferred by top-rated companies</p>
            </div>
            """, unsafe_allow_html=True)
            
            if not self.skill_demand_df.empty:
                enterprise_skills = self.skill_demand_df[
                    self.skill_demand_df['avg_company_rating'] >= 4.0
                ]['skill_name'].head(5).tolist()
                
                for skill in enterprise_skills:
                    st.markdown(f'<span class="skill-tag">{skill}</span>', unsafe_allow_html=True)
    
    def render_geographic_analysis(self):
        """Render geographic job market analysis"""
        st.header("Geographic Job Market Analysis")
        
        if self.location_analysis_df.empty:
            st.warning("No location analysis data available")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Job distribution by location
            top_locations = self.location_analysis_df.head(15)
            
            fig_locations = px.bar(
                top_locations,
                x='job_count',
                y='location',
                title='Job Distribution by Location',
                color='avg_company_rating',
                color_continuous_scale='RdYlGn',
                labels={'job_count': 'Number of Jobs', 'location': 'Location'}
            )
            fig_locations.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_locations, use_container_width=True)
        
        with col2:
            # Location competitiveness matrix
            if len(self.location_analysis_df) > 5:
                fig_competitive = px.scatter(
                    self.location_analysis_df.head(20),
                    x='company_count',
                    y='avg_skills_per_job',
                    size='job_count',
                    color='avg_company_rating',
                    hover_name='location',
                    title='Location Competitiveness Matrix',
                    labels={
                        'company_count': 'Company Diversity',
                        'avg_skills_per_job': 'Avg Skills Required'
                    },
                    color_continuous_scale='viridis'
                )
                st.plotly_chart(fig_competitive, use_container_width=True)
        
        # Remote work analysis
        st.subheader("Remote Work Trends")
        
        if 'remote_job_percentage' in self.location_analysis_df.columns:
            remote_data = self.location_analysis_df.head(10).copy()
            
            fig_remote = px.bar(
                remote_data,
                x='location',
                y='remote_job_percentage', 
                title='Remote Work Availability by Location',
                color='remote_job_percentage',
                color_continuous_scale='blues'
            )
            fig_remote.update_xaxes(tickangle=45)
            st.plotly_chart(fig_remote, use_container_width=True)
    
    def render_company_analysis(self):
        """Render company hiring analysis"""
        st.header("Company Hiring Intelligence")
        
        if self.company_analysis_df.empty:
            st.warning("No company analysis data available")
            return
        
        # Company overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Top hiring companies
            top_companies = self.company_analysis_df.head(10)
            
            fig_companies = px.bar(
                top_companies,
                x='total_jobs',
                y='company_name',
                title='Top 10 Hiring Companies',
                color='avg_rating',
                color_continuous_scale='RdYlGn',
                orientation='h'
            )
            fig_companies.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_companies, use_container_width=True)
        
        with col2:
            # Company rating vs hiring activity
            fig_rating = px.scatter(
                self.company_analysis_df.head(50),
                x='avg_rating',
                y='total_jobs',
                size='avg_skills_per_job',
                hover_name='company_name',
                title='Rating vs Hiring Activity',
                labels={
                    'avg_rating': 'Average Rating',
                    'total_jobs': 'Total Jobs Posted'
                }
            )
            st.plotly_chart(fig_rating, use_container_width=True)
        
        with col3:
            # Remote-friendly companies
            if 'remote_percentage' in self.company_analysis_df.columns:
                remote_companies = self.company_analysis_df[
                    self.company_analysis_df['remote_percentage'] > 0
                ].head(10)
                
                fig_remote_companies = px.bar(
                    remote_companies,
                    x='remote_percentage',
                    y='company_name',
                    title='Most Remote-Friendly Companies',
                    color='remote_percentage',
                    color_continuous_scale='greens',
                    orientation='h'
                )
                fig_remote_companies.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_remote_companies, use_container_width=True)
    
    def render_interactive_filters(self):
        """Render interactive filtering sidebar"""
        st.sidebar.header("Dashboard Filters")
        
        if self.jobs_df.empty:
            return {}
        
        filters = {}
        
        # Skill category filter
        if 'skills_categorized' in self.jobs_df.columns:
            all_categories = set()
            for categories in self.jobs_df['skills_categorized'].dropna():
                if isinstance(categories, dict):
                    all_categories.update(categories.keys())
            
            if all_categories:
                filters['skill_categories'] = st.sidebar.multiselect(
                    "Skill Categories",
                    options=sorted(list(all_categories)),
                    default=[]
                )
        
        # Location filter
        if 'location' in self.jobs_df.columns:
            locations = sorted(self.jobs_df['location'].dropna().unique())
            filters['locations'] = st.sidebar.multiselect(
                "Locations",
                options=locations,
                default=[]
            )
        
        # Company filter
        if 'company' in self.jobs_df.columns:
            top_companies = self.jobs_df['company'].value_counts().head(20).index.tolist()
            filters['companies'] = st.sidebar.multiselect(
                "Companies",
                options=top_companies,
                default=[]
            )
        
        # Experience range
        if 'experience_min' in self.jobs_df.columns:
            exp_min = int(self.jobs_df['experience_min'].fillna(0).min())
            exp_max = int(self.jobs_df['experience_max'].fillna(10).max())
            
            filters['experience_range'] = st.sidebar.slider(
                "Experience Range (Years)",
                min_value=exp_min,
                max_value=exp_max,
                value=(exp_min, exp_max)
            )
        
        # Remote work preference
        filters['remote_only'] = st.sidebar.checkbox("Remote Jobs Only")
        
        return filters
    
    def render_export_options(self):
        """Render data export options"""
        st.sidebar.header("Export Data")
        
        if st.sidebar.button("Export Skill Analysis"):
            if not self.skill_demand_df.empty:
                csv = self.skill_demand_df.to_csv(index=False)
                st.sidebar.download_button(
                    label="Download Skill Data CSV",
                    data=csv,
                    file_name=f"skill_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        if st.sidebar.button("Export Location Analysis"):
            if not self.location_analysis_df.empty:
                csv = self.location_analysis_df.to_csv(index=False)
                st.sidebar.download_button(
                    label="Download Location Data CSV",
                    data=csv,
                    file_name=f"location_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        if st.sidebar.button("Export Company Analysis"):
            if not self.company_analysis_df.empty:
                csv = self.company_analysis_df.to_csv(index=False)
                st.sidebar.download_button(
                    label="Download Company Data CSV", 
                    data=csv,
                    file_name=f"company_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

    def render_advanced_visualizations(self):
        """Render advanced Plotly visualizations"""
        st.header("Advanced Market Visualizations")
    
        # Initialize advanced charts
        charts = AdvancedJobMarketCharts()
    
        # Create tabs for different advanced visualizations
        viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
            "3D Landscape", "Skills Network", "Animated Trends", "Category Sunburst"
        ])
    
        with viz_tab1:
            if not self.skill_demand_df.empty:
                st.plotly_chart(
                    charts.create_3d_skill_landscape(self.skill_demand_df),
                    use_container_width=True
                )
                st.info("💡 **3D Skills Landscape**: Explore skills in 3D space where X=Job Count, Y=Company Count, Z=Market Demand%")
        with viz_tab2:
            if not self.jobs_df.empty:
                st.plotly_chart(
                    charts.create_skill_network_graph(self.jobs_df),
                    use_container_width=True
                )
                st.info("💡 **Skills Network**: Connected skills that frequently appear together in job postings")
        with viz_tab3:
            if not self.jobs_df.empty:
                st.plotly_chart(
                    charts.create_animated_skill_demand(self.jobs_df),
                    use_container_width=True
                )
                st.info("💡 **Animated Trends**: Press play to see how skill demand evolved over time")
        with viz_tab4:
            if not self.jobs_df.empty:
                st.plotly_chart(
                    charts.create_skill_category_sunburst(self.jobs_df),
                    use_container_width=True
                )
                st.info("💡 **Category Sunburst**: Hierarchical view of skills organized by categories")
    
        # Advanced analytics section
        st.subheader("Advanced Analytics")
    
        col1, col2 = st.columns(2)
    
        with col1:
            if not self.skill_demand_df.empty:
                st.plotly_chart(
                    charts.create_skill_rarity_vs_value_chart(self.skill_demand_df),
                    use_container_width=True
                )
        with col2:
            if not self.jobs_df.empty:
                st.plotly_chart(
                    charts.create_company_skill_matrix(self.jobs_df),
                    use_container_width=True
                )
    
    def run(self):
        """Main dashboard execution"""
        # Render filters
        filters = self.render_interactive_filters()
        self.render_export_options()
        
        # Main dashboard content
        self.render_header()
        
        # Dashboard tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Skills Analytics", 
            "Geographic Insights", 
            "Company Intelligence", 
            "Advanced Visuals",
            "Market Trends"
        ])
        
        with tab1:
            self.render_skill_analytics()
        
        with tab2:
            self.render_geographic_analysis()
        
        with tab3:
            self.render_company_analysis()
        
        with tab4:
            self.render_advanced_visualizations()
        
        with tab5:
            self.render_market_trends()
        
        # Footer
        st.markdown("---")
        st.markdown("*Dashboard powered by Advanced NLP • Data updated daily from Naukri.com • Built with Streamlit & Plotly*")
    
    def render_market_trends(self):
        """Render market trends analysis"""
        st.header("Job Market Trends")
        
        if self.jobs_df.empty:
            st.warning("No job data available")
            return
        
        # Skills demand over time (if date data available)
        if 'scraped_at' in self.jobs_df.columns:
            self.jobs_df['scraped_date'] = pd.to_datetime(self.jobs_df['scraped_at']).dt.date
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Jobs by date
                jobs_by_date = self.jobs_df.groupby('scraped_date').size().reset_index(name='job_count')
                
                fig_timeline = px.line(
                    jobs_by_date,
                    x='scraped_date',
                    y='job_count',
                    title='Job Posting Timeline',
                    markers=True
                )
                st.plotly_chart(fig_timeline, use_container_width=True)
            
            with col2:
                # Category distribution
                if 'job_category' in self.jobs_df.columns:
                    category_counts = self.jobs_df['job_category'].value_counts()
                    
                    fig_categories = px.pie(
                        values=category_counts.values,
                        names=category_counts.index,
                        title='Job Categories Distribution'
                    )
                    st.plotly_chart(fig_categories, use_container_width=True)

def main():
    """Main dashboard function"""
    dashboard = JobMarketDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()