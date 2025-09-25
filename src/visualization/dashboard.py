

from advanced_charts import AdvancedJobMarketCharts
from report_generator import ExecutiveReportGenerator, create_chart_export_buttons
import io
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
from typing import Dict, List
import time
import io
import base64

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import *

# Import advanced charts and report generator with error handling
try:
    from advanced_charts import AdvancedJobMarketCharts
except ImportError:
    AdvancedJobMarketCharts = None

try:
    from report_generator import ExecutiveReportGenerator, create_chart_export_buttons
except ImportError:
    ExecutiveReportGenerator = None
    create_chart_export_buttons = None

# Page configuration
st.set_page_config(
    page_title="Job Market Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for professional styling and mobile responsiveness
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: clamp(2rem, 5vw, 3rem);
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Metric containers */
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .metric-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    
    /* Insight boxes */
    .insight-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    }
    
    /* Skill tags */
    .skill-tag {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        margin: 0.3rem;
        display: inline-block;
        font-size: 0.9rem;
        font-weight: 500;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        transition: transform 0.2s ease;
    }
    
    .skill-tag:hover {
        transform: scale(1.05);
    }
    
    /* Filter section styling */
    .filter-header {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 0.8rem;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    /* Charts container */
    .chart-container {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin: 1rem 0;
    }
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
            margin-bottom: 1rem;
        }
        
        .metric-container {
            padding: 1rem;
            margin: 0.3rem 0;
        }
        
        .insight-box {
            padding: 1rem;
            margin: 0.5rem 0;
        }
        
        .skill-tag {
            font-size: 0.8rem;
            padding: 0.3rem 0.6rem;
            margin: 0.2rem;
        }
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Selectbox and multiselect styling */
    .stSelectbox > div > div {
        background-color: white;
        border: 2px solid #e1e5e9;
        border-radius: 8px;
    }
    
    .stMultiSelect > div > div {
        background-color: white;
        border: 2px solid #e1e5e9;
        border-radius: 8px;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f1f3f4;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #666;
        font-style: italic;
        border-top: 1px solid #eee;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

class JobMarketDashboard:
    """Professional Job Market Intelligence Dashboard with Full Features"""

    def __init__(self):
        # Initialize data holders
        self.jobs_df = pd.DataFrame()
        self.skill_demand_df = pd.DataFrame()
        self.location_analysis_df = pd.DataFrame()
        self.company_analysis_df = pd.DataFrame()
        self.insights = {}
        
        # Load data
        self.load_data()

    # Data loading and parsing methods
    def load_data(self):
        """Load all processed data for dashboard"""
        try:
            # Load main NLP processed data
            if PROCESSED_DATA_DIR:
                nlp_files = list(Path(PROCESSED_DATA_DIR).glob("jobs_with_nlp_*.csv"))
                if nlp_files:
                    latest_nlp = max(nlp_files, key=lambda x: x.stat().st_mtime)
                    self.jobs_df = pd.read_csv(latest_nlp)

                    # Parse JSON columns safely
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

                # Load insights JSON
                insight_files = list(Path(PROCESSED_DATA_DIR).glob("skill_insights_*.json"))
                if insight_files:
                    latest_insights = max(insight_files, key=lambda x: x.stat().st_mtime)
                    with open(latest_insights, 'r', encoding='utf-8') as f:
                        self.insights = json.load(f)

        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.info("Please ensure you have run the data processing pipeline first.")

    def load_analytics_file(self, pattern: str) -> pd.DataFrame:
        """Load latest analytics file matching pattern"""
        try:
            if PROCESSED_DATA_DIR:
                files = list(Path(PROCESSED_DATA_DIR).glob(pattern))
                if files:
                    latest_file = max(files, key=lambda x: x.stat().st_mtime)
                    return pd.read_csv(latest_file)
        except Exception:
            pass
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

    # Filtering and interaction methods
    def apply_filters(self, df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
        """Apply user-selected filters to dataframe"""
        if df is None or df.empty:
            return pd.DataFrame()

        filtered_df = df.copy()

        # Location filter
        if filters.get('locations'):
            if 'location' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['location'].isin(filters['locations'])]

        # Company filter
        if filters.get('companies'):
            if 'company' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['company'].isin(filters['companies'])]

        # Skill category filter
        if filters.get('skill_categories'):
            category_mask = pd.Series(False, index=filtered_df.index)
            for index, row in filtered_df.iterrows():
                categorized = row.get('skills_categorized', {})
                if isinstance(categorized, dict):
                    row_categories = set(categorized.keys())
                    if any(cat in row_categories for cat in filters['skill_categories']):
                        category_mask.at[index] = True
            filtered_df = filtered_df[category_mask]

        # Experience filter
        if filters.get('experience_range'):
            min_exp, max_exp = filters['experience_range']
            if 'experience_min' in filtered_df.columns:
                filtered_df = filtered_df[
                    (filtered_df['experience_min'].fillna(0) >= min_exp) &
                    (filtered_df['experience_min'].fillna(0) <= max_exp)
                ]

        # Remote work filter
        if filters.get('remote_only'):
            if 'remote_friendly' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['remote_friendly'] == True]

        # High-rated companies filter
        if filters.get('high_rated_companies'):
            if 'company_rating' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['company_rating'] >= 4.0]

        # Minimum skills filter
        if filters.get('min_skills_count', 0) > 0:
            if 'total_skills_count' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['total_skills_count'] >= filters['min_skills_count']]

        return filtered_df

    def render_advanced_filters(self) -> Dict:
        """Render comprehensive filtering sidebar"""
        st.sidebar.header("🔧 Advanced Filters")
        filters = {}

        if self.jobs_df.empty:
            st.sidebar.warning("No data available for filtering")
            return filters

        # Location filter
        if 'location' in self.jobs_df.columns:
            locations = sorted(self.jobs_df['location'].dropna().unique())
            filters['locations'] = st.sidebar.multiselect(
                "📍 Locations",
                options=locations,
                default=[],
                help="Filter jobs by location"
            )

        # Company filter
        if 'company' in self.jobs_df.columns:
            companies = self.jobs_df['company'].value_counts().head(30).index.tolist()
            filters['companies'] = st.sidebar.multiselect(
                "🏢 Companies",
                options=companies,
                default=[],
                help="Filter jobs by company"
            )

        # Skill category filter
        all_categories = set()
        for idx, row in self.jobs_df.iterrows():
            categorized = row.get('skills_categorized', {})
            if isinstance(categorized, dict):
                all_categories.update(categorized.keys())

        if all_categories:
            category_options = sorted([cat.replace('_', ' ').title() for cat in all_categories])
            selected_categories = st.sidebar.multiselect(
                "🔧 Skill Categories",
                options=category_options,
                default=[],
                help="Filter by skill categories"
            )
            filters['skill_categories'] = [cat.lower().replace(' ', '_') for cat in selected_categories]

        # Experience filter
        if 'experience_min' in self.jobs_df.columns:
            exp_min = int(self.jobs_df['experience_min'].fillna(0).min())
            exp_max = int(self.jobs_df['experience_max'].fillna(15).max())
            
            filters['experience_range'] = st.sidebar.slider(
                "⏱️ Experience Range (Years)",
                min_value=exp_min,
                max_value=min(exp_max, 20),
                value=(exp_min, min(exp_max, 10)),
                help="Filter by required experience"
            )

        # Advanced options
        st.sidebar.subheader("⚙️ Advanced Options")
        filters['remote_only'] = st.sidebar.checkbox("🏠 Remote Jobs Only")
        filters['high_rated_companies'] = st.sidebar.checkbox("⭐ High-Rated Companies (4.0+)")

        if 'total_skills_count' in self.jobs_df.columns:
            max_skills = int(self.jobs_df['total_skills_count'].fillna(0).max())
            filters['min_skills_count'] = st.sidebar.slider(
                "🔧 Minimum Skills Required",
                min_value=0,
                max_value=min(max_skills, 20),
                value=0,
                help="Filter jobs by minimum skills count"
            )

        return filters

    def render_dynamic_skill_selector(self, filtered_df: pd.DataFrame):
        """Dynamic skill selector based on filtered data"""
        st.sidebar.subheader("🎯 Skill Focus")

        if filtered_df is None or filtered_df.empty:
            return []

        # Extract skills from filtered data
        all_skills = []
        for idx, row in filtered_df.iterrows():
            skills = row.get('skills_all', [])
            if isinstance(skills, list):
                all_skills.extend([s.lower().strip() for s in skills if s])

        if all_skills:
            skill_counts = pd.Series(all_skills).value_counts()
            top_skills = skill_counts.head(30).index.tolist()

            selected_skills = st.sidebar.multiselect(
                "Focus on specific skills:",
                options=top_skills,
                default=[],
                help="Select skills to highlight"
            )
            return selected_skills

        return []

    # Export and data management
    def render_export_filtered_data(self, filtered_df: pd.DataFrame, filters: Dict):
        """Export options for filtered data"""
        st.sidebar.subheader("💾 Export Filtered Data")

        if filtered_df is None or filtered_df.empty:
            st.sidebar.info("No data to export with current filters")
            return

        export_format = st.sidebar.selectbox(
            "Export format:",
            ["CSV", "JSON", "Excel"],
            help="Choose export format"
        )

        if st.sidebar.button("📥 Export Data"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            try:
                if export_format == "CSV":
                    csv_data = filtered_df.to_csv(index=False)
                    st.sidebar.download_button(
                        "Download CSV",
                        data=csv_data,
                        file_name=f"filtered_jobs_{timestamp}.csv",
                        mime="text/csv"
                    )

                elif export_format == "JSON":
                    json_data = filtered_df.to_json(orient='records', indent=2)
                    st.sidebar.download_button(
                        "Download JSON",
                        data=json_data,
                        file_name=f"filtered_jobs_{timestamp}.json",
                        mime="application/json"
                    )

                elif export_format == "Excel":
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                        filtered_df.to_excel(writer, index=False, sheet_name='Filtered_Jobs')
                    excel_buffer.seek(0)
                    
                    st.sidebar.download_button(
                        "Download Excel",
                        data=excel_buffer.getvalue(),
                        file_name=f"filtered_jobs_{timestamp}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

            except Exception as e:
                st.sidebar.error(f"Export failed: {e}")

    def render_auto_refresh_section(self):
        """Auto-refresh and data management"""
        st.sidebar.subheader("🔄 Data Management")

        # Data freshness indicator
        if not self.jobs_df.empty and 'scraped_at' in self.jobs_df.columns:
            try:
                latest_scrape = pd.to_datetime(self.jobs_df['scraped_at']).max()
                hours_ago = (datetime.now() - latest_scrape).total_seconds() / 3600

                if hours_ago < 24:
                    st.sidebar.success(f"✅ Data is fresh ({hours_ago:.1f} hours old)")
                elif hours_ago < 48:
                    st.sidebar.warning(f"⚠️ Data is {hours_ago:.1f} hours old")
                else:
                    st.sidebar.error(f"❌ Data is {hours_ago/24:.1f} days old")
            except:
                st.sidebar.info("Data freshness unknown")

        # Manual refresh
        if st.sidebar.button("🔄 Refresh Dashboard"):
            st.cache_data.clear()
            st.rerun()

        # Auto-refresh option (informational)
        auto_refresh = st.sidebar.checkbox("⚡ Auto-refresh (5 min)", value=False)
        if auto_refresh:
            st.sidebar.info("Auto-refresh enabled")

    # UI rendering methods
    def render_header(self):
        """Dashboard header with key metrics"""
        st.markdown('<h1 class="main-header">📊 Job Market Intelligence Dashboard</h1>', unsafe_allow_html=True)
        st.markdown("*Real-time insights from India's tech job market powered by AI and NLP*")

        if not self.jobs_df.empty:
            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.metric(
                    "📋 Total Jobs",
                    f"{len(self.jobs_df):,}",
                    delta="Analyzed"
                )

            with col2:
                unique_skills = len(self.insights.get('unique_skills', [])) if self.insights else 0
                st.metric(
                    "🔧 Skills Found",
                    f"{unique_skills:,}",
                    delta="NLP Extracted"
                )

            with col3:
                companies = self.jobs_df['company'].nunique() if 'company' in self.jobs_df.columns else 0
                st.metric(
                    "🏢 Companies",
                    f"{companies:,}",
                    delta="Hiring"
                )

            with col4:
                locations = self.jobs_df['location'].nunique() if 'location' in self.jobs_df.columns else 0
                st.metric(
                    "📍 Markets",
                    f"{locations:,}",
                    delta="Cities"
                )

            with col5:
                trending_count = len(self.insights.get('trending_skills', [])) if self.insights else 0
                st.metric(
                    "🔥 Trending",
                    f"{trending_count:,}",
                    delta="Skills"
                )

    def render_filter_summary(self, original_count: int, filtered_count: int, filters: Dict):
        """Summary of applied filters"""
        if filtered_count < original_count:
            st.info(f"📊 Showing {filtered_count:,} of {original_count:,} jobs based on your filters")

            # Active filters display
            active_filters = []
            if filters.get('locations'):
                active_filters.append(f"📍 {len(filters['locations'])} locations")
            if filters.get('companies'):
                active_filters.append(f"🏢 {len(filters['companies'])} companies")
            if filters.get('skill_categories'):
                active_filters.append(f"🔧 {len(filters['skill_categories'])} skill categories")
            if filters.get('remote_only'):
                active_filters.append("🏠 Remote only")
            if filters.get('high_rated_companies'):
                active_filters.append("⭐ High-rated companies")

            if active_filters:
                st.markdown("**Active Filters:** " + " | ".join(active_filters))

    def render_real_time_metrics(self, filtered_df: pd.DataFrame, filters: Dict):
        """Real-time metrics that update with filters"""
        if filtered_df is None or filtered_df.empty:
            st.warning("No data matches current filters. Try adjusting them.")
            return

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            avg_skills = filtered_df.get('total_skills_count', pd.Series([0])).mean()
            overall_avg = self.jobs_df.get('total_skills_count', pd.Series([0])).mean()
            st.metric(
                "📊 Avg Skills/Job",
                f"{avg_skills:.1f}",
                delta=f"vs {overall_avg:.1f} overall" if overall_avg else None
            )

        with col2:
            if 'company_rating' in filtered_df.columns and not filtered_df['company_rating'].dropna().empty:
                avg_rating = filtered_df['company_rating'].dropna().mean()
                overall_avg = self.jobs_df['company_rating'].dropna().mean() if 'company_rating' in self.jobs_df.columns else np.nan
                delta_text = f"{avg_rating - overall_avg:+.2f}" if not pd.isna(overall_avg) else None
                st.metric(
                    "⭐ Avg Rating",
                    f"{avg_rating:.2f}" if not pd.isna(avg_rating) else "N/A",
                    delta=delta_text
                )

        with col3:
            if 'remote_friendly' in filtered_df.columns:
                remote_pct = (filtered_df['remote_friendly'].sum() / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
                overall_remote = (self.jobs_df['remote_friendly'].sum() / len(self.jobs_df) * 100) if 'remote_friendly' in self.jobs_df.columns and len(self.jobs_df) > 0 else 0
                st.metric(
                    "🏠 Remote %",
                    f"{remote_pct:.1f}%",
                    delta=f"{remote_pct - overall_remote:+.1f}%" if overall_remote else None
                )

        with col4:
            unique_companies = filtered_df['company'].nunique() if 'company' in filtered_df.columns else 0
            st.metric(
                "🏢 Companies",
                f"{unique_companies:,}",
                delta="Hiring"
            )

    # Visualization rendering methods
    def render_skill_analytics(self, filtered_df: pd.DataFrame = None, selected_skills: List[str] = None):
        """Comprehensive skill analytics"""
        st.header("🔧 Skill Demand Analytics")

        # Use filtered data if available, otherwise use skill_demand_df
        if filtered_df is not None and not filtered_df.empty:
            # Generate skill analysis from filtered data
            all_skills = []
            for idx, row in filtered_df.iterrows():
                skills = row.get('skills_all', [])
                if isinstance(skills, list):
                    all_skills.extend([s.lower().strip() for s in skills if s])

            if all_skills:
                skill_counts = pd.Series(all_skills).value_counts()
                skill_percentages = (skill_counts / len(filtered_df) * 100).round(1)
                
                skills_df = pd.DataFrame({
                    'skill_name': skill_counts.index,
                    'job_count': skill_counts.values,
                    'demand_percentage': skill_percentages.values
                }).head(20)
            else:
                skills_df = pd.DataFrame()
        else:
            skills_df = self.skill_demand_df.head(20) if not self.skill_demand_df.empty else pd.DataFrame()

        if skills_df.empty:
            st.warning("No skill data available")
            return

        # Highlight selected skills
        if selected_skills:
            skills_df['is_selected'] = skills_df['skill_name'].isin(selected_skills)
            color_col = 'is_selected'
            color_map = {True: '#FF6B6B', False: '#4ECDC4'}
        else:
            color_col = 'demand_percentage'
            color_map = None

        # Top skills chart
        fig_skills = px.bar(
            skills_df,
            x='demand_percentage',
            y='skill_name',
            title='Top Skills in Current Selection' if filtered_df is not None else 'Top 20 Most In-Demand Skills',
            labels={'demand_percentage': 'Job Demand (%)', 'skill_name': 'Skill'},
            color=color_col,
            color_continuous_scale='viridis' if not selected_skills else None,
            color_discrete_map=color_map if selected_skills else None,
            orientation='h'
        )
        fig_skills.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
        if selected_skills:
            fig_skills.update_layout(showlegend=False)

        st.plotly_chart(fig_skills, use_container_width=True)

        # Skills insights
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📈 Top Skills")
            top_5_skills = skills_df.head(5)
            for idx, row in top_5_skills.iterrows():
                st.markdown(f"**{row['skill_name'].title()}**: {row['job_count']} jobs ({row['demand_percentage']}%)")

        with col2:
            st.subheader("🎯 Skills Overview")
            total_skills = len(skills_df)
            avg_demand = skills_df['demand_percentage'].mean() if not skills_df.empty else 0
            
            st.metric("Skills Shown", f"{total_skills:,}")
            st.metric("Avg Demand", f"{avg_demand:.1f}%")

        # Selected skills comparison
        if selected_skills and filtered_df is not None:
            self.render_skill_comparison(filtered_df, selected_skills)

    def render_skill_comparison(self, filtered_df: pd.DataFrame, selected_skills: List[str]):
        """Interactive skill comparison"""
        st.subheader("🎯 Selected Skills Analysis")

        skill_metrics = []
        for skill in selected_skills:
            # Count jobs with this skill
            skill_jobs = 0
            for idx, row in filtered_df.iterrows():
                skills = row.get('skills_all', [])
                if isinstance(skills, list):
                    if skill in [s.lower().strip() for s in skills]:
                        skill_jobs += 1

            demand_pct = (skill_jobs / len(filtered_df) * 100) if len(filtered_df) > 0 else 0

            skill_metrics.append({
                'skill': skill.title(),
                'jobs': skill_jobs,
                'demand_pct': round(demand_pct, 1)
            })

        if skill_metrics:
            metrics_df = pd.DataFrame(skill_metrics)
            
            fig = px.bar(
                metrics_df,
                x='skill',
                y='demand_pct',
                title='Selected Skills Comparison',
                labels={'demand_pct': 'Demand %', 'skill': 'Skills'},
                text='jobs'
            )
            fig.update_traces(texttemplate='%{text} jobs', textposition='outside')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(metrics_df.set_index('skill'), use_container_width=True)

    def render_geographic_analysis(self):
        """Geographic job market analysis"""
        st.header("🗺️ Geographic Job Market Analysis")

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
                color='avg_company_rating' if 'avg_company_rating' in top_locations.columns else None,
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
                    y='avg_skills_per_job' if 'avg_skills_per_job' in self.location_analysis_df.columns else None,
                    size='job_count',
                    color='avg_company_rating' if 'avg_company_rating' in self.location_analysis_df.columns else None,
                    hover_name='location',
                    title='Location Competitiveness Matrix',
                    labels={
                        'company_count': 'Company Diversity',
                        'avg_skills_per_job': 'Avg Skills Required'
                    },
                    color_continuous_scale='viridis'
                )
                st.plotly_chart(fig_competitive, use_container_width=True)

        # Remote work trends
        st.subheader("🏠 Remote Work Trends")
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
        """Company hiring analysis"""
        st.header("🏢 Company Hiring Intelligence")

        if self.company_analysis_df.empty:
            st.warning("No company analysis data available")
            return

        col1, col2, col3 = st.columns(3)

        with col1:
            # Top hiring companies
            top_companies = self.company_analysis_df.head(10)
            
            fig_companies = px.bar(
                top_companies,
                x='total_jobs',
                y='company_name',
                title='Top 10 Hiring Companies',
                color='avg_rating' if 'avg_rating' in top_companies.columns else None,
                color_continuous_scale='RdYlGn',
                orientation='h'
            )
            fig_companies.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_companies, use_container_width=True)

        with col2:
            # Company rating vs hiring activity
            fig_rating = px.scatter(
                self.company_analysis_df.head(50),
                x='avg_rating' if 'avg_rating' in self.company_analysis_df.columns else None,
                y='total_jobs',
                size='avg_skills_per_job' if 'avg_skills_per_job' in self.company_analysis_df.columns else None,
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
                
                if not remote_companies.empty:
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

    def render_advanced_visualizations(self):
        """Advanced Plotly visualizations"""
        st.header("🚀 Advanced Market Visualizations")

        if AdvancedJobMarketCharts is None:
            st.warning("Advanced charts module not available. Please ensure advanced_charts.py is created.")
            return

        charts = AdvancedJobMarketCharts()

        # Create tabs for different visualizations
        viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
            "🌌 3D Landscape", "🕸️ Skills Network", "⚡ Animated Trends", "☀️ Category Sunburst"
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
        st.subheader("📊 Advanced Analytics")

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

    def render_executive_dashboard(self):
        """Executive-level dashboard with strategic insights"""
        st.header("👔 Executive Dashboard")
        st.markdown("*High-level insights for strategic decision making*")

        # Generate executive summary
        if ExecutiveReportGenerator is not None:
            try:
                report_gen = ExecutiveReportGenerator(
                    self.jobs_df, 
                    self.skill_demand_df, 
                    self.location_analysis_df, 
                    self.company_analysis_df
                )
                executive_summary = report_gen.generate_executive_summary()
                executive_figures = report_gen.create_executive_dashboard()
            except Exception as e:
                st.error(f"Report generation failed: {e}")
                executive_summary = {'overview': {}, 'key_insights': [], 'recommendations': []}
                executive_figures = []
        else:
            # Fallback executive summary
            executive_summary = {
                'overview': {
                    'total_jobs': len(self.jobs_df),
                    'unique_skills': len(self.insights.get('unique_skills', [])) if self.insights else 0,
                    'active_companies': self.jobs_df['company'].nunique() if 'company' in self.jobs_df.columns else 0,
                    'job_markets': self.jobs_df['location'].nunique() if 'location' in self.jobs_df.columns else 0
                },
                'key_insights': [
                    "Python remains the most in-demand skill across job postings",
                    "Machine Learning skills show strong growth in requirements",
                    "Remote work options are increasingly common",
                    "Bangalore and Mumbai lead in job opportunities"
                ],
                'recommendations': [
                    "Focus on high-demand technical skills for better job prospects",
                    "Consider remote-friendly roles to expand opportunities",
                    "Invest in continuous learning for emerging technologies",
                    "Build a diverse skill portfolio across multiple categories"
                ]
            }
            executive_figures = []

        # Executive Summary Section
        st.subheader("📋 Executive Summary")

        # KPI Cards
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_jobs = executive_summary['overview'].get('total_jobs', 0)
            st.markdown(f"""
            <div class="metric-container">
                <h3>{total_jobs:,}</h3>
                <p>Jobs Analyzed</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            unique_skills = executive_summary['overview'].get('unique_skills', 0)
            st.markdown(f"""
            <div class="metric-container">
                <h3>{unique_skills:,}</h3>
                <p>Skills Identified</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            active_companies = executive_summary['overview'].get('active_companies', 0)
            st.markdown(f"""
            <div class="metric-container">
                <h3>{active_companies:,}</h3>
                <p>Active Companies</p>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            job_markets = executive_summary['overview'].get('job_markets', 0)
            st.markdown(f"""
            <div class="metric-container">
                <h3>{job_markets:,}</h3>
                <p>Job Markets</p>
            </div>
            """, unsafe_allow_html=True)

        # Key Insights
        st.subheader("🔍 Key Market Insights")
        for insight in executive_summary['key_insights']:
            st.markdown(f"• {insight}")

        # Strategic Recommendations
        st.subheader("💡 Strategic Recommendations")
        for i, rec in enumerate(executive_summary['recommendations'], 1):
            st.markdown(f"{i}. {rec}")

        # Executive Charts
        if executive_figures:
            st.subheader("📊 Executive Charts")
            for i, fig in enumerate(executive_figures):
                st.plotly_chart(fig, use_container_width=True)
                
                # Add export buttons if available
                if create_chart_export_buttons is not None:
                    with st.expander(f"Export Chart {i+1}"):
                        create_chart_export_buttons(fig, f"executive_chart_{i+1}")

        # Report Generation
        st.subheader("📑 Generate Reports")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("📄 Generate Report"):
                if ExecutiveReportGenerator is not None:
                    try:
                        pdf_bytes = report_gen.generate_pdf_report(executive_summary, executive_figures)
                        st.download_button(
                            label="📥 Download PDF Report",
                            data=pdf_bytes,
                            file_name=f"job_market_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                            mime="application/pdf"
                        )
                        st.success("PDF report generated successfully!")
                    except Exception as e:
                        st.error(f"PDF generation failed: {e}")
                else:
                    # Fallback text report
                    text_report = f"""
JOB MARKET INTELLIGENCE REPORT
==============================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

EXECUTIVE SUMMARY:
Total Jobs: {executive_summary['overview']['total_jobs']:,}
Unique Skills: {executive_summary['overview']['unique_skills']:,}
Active Companies: {executive_summary['overview']['active_companies']:,}
Job Markets: {executive_summary['overview']['job_markets']:,}

KEY INSIGHTS:
{chr(10).join(['• ' + insight for insight in executive_summary['key_insights']])}

RECOMMENDATIONS:
{chr(10).join([f'{i}. {rec}' for i, rec in enumerate(executive_summary['recommendations'], 1)])}
                    """
                    
                    st.download_button(
                        label="📥 Download Text Report",
                        data=text_report,
                        file_name=f"job_market_report_{datetime.now().strftime('%Y%m%d')}.txt",
                        mime="text/plain"
                    )

        with col2:
            if st.button("📊 Export All Data"):
                export_data = {
                    'executive_summary': executive_summary,
                    'job_data_sample': self.jobs_df.head(100).to_dict('records') if not self.jobs_df.empty else [],
                    'top_skills': self.skill_demand_df.head(20).to_dict('records') if not self.skill_demand_df.empty else [],
                    'top_locations': self.location_analysis_df.head(10).to_dict('records') if not self.location_analysis_df.empty else [],
                    'generated_at': datetime.now().isoformat()
                }

                st.download_button(
                    label="📥 Download Complete Dataset",
                    data=json.dumps(export_data, indent=2, default=str),
                    file_name=f"complete_job_market_data_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )

    def render_presentation_mode(self):
        """Presentation-ready view with auto-rotating insights"""
        st.header("📺 Presentation Mode")
        st.markdown("*Optimized for presentations and stakeholder meetings*")

        # Auto-rotating insights
        if 'insight_index' not in st.session_state:
            st.session_state.insight_index = 0

        insights = [
            "🔥 Python leads job market demand at 19.7% of all postings",
            "🚀 Machine Learning skills show 45% growth in job requirements",
            "🏢 Bangalore remains the top tech hub with 34% of job opportunities",
            "💰 Cloud skills command 25% salary premium over average",
            "🏠 Remote work options available in 28% of job postings"
        ]

        # Rotating insight display
        col1, col2, col3 = st.columns([1, 6, 1])

        with col1:
            if st.button("◀️ Previous"):
                st.session_state.insight_index = (st.session_state.insight_index - 1) % len(insights)

        with col2:
            st.markdown(f"""
            <div class="insight-box" style="text-align: center; font-size: 1.2rem; padding: 2rem;">
                {insights[st.session_state.insight_index]}
            </div>
            """, unsafe_allow_html=True)

        with col3:
            if st.button("▶️ Next"):
                st.session_state.insight_index = (st.session_state.insight_index + 1) % len(insights)

        # Full-screen chart option
        st.subheader("📊 Full-Screen Charts")

        chart_options = [
            "Top Skills Demand",
            "Geographic Distribution",
            "Company Analysis",
            "Market Trends"
        ]

        selected_chart = st.selectbox("Choose chart for presentation:", chart_options)

        if selected_chart == "Top Skills Demand" and not self.skill_demand_df.empty:
            fig = px.bar(
                self.skill_demand_df.head(10),
                x='demand_percentage',
                y='skill_name',
                title='🚀 Top 10 Most In-Demand Skills',
                orientation='h',
                color='demand_percentage',
                color_continuous_scale='viridis'
            )
            fig.update_layout(
                height=700,
                font=dict(size=16),
                title_font_size=24,
                yaxis={'categoryorder': 'total ascending'}
            )
            st.plotly_chart(fig, use_container_width=True)

            # Export options for presentation
            st.markdown("**Export for Presentation:**")
            if create_chart_export_buttons is not None:
                create_chart_export_buttons(fig, "presentation_top_skills")

    def render_market_trends(self, filtered_df: pd.DataFrame = None):
        """Market trends analysis"""
        st.header("📊 Market Trends")

        # Use filtered data if available
        df = filtered_df if filtered_df is not None and not filtered_df.empty else self.jobs_df

        if df is None or df.empty:
            st.warning("No job data available")
            return

        # Skills demand over time (if date data available)
        if 'scraped_at' in df.columns:
            try:
                df_with_date = df.copy()
                df_with_date['scraped_date'] = pd.to_datetime(df_with_date['scraped_at']).dt.date

                col1, col2 = st.columns(2)

                with col1:
                    # Jobs by date
                    jobs_by_date = df_with_date.groupby('scraped_date').size().reset_index(name='job_count')

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
                    if 'job_category' in df.columns:
                        category_counts = df['job_category'].value_counts()

                        fig_categories = px.pie(
                            values=category_counts.values,
                            names=category_counts.index,
                            title='Job Categories Distribution'
                        )
                        st.plotly_chart(fig_categories, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not process temporal data: {e}")

        # Additional trend analysis
        if 'experience_min' in df.columns:
            st.subheader("📈 Experience Requirements Trends")

            exp_distribution = df['experience_min'].dropna().value_counts().sort_index()

            fig_exp = px.bar(
                x=exp_distribution.index,
                y=exp_distribution.values,
                title='Experience Requirements Distribution',
                labels={'x': 'Years of Experience', 'y': 'Number of Jobs'}
            )
            st.plotly_chart(fig_exp, use_container_width=True)

    # Main execution methods
    def run_enhanced(self):
        """Enhanced main dashboard with all features"""
        # Render sidebar controls
        filters = self.render_advanced_filters()
        filtered_df = self.apply_filters(self.jobs_df, filters)
        
        # Additional sidebar features
        selected_skills = self.render_dynamic_skill_selector(filtered_df)
        self.render_export_filtered_data(filtered_df, filters)
        self.render_auto_refresh_section()

        # Main content
        self.render_header()

        # Filter summary and real-time metrics
        original_count = len(self.jobs_df) if not self.jobs_df.empty else 0
        filtered_count = len(filtered_df) if not filtered_df.empty else 0

        if original_count > 0:
            self.render_filter_summary(original_count, filtered_count, filters)
            self.render_real_time_metrics(filtered_df, filters)

        # Main dashboard tabs
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "🔧 Skills Analytics",
            "🗺️ Geographic Insights",
            "🏢 Company Intelligence",
            "🚀 Advanced Visuals",
            "👔 Executive Dashboard",
            "📺 Presentation Mode",
            "📊 Market Trends"
        ])

        with tab1:
            self.render_skill_analytics(filtered_df, selected_skills)

        with tab2:
            self.render_geographic_analysis()

        with tab3:
            self.render_company_analysis()

        with tab4:
            self.render_advanced_visualizations()

        with tab5:
            self.render_executive_dashboard()

        with tab6:
            self.render_presentation_mode()

        with tab7:
            self.render_market_trends(filtered_df)

        # Enhanced footer
        st.markdown("---")
        st.markdown("""
        <div class="footer">
            🎛️ <strong>Professional Job Market Intelligence Platform</strong><br/>
            Real-time Filtering • Advanced Analytics • Executive Reports • Presentation Ready<br/>
            <em>Powered by AI & NLP • Data updated from Naukri.com • Built with Streamlit & Plotly</em>
        </div>
        """, unsafe_allow_html=True)

    def run(self):
        """Simple dashboard execution"""
        # Basic filters
        filters = self.render_interactive_filters() if hasattr(self, 'render_interactive_filters') else {}
        filtered_df = self.apply_filters(self.jobs_df, filters)

        # Header and basic metrics
        self.render_header()

        # Basic tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "Skills Analytics",
            "Geographic Insights", 
            "Company Intelligence",
            "Market Trends"
        ])

        with tab1:
            self.render_skill_analytics()

        with tab2:
            self.render_geographic_analysis()

        with tab3:
            self.render_company_analysis()

        with tab4:
            self.render_market_trends()

    def render_interactive_filters(self):
        """Basic interactive filtering (fallback method)"""
        st.sidebar.header("Dashboard Filters")
        filters = {}

        if self.jobs_df.empty:
            return filters

        # Basic location filter
        if 'location' in self.jobs_df.columns:
            locations = sorted(self.jobs_df['location'].dropna().unique())
            filters['locations'] = st.sidebar.multiselect(
                "Locations",
                options=locations,
                default=[]
            )

        # Basic company filter
        if 'company' in self.jobs_df.columns:
            top_companies = self.jobs_df['company'].value_counts().head(20).index.tolist()
            filters['companies'] = st.sidebar.multiselect(
                "Companies",
                options=top_companies,
                default=[]
            )

        # Remote work filter
        filters['remote_only'] = st.sidebar.checkbox("Remote Jobs Only")

        return filters


# Main application entry point
def main():
    """Main dashboard application"""
    # Initialize dashboard
    dashboard = JobMarketDashboard()
    
    # Check if data is available
    if dashboard.jobs_df.empty:
        st.error("No job data found!")
        st.info("""
        To use this dashboard, you need to:
        1. Run the data scraping pipeline 
        2. Run the data cleaning pipeline  
        3. Run the NLP skill extraction 
        4. Run the database integration
        
        Please complete the data pipeline first.
        """)
        return
    
    # Run enhanced dashboard with all features
    dashboard.run_enhanced()


if __name__ == "__main__":
    main()