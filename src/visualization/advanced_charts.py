import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import json
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import *

class AdvancedJobMarketCharts:
    """Advanced Plotly visualizations for job market data"""
    
    def __init__(self):
        self.color_palette = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'warning': '#d62728',
            'info': '#9467bd',
            'accent': '#8c564b'
        }
        
        self.skill_colors = {
            'programming_languages': '#FF6B6B',
            'machine_learning_ai': '#4ECDC4', 
            'data_science_analytics': '#45B7D1',
            'cloud_platforms': '#96CEB4',
            'web_frameworks': '#FFEAA7',
            'databases': '#DDA0DD',
            'devops_tools': '#98D8C8',
            'soft_skills': '#F7DC6F'
        }
    
    def create_3d_skill_landscape(self, skill_df: pd.DataFrame) -> go.Figure:
        """Create 3D scatter plot of skills landscape"""
        if skill_df.empty or len(skill_df) < 10:
            return go.Figure().add_annotation(text="Insufficient data for 3D visualization")
        
        # Prepare data for 3D visualization
        skill_df = skill_df.head(50)  # Top 50 skills
        
        fig = go.Figure(data=[go.Scatter3d(
            x=skill_df['job_count'],
            y=skill_df['company_count'], 
            z=skill_df['demand_percentage'],
            mode='markers+text',
            marker=dict(
                size=skill_df['job_count'] / skill_df['job_count'].max() * 30 + 5,
                color=skill_df['demand_percentage'],
                colorscale='viridis',
                opacity=0.8,
                colorbar=dict(title="Demand %")
            ),
            text=skill_df['skill_name'],
            textposition="middle center",
            hovertemplate='<b>%{text}</b><br>' +
                         'Jobs: %{x}<br>' +
                         'Companies: %{y}<br>' +
                         'Demand: %{z}%<br>' +
                         '<extra></extra>'
        )])
        
        fig.update_layout(
            title={
                'text': '🚀 3D Skills Market Landscape',
                'x': 0.5,
                'font': {'size': 20}
            },
            scene=dict(
                xaxis_title='Number of Jobs',
                yaxis_title='Number of Companies',
                zaxis_title='Market Demand (%)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            height=600,
            margin=dict(l=0, r=0, b=0, t=40)
        )
        
        return fig
    
    def create_skill_network_graph(self, jobs_df: pd.DataFrame) -> go.Figure:
        """Create network graph showing skill co-occurrence"""
        if jobs_df.empty:
            return go.Figure().add_annotation(text="No data available for network visualization")
        
        # Extract skill co-occurrence data
        skill_pairs = {}
        
        for idx, row in jobs_df.iterrows():
            skills = row.get('skills_all', [])
            if isinstance(skills, str):
                try:
                    skills = json.loads(skills) if skills.startswith('[') else []
                except:
                    skills = []
            
            if isinstance(skills, list) and len(skills) > 1:
                for i, skill1 in enumerate(skills):
                    for skill2 in skills[i+1:]:
                        pair = tuple(sorted([skill1.lower(), skill2.lower()]))
                        skill_pairs[pair] = skill_pairs.get(pair, 0) + 1
        
        # Get top skill pairs
        top_pairs = sorted(skill_pairs.items(), key=lambda x: x[1], reverse=True)[:30]
        
        if not top_pairs:
            return go.Figure().add_annotation(text="No skill co-occurrence data found")
        
        # Create network data
        nodes = set()
        edges = []
        
        for (skill1, skill2), count in top_pairs:
            nodes.add(skill1)
            nodes.add(skill2)
            edges.append((skill1, skill2, count))
        
        nodes = list(nodes)
        
        # Position nodes in circle
        import math
        positions = {}
        n = len(nodes)
        for i, node in enumerate(nodes):
            angle = 2 * math.pi * i / n
            positions[node] = (math.cos(angle), math.sin(angle))
        
        # Create edge traces
        edge_traces = []
        for skill1, skill2, count in edges:
            x0, y0 = positions[skill1]
            x1, y1 = positions[skill2]
            
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=count/5, color='lightgray'),
                hoverinfo='none',
                showlegend=False
            )
            edge_traces.append(edge_trace)
        
        # Create node trace
        node_trace = go.Scatter(
            x=[positions[node][0] for node in nodes],
            y=[positions[node][1] for node in nodes],
            mode='markers+text',
            marker=dict(
                size=20,
                color=list(range(len(nodes))),
                colorscale='viridis',
                line=dict(width=2, color='white')
            ),
            text=nodes,
            textposition="middle center",
            hovertemplate='<b>%{text}</b><extra></extra>',
            showlegend=False
        )
        
        # Combine traces
        fig = go.Figure(data=edge_traces + [node_trace])
        
        fig.update_layout(
            title={
                'text': '🕸️ Skills Co-occurrence Network',
                'x': 0.5,
                'font': {'size': 20}
            },
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[
                dict(
                    text="Skills that frequently appear together in job postings",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    font=dict(size=12),
                    align="left"
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600
        )
        
        return fig
    
    def create_animated_skill_demand(self, jobs_df: pd.DataFrame) -> go.Figure:
        """Create animated chart showing skill demand evolution"""
        if jobs_df.empty or 'scraped_at' not in jobs_df.columns:
            return go.Figure().add_annotation(text="No temporal data available for animation")
        
        # Prepare temporal data
        jobs_df['date'] = pd.to_datetime(jobs_df['scraped_at']).dt.date
        
        # Get top skills
        all_skills = []
        for idx, row in jobs_df.iterrows():
            skills = row.get('skills_all', [])
            if isinstance(skills, str):
                try:
                    skills = json.loads(skills) if skills.startswith('[') else []
                except:
                    skills = []
            if isinstance(skills, list):
                all_skills.extend([s.lower() for s in skills])
        
        top_skills = pd.Series(all_skills).value_counts().head(10).index.tolist()
        
        # Create temporal skill data
        dates = sorted(jobs_df['date'].unique())
        skill_timeline = []
        
        for date in dates:
            date_jobs = jobs_df[jobs_df['date'] == date]
            date_skills = []
            
            for idx, row in date_jobs.iterrows():
                skills = row.get('skills_all', [])
                if isinstance(skills, str):
                    try:
                        skills = json.loads(skills) if skills.startswith('[') else []
                    except:
                        skills = []
                if isinstance(skills, list):
                    date_skills.extend([s.lower() for s in skills])
            
            skill_counts = pd.Series(date_skills).value_counts()
            
            for skill in top_skills:
                skill_timeline.append({
                    'date': date,
                    'skill': skill,
                    'count': skill_counts.get(skill, 0),
                    'percentage': (skill_counts.get(skill, 0) / len(date_jobs) * 100) if len(date_jobs) > 0 else 0
                })
        
        timeline_df = pd.DataFrame(skill_timeline)
        
        if timeline_df.empty:
            return go.Figure().add_annotation(text="Insufficient temporal data")
        
        # Create animated bar chart
        fig = px.bar(
            timeline_df,
            x='skill',
            y='count',
            animation_frame='date',
            color='skill',
            title='📈 Skill Demand Evolution Over Time',
            labels={'count': 'Number of Mentions', 'skill': 'Skills'},
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig.update_layout(
            xaxis_tickangle=-45,
            height=500,
            showlegend=False
        )
        
        # Update animation settings
        fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 1000
        fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 500
        
        return fig
    
    def create_skill_category_sunburst(self, jobs_df: pd.DataFrame) -> go.Figure:
        """Create sunburst chart for skill categories"""
        if jobs_df.empty:
            return go.Figure().add_annotation(text="No data available for sunburst chart")
        
        # Collect categorized skills data
        category_data = {}
        
        for idx, row in jobs_df.iterrows():
            categorized = row.get('skills_categorized', {})
            if isinstance(categorized, str):
                try:
                    categorized = json.loads(categorized) if categorized.startswith('{') else {}
                except:
                    categorized = {}
            
            if isinstance(categorized, dict):
                for category, skills in categorized.items():
                    if isinstance(skills, list):
                        if category not in category_data:
                            category_data[category] = {}
                        for skill in skills:
                            skill_lower = skill.lower()
                            category_data[category][skill_lower] = category_data[category].get(skill_lower, 0) + 1
        
        # Prepare sunburst data
        ids = ['Skills']
        labels = ['All Skills']
        parents = ['']
        values = [sum(sum(skills.values()) for skills in category_data.values())]
        colors = ['#FFFFFF']
        
        # Add categories
        for category, skills in category_data.items():
            category_total = sum(skills.values())
            category_id = f"Skills/{category}"
            
            ids.append(category_id)
            labels.append(category.replace('_', ' ').title())
            parents.append('Skills')
            values.append(category_total)
            colors.append(self.skill_colors.get(category, '#CCCCCC'))
            
            # Add top skills in each category
            top_category_skills = sorted(skills.items(), key=lambda x: x[1], reverse=True)[:5]
            for skill, count in top_category_skills:
                skill_id = f"{category_id}/{skill}"
                
                ids.append(skill_id)
                labels.append(skill.title())
                parents.append(category_id)
                values.append(count)
                colors.append('#E8E8E8')
        
        fig = go.Figure(go.Sunburst(
            ids=ids,
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            marker=dict(
                colors=colors,
                line=dict(color='white', width=2)
            ),
            hovertemplate='<b>%{label}</b><br>Jobs: %{value}<br><extra></extra>',
            maxdepth=3
        ))
        
        fig.update_layout(
            title={
                'text': '☀️ Skills Category Hierarchy',
                'x': 0.5,
                'font': {'size': 20}
            },
            height=600,
            margin=dict(t=50, l=0, r=0, b=0)
        )
        
        return fig
    
    def create_salary_vs_skills_heatmap(self, jobs_df: pd.DataFrame) -> go.Figure:
        """Create heatmap showing salary correlation with skills"""
        if jobs_df.empty:
            return go.Figure().add_annotation(text="No data available for salary analysis")
        
        # Extract salary and skills data
        salary_skill_data = []
        
        for idx, row in jobs_df.iterrows():
            # Parse salary if available
            salary_str = str(row.get('salary', ''))
            salary_value = None
            
            # Simple salary extraction (extend this as needed)
            import re
            salary_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:lakh|lac)', salary_str.lower())
            if salary_match:
                salary_value = float(salary_match.group(1))
            
            if salary_value:
                skills = row.get('skills_all', [])
                if isinstance(skills, str):
                    try:
                        skills = json.loads(skills) if skills.startswith('[') else []
                    except:
                        skills = []
                
                if isinstance(skills, list):
                    for skill in skills:
                        salary_skill_data.append({
                            'skill': skill.lower(),
                            'salary': salary_value
                        })
        
        if not salary_skill_data:
            return go.Figure().add_annotation(text="No salary data available for analysis")
        
        salary_df = pd.DataFrame(salary_skill_data)
        
        # Get top skills and salary ranges
        top_skills = salary_df['skill'].value_counts().head(20).index.tolist()
        salary_df_filtered = salary_df[salary_df['skill'].isin(top_skills)]
        
        # Create salary ranges
        salary_df_filtered['salary_range'] = pd.cut(
            salary_df_filtered['salary'],
            bins=[0, 5, 10, 15, 20, 50],
            labels=['0-5L', '5-10L', '10-15L', '15-20L', '20L+']
        )
        
        # Create pivot table for heatmap
        heatmap_data = salary_df_filtered.groupby(['skill', 'salary_range']).size().unstack(fill_value=0)
        
        if heatmap_data.empty:
            return go.Figure().add_annotation(text="Insufficient data for salary heatmap")
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale='viridis',
            hoverongaps=False,
            hovertemplate='<b>%{y}</b><br>Salary Range: %{x}<br>Jobs: %{z}<extra></extra>'
        ))
        
        fig.update_layout(
            title={
                'text': '💰 Skills vs Salary Distribution Heatmap',
                'x': 0.5,
                'font': {'size': 20}
            },
            xaxis_title='Salary Range (Lakhs)',
            yaxis_title='Skills',
            height=600,
            margin=dict(l=150)
        )
        
        return fig
    
    def create_company_skill_matrix(self, jobs_df: pd.DataFrame) -> go.Figure:
        """Create matrix showing which companies prefer which skills"""
        if jobs_df.empty:
            return go.Figure().add_annotation(text="No data available for company-skill matrix")
        
        # Get top companies and skills
        top_companies = jobs_df['company'].value_counts().head(15).index.tolist()
        
        # Collect all skills
        all_skills = []
        for idx, row in jobs_df.iterrows():
            skills = row.get('skills_all', [])
            if isinstance(skills, str):
                try:
                    skills = json.loads(skills) if skills.startswith('[') else []
                except:
                    skills = []
            if isinstance(skills, list):
                all_skills.extend([s.lower() for s in skills])
        
        top_skills = pd.Series(all_skills).value_counts().head(20).index.tolist()
        
        # Create company-skill matrix
        matrix_data = []
        
        for company in top_companies:
            company_jobs = jobs_df[jobs_df['company'] == company]
            company_skills = []
            
            for idx, row in company_jobs.iterrows():
                skills = row.get('skills_all', [])
                if isinstance(skills, str):
                    try:
                        skills = json.loads(skills) if skills.startswith('[') else []
                    except:
                        skills = []
                if isinstance(skills, list):
                    company_skills.extend([s.lower() for s in skills])
            
            company_skill_counts = pd.Series(company_skills).value_counts()
            
            row_data = []
            for skill in top_skills:
                count = company_skill_counts.get(skill, 0)
                # Normalize by company's total jobs
                normalized = count / len(company_jobs) if len(company_jobs) > 0 else 0
                row_data.append(normalized)
            
            matrix_data.append(row_data)
        
        fig = go.Figure(data=go.Heatmap(
            z=matrix_data,
            x=top_skills,
            y=top_companies,
            colorscale='Blues',
            hoverongaps=False,
            hovertemplate='<b>%{y}</b><br>Skill: %{x}<br>Demand: %{z:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title={
                'text': '🏢 Company-Skill Preference Matrix',
                'x': 0.5,
                'font': {'size': 20}
            },
            xaxis_title='Skills',
            yaxis_title='Companies',
            height=600,
            margin=dict(l=200, b=150),
            xaxis_tickangle=-45
        )
        
        return fig
    
    def create_skill_rarity_vs_value_chart(self, skill_demand_df: pd.DataFrame) -> go.Figure:
        """Create chart showing skill rarity vs market value"""
        if skill_demand_df.empty:
            return go.Figure().add_annotation(text="No skill demand data available")
        
        # Calculate skill rarity (inverse of job count)
        skill_demand_df = skill_demand_df.copy()
        skill_demand_df['rarity_score'] = 1 / (skill_demand_df['job_count'] + 1)  # +1 to avoid division by zero
        
        # Use average company rating as proxy for "value"
        skill_demand_df['value_score'] = skill_demand_df.get('avg_company_rating', 3.0)
        
        fig = px.scatter(
            skill_demand_df.head(50),
            x='rarity_score',
            y='value_score',
            size='demand_percentage',
            color='skill_category',
            hover_name='skill_name',
            title='💎 Skill Rarity vs Market Value Analysis',
            labels={
                'rarity_score': 'Skill Rarity (Higher = More Rare)',
                'value_score': 'Market Value (Company Rating)',
                'demand_percentage': 'Demand %'
            },
            color_discrete_map=self.skill_colors
        )
        
        # Add quadrant lines
        median_rarity = skill_demand_df['rarity_score'].median()
        median_value = skill_demand_df['value_score'].median()
        
        fig.add_hline(y=median_value, line_dash="dash", annotation_text="Median Value")
        fig.add_vline(x=median_rarity, line_dash="dash", annotation_text="Median Rarity")
        
        # Add quadrant annotations
        fig.add_annotation(x=median_rarity * 1.5, y=median_value * 1.1, text="💎 Rare & Valuable", showarrow=False)
        fig.add_annotation(x=median_rarity * 0.5, y=median_value * 1.1, text="🔥 Common & Valuable", showarrow=False)
        fig.add_annotation(x=median_rarity * 1.5, y=median_value * 0.9, text="🎯 Niche Market", showarrow=False)
        fig.add_annotation(x=median_rarity * 0.5, y=median_value * 0.9, text="📚 Basic Skills", showarrow=False)
        
        fig.update_layout(height=600)
        
        return fig

# Usage example and testing
def main():
    """Test the advanced charts"""
    charts = AdvancedJobMarketCharts()
    
    # Load sample data for testing
    from pathlib import Path
    processed_dir = Path(__file__).parent.parent.parent / "data" / "processed"
    
    # Load skill demand data
    skill_files = list(processed_dir.glob("skill_demand_analysis_*.csv"))
    if skill_files:
        latest_skill_file = max(skill_files, key=lambda x: x.stat().st_mtime)
        skill_df = pd.read_csv(latest_skill_file)
        
        # Create 3D landscape
        fig_3d = charts.create_3d_skill_landscape(skill_df)
        fig_3d.show()
        
        print("Advanced charts library ready!")
    else:
        print(" No skill demand data found. Run database integration first.")

if __name__ == "__main__":
    main()