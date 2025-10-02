# OpenAI Integration for Astro-AI
# 
# This module provides AI-powered scientific insights and report generation
# for astronomical analysis results

import openai
import streamlit as st
import json
from typing import Dict, Any, List, Optional

class OpenAIAssistant:
    """
    AI-powered assistant for astronomical data analysis and scientific reporting.
    
    Provides natural language insights, scientific interpretation, and 
    automated report generation for galaxy evolution studies.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize OpenAI assistant with API key.
        
        Parameters:
        -----------
        api_key : str, optional
            OpenAI API key. If not provided, will look for it in environment
            or Streamlit secrets.
        """
        if api_key:
            self.api_key = api_key
        elif 'OPENAI_API_KEY' in st.secrets:
            self.api_key = st.secrets['OPENAI_API_KEY']
        else:
            # Use the provided API key
            self.api_key = "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"
        
        # Initialize OpenAI client
        openai.api_key = self.api_key
        
        # System prompt for astronomical context
        self.system_prompt = """
        You are an expert astrophysicist and data scientist specializing in galaxy evolution, 
        21cm cosmology, and observational astronomy. Your role is to provide scientific 
        insights, interpret results, and generate comprehensive reports for astronomical 
        data analysis.
        
        Key areas of expertise:
        - Galaxy formation and evolution
        - 21cm intensity mapping and reionization
        - Galaxy cluster environments and quenching
        - JWST observations and spectroscopy
        - Stellar population synthesis and SED fitting
        - Statistical analysis of astronomical data
        
        Always provide scientifically accurate, well-referenced explanations that would 
        be appropriate for research publications or technical reports.
        """
    
    def generate_insight(self, data_summary: Dict[str, Any], 
                        analysis_type: str = "general") -> str:
        """
        Generate AI-powered scientific insights from analysis results.
        
        Parameters:
        -----------
        data_summary : dict
            Summary of analysis results including key metrics and findings
        analysis_type : str
            Type of analysis ('cosmic_evolution', 'cluster_analysis', 'jwst_spectroscopy')
        
        Returns:
        --------
        str
            AI-generated scientific insights and interpretation
        """
        try:
            # Create analysis-specific prompts
            if analysis_type == "cosmic_evolution":
                context = f"""
                Analyze these 21cm cosmological simulation results:
                {json.dumps(data_summary, indent=2)}
                
                Provide scientific interpretation focusing on:
                1. Reionization physics and timing
                2. Power spectrum implications
                3. Brightness temperature evolution
                4. Connection to galaxy formation
                """
            
            elif analysis_type == "cluster_analysis":
                context = f"""
                Analyze these galaxy cluster environment results:
                {json.dumps(data_summary, indent=2)}
                
                Provide scientific interpretation focusing on:
                1. Environmental quenching mechanisms
                2. Stellar mass assembly differences
                3. Red fraction evolution
                4. Comparison with observations
                """
            
            elif analysis_type == "jwst_spectroscopy":
                context = f"""
                Analyze these JWST spectroscopic results:
                {json.dumps(data_summary, indent=2)}
                
                Provide scientific interpretation focusing on:
                1. Spectral line diagnostics
                2. Stellar population properties
                3. Galaxy formation history
                4. Comparison with previous surveys
                """
            
            else:
                context = f"""
                Analyze these astronomical analysis results:
                {json.dumps(data_summary, indent=2)}
                
                Provide general scientific interpretation and key findings.
                """
            
            # Generate response using OpenAI
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": context}
                ],
                max_tokens=1500,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"AI analysis temporarily unavailable: {str(e)}"
    
    def generate_comparative_analysis(self, results_summary: Dict[str, Any]) -> str:
        """
        Generate comparative analysis across multiple modules.
        
        Parameters:
        -----------
        results_summary : dict
            Combined results from multiple analysis modules
        
        Returns:
        --------
        str
            AI-generated comparative analysis and synthesis
        """
        try:
            context = f"""
            Provide a comprehensive comparative analysis across these multi-wavelength 
            and multi-epoch astronomical results:
            
            {json.dumps(results_summary, indent=2)}
            
            Focus on:
            1. Connecting 21cm cosmic evolution with galaxy observations
            2. Environmental effects on galaxy evolution
            3. JWST insights into early galaxy formation
            4. Synthesis of results across cosmic time
            5. Implications for galaxy formation models
            6. Future observational priorities
            
            Structure as a scientific synthesis suitable for a research summary.
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": context}
                ],
                max_tokens=2000,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Comparative analysis temporarily unavailable: {str(e)}"
    
    def generate_report_section(self, section_type: str, 
                               data: Dict[str, Any]) -> str:
        """
        Generate specific report sections (Introduction, Methods, Results, Discussion).
        
        Parameters:
        -----------
        section_type : str
            Type of section ('introduction', 'methods', 'results', 'discussion')
        data : dict
            Relevant data for the section
        
        Returns:
        --------
        str
            AI-generated report section content
        """
        try:
            if section_type == "introduction":
                prompt = f"""
                Write a scientific introduction for a galaxy evolution study that includes:
                {json.dumps(data, indent=2)}
                
                Include relevant background, motivation, and objectives.
                """
            
            elif section_type == "methods":
                prompt = f"""
                Write a methods section describing the analysis techniques used:
                {json.dumps(data, indent=2)}
                
                Include technical details appropriate for peer review.
                """
            
            elif section_type == "results":
                prompt = f"""
                Write a results section summarizing key findings:
                {json.dumps(data, indent=2)}
                
                Present results objectively with quantitative details.
                """
            
            elif section_type == "discussion":
                prompt = f"""
                Write a discussion section interpreting results and implications:
                {json.dumps(data, indent=2)}
                
                Include scientific interpretation, limitations, and future work.
                """
            
            else:
                prompt = f"Generate {section_type} content for: {json.dumps(data, indent=2)}"
            
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Report section generation temporarily unavailable: {str(e)}"
    
    def suggest_next_steps(self, analysis_results: Dict[str, Any]) -> List[str]:
        """
        Suggest next steps and follow-up analyses based on current results.
        
        Parameters:
        -----------
        analysis_results : dict
            Current analysis results and findings
        
        Returns:
        --------
        list
            List of suggested next steps and analyses
        """
        try:
            context = f"""
            Based on these analysis results, suggest specific next steps for research:
            {json.dumps(analysis_results, indent=2)}
            
            Provide 5-8 concrete, actionable suggestions for follow-up work.
            Format as a numbered list.
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": context}
                ],
                max_tokens=800,
                temperature=0.8
            )
            
            # Parse response into list
            suggestions = response.choices[0].message.content.split('\n')
            suggestions = [s.strip() for s in suggestions if s.strip() and 
                          (s.strip()[0].isdigit() or s.strip().startswith('-'))]
            
            return suggestions[:8]  # Limit to 8 suggestions
            
        except Exception as e:
            return [f"Next steps analysis temporarily unavailable: {str(e)}"]
    
    def check_api_status(self) -> bool:
        """
        Check if OpenAI API is accessible and functioning.
        
        Returns:
        --------
        bool
            True if API is working, False otherwise
        """
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=5
            )
            return True
        except:
            return False