"""OpenAI Integration for Astro-AI

Modernized for openai>=1.0.0.

Key points:
1. Uses the new `OpenAI` client (Responses API optional) while retaining a legacy fallback
2. Removes hard-coded API key. Provide credentials via:
   - Explicit constructor arg `api_key`
   - Streamlit `st.secrets['OPENAI_API_KEY']`
   - Environment variable `OPENAI_API_KEY`
3. Unified private helper `_chat` abstracts differences between SDK versions
4. Optional streaming via `stream_chat` when using the new SDK's chat.completions
5. Toggle Responses API usage with `use_responses_api=True` (experimental for richer multimodal inputs)

Environment setup:
    set OPENAI_API_KEY=sk-...   (Windows PowerShell)

Example:
    assistant = OpenAIAssistant(model="gpt-4o")
    insight = assistant.generate_insight({"stat":"value"}, analysis_type="cosmic_evolution")
    for token in assistant.stream_chat([...]):
        print(token, end="")

Security: ensure you NEVER commit real API keys to source control.
"""

import os
import json
from typing import Dict, Any, List, Optional, Union

import streamlit as st

try:
    # New >=1.0 SDK style
    from openai import OpenAI, AsyncOpenAI, APIError
    _NEW_OPENAI_SDK = True
except Exception:  # pragma: no cover - fallback if old package version
    import openai  # type: ignore
    _NEW_OPENAI_SDK = False

class OpenAIAssistant:
    """
    AI-powered assistant for astronomical data analysis and scientific reporting.
    
    Provides natural language insights, scientific interpretation, and 
    automated report generation for galaxy evolution studies.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o", use_responses_api: bool = False):
        """
        Initialize OpenAI assistant with API key.
        
        Parameters:
        -----------
        api_key : str, optional
            OpenAI API key. If not provided, will look for it in environment
            or Streamlit secrets.
        """
        # API key sourcing hierarchy (no hard-coded secrets):
        # 1. explicit parameter, 2. Streamlit secrets, 3. env var OPENAI_API_KEY
        if api_key:
            self.api_key = api_key
        elif 'OPENAI_API_KEY' in st.secrets:  # type: ignore[attr-defined]
            self.api_key = st.secrets['OPENAI_API_KEY']  # type: ignore[index]
        else:
            self.api_key = os.getenv('OPENAI_API_KEY', '')

        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY env var or Streamlit secret.")

        self.model = model
        self.use_responses_api = use_responses_api

        if _NEW_OPENAI_SDK:
            # Instantiate reusable client
            self.client = OpenAI(api_key=self.api_key)
        else:  # legacy fallback
            import openai  # type: ignore
            openai.api_key = self.api_key
            self.client = openai  # type: ignore
        
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
            return self._chat(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": context},
                ],
                max_tokens=1500,
                temperature=0.7,
            )
            
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
            
            return self._chat(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": context},
                ],
                max_tokens=2000,
                temperature=0.7,
            )
            
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
            
            return self._chat(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1000,
                temperature=0.7,
            )
            
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
            
            content = self._chat(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": context},
                ],
                max_tokens=800,
                temperature=0.8,
            )

            suggestions = content.split('\n')
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
            _ = self._chat(
                messages=[{"role": "user", "content": "Ping"}],
                max_tokens=5,
                temperature=0.0,
            )
            return True
        except Exception:
            return False

    # ------------------------- Internal Helpers ------------------------- #
    def _chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """Unified chat abstraction supporting both new and legacy SDKs.

        Parameters
        ----------
        messages : list
            List of role/content dicts.
        max_tokens : int
            Completion max tokens.
        temperature : float
            Sampling temperature.
        """
        if _NEW_OPENAI_SDK:
            try:
                if self.use_responses_api:
                    # Convert messages into Responses API input format
                    # First system prompt (if present) becomes instructions
                    instructions = None
                    user_inputs: List[Dict[str, Union[str, list]]] = []
                    for m in messages:
                        if m["role"] == "system" and instructions is None:
                            instructions = m["content"]
                        else:
                            user_inputs.append(
                                {
                                    "role": m["role"],
                                    "content": [
                                        {"type": "input_text", "text": m["content"]}
                                    ],
                                }
                            )
                    resp = self.client.responses.create(
                        model=self.model,
                        instructions=instructions,
                        input=user_inputs,
                        max_output_tokens=max_tokens,
                        temperature=temperature,
                    )
                    return getattr(resp, "output_text", "").strip()
                else:
                    completion = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    return completion.choices[0].message.content  # type: ignore
            except APIError as e:  # type: ignore[name-defined]
                raise RuntimeError(f"OpenAI API error: {e}") from e
            except Exception as e:  # pragma: no cover
                raise RuntimeError(f"OpenAI request failed: {e}") from e
        else:  # legacy path
            try:
                completion = self.client.ChatCompletion.create(  # type: ignore[attr-defined]
                    model=self.model if self.model.startswith("gpt-") else "gpt-4",
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return completion.choices[0].message.content  # type: ignore
            except Exception as e:  # pragma: no cover
                raise RuntimeError(f"Legacy OpenAI request failed: {e}") from e

    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
    ):
        """Generator yielding streamed tokens (new SDK only).

        Falls back to non-streaming if streaming unsupported.
        """
        if not _NEW_OPENAI_SDK or self.use_responses_api:
            # Fallback: return single chunk
            yield self._chat(messages, temperature=temperature)
            return
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )
            for event in stream:  # type: ignore
                delta = getattr(event.choices[0].delta, "content", None)  # type: ignore
                if delta:
                    yield delta
        except Exception as e:  # pragma: no cover
            yield f"[Streaming failed: {e}]"