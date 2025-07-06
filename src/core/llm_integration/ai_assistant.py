"""
AI Assistant
============

Handles LLM integration for data analysis and visualization commands.
"""

import openai
import logging
from typing import Dict, Any, Optional, List
import json
import pandas as pd

logger = logging.getLogger(__name__)

class AIAssistant:
    """AI-powered assistant for data analysis and visualization."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the AI assistant.
        
        Args:
            config (Dict[str, Any]): Configuration including LLM settings
        """
        self.config = config
        self.client = None
        self.conversation_history = []
        self.is_initialized = False
        
    def initialize(self) -> bool:
        """
        Initialize OpenAI client.
        
        Returns:
            bool: True if successful
        """
        try:
            llm_config = self.config.get("llm", {})
            
            # Initialize OpenAI client
            openai.api_key = llm_config.get("api_key")
            if llm_config.get("org_id"):
                openai.organization = llm_config.get("org_id")
            
            self.client = openai.OpenAI(api_key=llm_config.get("api_key"))
            self.is_initialized = True
            
            logger.info("AI assistant initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing AI assistant: {str(e)}")
            return False
    
    def process_command(self, command: str, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a natural language command for data analysis.
        
        Args:
            command (str): User command
            dataset_info (Dict[str, Any]): Current dataset information
            
        Returns:
            Dict[str, Any]: Processing result
        """
        if not self.is_initialized:
            return {"error": "AI assistant not initialized"}
        
        try:
            # Create system prompt with dataset context
            system_prompt = self._create_system_prompt(dataset_info)
            
            # Create user message
            user_message = f"User command: {command}"
            
            # Get AI response
            response = self.client.chat.completions.create(
                model=self.config["llm"]["model"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=self.config["llm"]["temperature"],
                max_tokens=self.config["llm"]["max_tokens"]
            )
            
            # Parse response
            ai_response = response.choices[0].message.content
            result = self._parse_ai_response(ai_response)
            
            # Add to conversation history
            self.conversation_history.append({
                "user": command,
                "assistant": ai_response,
                "result": result
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing command: {str(e)}")
            return {"error": f"Failed to process command: {str(e)}"}
    
    def _create_system_prompt(self, dataset_info: Dict[str, Any]) -> str:
        """Create system prompt with dataset context."""
        
        prompt = """You are a data science assistant that helps users analyze and visualize data through natural language commands.

Current Dataset Information:
"""
        
        if dataset_info.get("name"):
            prompt += f"- Dataset Name: {dataset_info['name']}\n"
        
        if dataset_info.get("shape"):
            prompt += f"- Shape: {dataset_info['shape'][0]} rows, {dataset_info['shape'][1]} columns\n"
        
        if dataset_info.get("columns"):
            prompt += f"- Columns: {', '.join(dataset_info['columns'])}\n"
        
        if dataset_info.get("dtypes"):
            prompt += "- Data Types:\n"
            for col, dtype in dataset_info["dtypes"].items():
                prompt += f"  - {col}: {dtype}\n"
        
        prompt += """
Instructions:
1. Analyze the user's command and determine the appropriate action
2. Respond with a JSON object containing:
   - "action": The type of action (e.g., "visualize", "analyze", "filter", "select")
   - "parameters": Specific parameters for the action
   - "explanation": Brief explanation of what will be done
   - "code": Python code to execute (if applicable)

Example response:
{
    "action": "visualize",
    "parameters": {
        "chart_type": "histogram",
        "column": "age",
        "title": "Age Distribution"
    },
    "explanation": "Creating a histogram to show the distribution of ages in the dataset",
    "code": "import plotly.express as px\nfig = px.histogram(df, x='age', title='Age Distribution')\nfig.show()"
}
"""
        
        return prompt
    
    def _parse_ai_response(self, response: str) -> Dict[str, Any]:
        """Parse AI response and extract structured information."""
        
        try:
            # Try to extract JSON from response
            if "{" in response and "}" in response:
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                json_str = response[json_start:json_end]
                
                result = json.loads(json_str)
                return result
            
        except json.JSONDecodeError:
            logger.warning("Could not parse AI response as JSON")
        
        # Fallback: Return simple text response
        return {
            "action": "unknown",
            "parameters": {},
            "explanation": response,
            "code": ""
        }
    
    def generate_visualization_code(self, chart_type: str, dataset_info: Dict[str, Any], 
                                  parameters: Dict[str, Any]) -> str:
        """
        Generate visualization code based on parameters.
        
        Args:
            chart_type (str): Type of chart
            dataset_info (Dict[str, Any]): Dataset information
            parameters (Dict[str, Any]): Chart parameters
            
        Returns:
            str: Python code for visualization
        """
        
        code_templates = {
            "histogram": """
import plotly.express as px
fig = px.histogram(df, x='{column}', title='{title}')
fig.show()
""",
            "scatter": """
import plotly.express as px
fig = px.scatter(df, x='{x_column}', y='{y_column}', title='{title}')
fig.show()
""",
            "bar": """
import plotly.express as px
fig = px.bar(df, x='{x_column}', y='{y_column}', title='{title}')
fig.show()
""",
            "line": """
import plotly.express as px
fig = px.line(df, x='{x_column}', y='{y_column}', title='{title}')
fig.show()
"""
        }
        
        template = code_templates.get(chart_type, "# Chart type not supported")
        return template.format(**parameters)
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history."""
        return self.conversation_history
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        logger.info("Conversation history cleared") 