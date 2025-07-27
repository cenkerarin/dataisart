"""
AI Assistant
============

Enhanced AI-powered assistant for data analysis and visualization commands.
Integrates with voice recognition and provides actionable data analysis capabilities.
"""

import openai
import logging
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import traceback

logger = logging.getLogger(__name__)

class ActionType(Enum):
    """Types of actions the AI assistant can perform."""
    DESCRIBE_DATA = "describe_data"
    VISUALIZE = "visualize"
    FILTER_DATA = "filter_data"
    SORT_DATA = "sort_data"
    ANALYZE_COLUMN = "analyze_column"
    COMPARE_COLUMNS = "compare_columns"
    SUMMARY_STATS = "summary_stats"
    CORRELATION = "correlation"
    EXPORT_DATA = "export_data"
    LOAD_DATA = "load_data"
    UNKNOWN = "unknown"

class VisualizationType(Enum):
    """Types of visualizations available."""
    HISTOGRAM = "histogram"
    BAR_CHART = "bar_chart"
    LINE_CHART = "line_chart"
    SCATTER_PLOT = "scatter_plot"
    PIE_CHART = "pie_chart"
    BOX_PLOT = "box_plot"
    HEATMAP = "heatmap"
    CORRELATION_MATRIX = "correlation_matrix"

@dataclass
class ActionResult:
    """Result of an AI assistant action."""
    success: bool
    action_type: ActionType
    data: Optional[Dict[str, Any]] = None
    message: str = ""
    visualization: Optional[Any] = None  # Plotly figure
    error: Optional[str] = None
    suggestions: List[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for UI consumption."""
        result = {
            "success": self.success,
            "action_type": self.action_type.value,
            "message": self.message,
            "suggestions": self.suggestions or []
        }
        
        if self.data:
            result["data"] = self.data
        if self.error:
            result["error"] = self.error
        if self.visualization:
            # Convert plotly figure to JSON for UI
            result["visualization"] = self.visualization.to_json()
            
        return result

@dataclass
class DatasetContext:
    """Current dataset context for the AI assistant."""
    dataframe: Optional[pd.DataFrame] = None
    name: str = ""
    file_path: str = ""
    shape: tuple = (0, 0)
    columns: List[str] = None
    dtypes: Dict[str, str] = None
    memory_usage: str = ""
    
    def update_from_dataframe(self, df: pd.DataFrame, name: str = "", file_path: str = ""):
        """Update context from a pandas DataFrame."""
        self.dataframe = df
        self.name = name or "Current Dataset"
        self.file_path = file_path
        self.shape = df.shape
        self.columns = df.columns.tolist()
        self.dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
        self.memory_usage = f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for prompt context."""
        return {
            "name": self.name,
            "shape": self.shape,
            "columns": self.columns,
            "dtypes": self.dtypes,
            "memory_usage": self.memory_usage,
            "sample_data": self.dataframe.head(3).to_dict() if self.dataframe is not None else None
        }

class DataAnalysisActions:
    """Handles execution of data analysis actions."""
    
    def __init__(self, dataset_context: DatasetContext):
        self.dataset_context = dataset_context
    
    def describe_data(self, **kwargs) -> ActionResult:
        """Provide a comprehensive description of the dataset."""
        try:
            df = self.dataset_context.dataframe
            if df is None:
                return ActionResult(
                    success=False,
                    action_type=ActionType.DESCRIBE_DATA,
                    error="No dataset loaded"
                )
            
            # Generate comprehensive description
            description = {
                "basic_info": {
                    "name": self.dataset_context.name,
                    "shape": self.dataset_context.shape,
                    "memory_usage": self.dataset_context.memory_usage
                },
                "column_info": {},
                "summary_stats": df.describe().to_dict(),
                "data_types": self.dataset_context.dtypes,
                "missing_values": df.isnull().sum().to_dict(),
                "sample_data": df.head().to_dict()
            }
            
            # Column-specific information
            for col in df.columns:
                col_info = {
                    "type": str(df[col].dtype),
                    "non_null_count": df[col].count(),
                    "null_count": df[col].isnull().sum(),
                    "unique_values": df[col].nunique()
                }
                
                if df[col].dtype in ['object', 'string']:
                    col_info["sample_values"] = df[col].dropna().unique()[:5].tolist()
                elif df[col].dtype in ['int64', 'float64']:
                    col_info["min"] = df[col].min()
                    col_info["max"] = df[col].max()
                    col_info["mean"] = df[col].mean()
                
                description["column_info"][col] = col_info
            
            message = f"Dataset '{self.dataset_context.name}' has {df.shape[0]} rows and {df.shape[1]} columns. "
            message += f"Memory usage: {self.dataset_context.memory_usage}. "
            
            null_columns = [col for col, count in description["missing_values"].items() if count > 0]
            if null_columns:
                message += f"Columns with missing values: {', '.join(null_columns)}."
            
            suggestions = [
                "Try 'visualize column distribution' to see data patterns",
                "Use 'show correlation matrix' to find relationships",
                "Say 'filter data where [condition]' to subset the data"
            ]
            
            return ActionResult(
                success=True,
                action_type=ActionType.DESCRIBE_DATA,
                data=description,
                message=message,
                suggestions=suggestions
            )
            
        except Exception as e:
            logger.error(f"Error describing data: {str(e)}")
            return ActionResult(
                success=False,
                action_type=ActionType.DESCRIBE_DATA,
                error=str(e)
            )
    
    def visualize_data(self, column: str = None, chart_type: str = "histogram", 
                      x_column: str = None, y_column: str = None, **kwargs) -> ActionResult:
        """Create visualizations of the data."""
        try:
            df = self.dataset_context.dataframe
            if df is None:
                return ActionResult(
                    success=False,
                    action_type=ActionType.VISUALIZE,
                    error="No dataset loaded"
                )
            
            fig = None
            message = ""
            
            if chart_type.lower() in ["histogram", "hist"]:
                if not column:
                    # Find first numeric column
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    if len(numeric_cols) == 0:
                        return ActionResult(
                            success=False,
                            action_type=ActionType.VISUALIZE,
                            error="No numeric columns found for histogram"
                        )
                    column = numeric_cols[0]
                
                fig = px.histogram(df, x=column, title=f"Distribution of {column}")
                message = f"Created histogram for {column}"
                
            elif chart_type.lower() in ["scatter", "scatter_plot"]:
                if not x_column or not y_column:
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    if len(numeric_cols) < 2:
                        return ActionResult(
                            success=False,
                            action_type=ActionType.VISUALIZE,
                            error="Need at least 2 numeric columns for scatter plot"
                        )
                    x_column = numeric_cols[0]
                    y_column = numeric_cols[1]
                
                fig = px.scatter(df, x=x_column, y=y_column, 
                               title=f"{x_column} vs {y_column}")
                message = f"Created scatter plot: {x_column} vs {y_column}"
                
            elif chart_type.lower() in ["bar", "bar_chart"]:
                if not column:
                    # Find first categorical column
                    cat_cols = df.select_dtypes(include=['object', 'category']).columns
                    if len(cat_cols) == 0:
                        return ActionResult(
                            success=False,
                            action_type=ActionType.VISUALIZE,
                            error="No categorical columns found for bar chart"
                        )
                    column = cat_cols[0]
                
                value_counts = df[column].value_counts()
                fig = px.bar(x=value_counts.index, y=value_counts.values,
                           title=f"Distribution of {column}")
                message = f"Created bar chart for {column}"
                
            elif chart_type.lower() in ["correlation", "correlation_matrix"]:
                numeric_df = df.select_dtypes(include=['number'])
                if numeric_df.empty:
                    return ActionResult(
                        success=False,
                        action_type=ActionType.VISUALIZE,
                        error="No numeric columns found for correlation matrix"
                    )
                
                corr_matrix = numeric_df.corr()
                fig = px.imshow(corr_matrix, title="Correlation Matrix")
                message = "Created correlation matrix heatmap"
            
            if fig is None:
                return ActionResult(
                    success=False,
                    action_type=ActionType.VISUALIZE,
                    error=f"Unsupported chart type: {chart_type}"
                )
            
            return ActionResult(
                success=True,
                action_type=ActionType.VISUALIZE,
                visualization=fig,
                message=message,
                suggestions=["Try different chart types", "Filter data before visualizing"]
            )
            
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            return ActionResult(
                success=False,
                action_type=ActionType.VISUALIZE,
                error=str(e)
            )
    
    def analyze_column(self, column: str, **kwargs) -> ActionResult:
        """Analyze a specific column in detail."""
        try:
            df = self.dataset_context.dataframe
            if df is None:
                return ActionResult(
                    success=False,
                    action_type=ActionType.ANALYZE_COLUMN,
                    error="No dataset loaded"
                )
            
            if column not in df.columns:
                return ActionResult(
                    success=False,
                    action_type=ActionType.ANALYZE_COLUMN,
                    error=f"Column '{column}' not found in dataset"
                )
            
            col_data = df[column]
            analysis = {
                "column_name": column,
                "data_type": str(col_data.dtype),
                "total_values": len(col_data),
                "non_null_values": col_data.count(),
                "null_values": col_data.isnull().sum(),
                "unique_values": col_data.nunique()
            }
            
            if col_data.dtype in ['int64', 'float64']:
                analysis.update({
                    "min": col_data.min(),
                    "max": col_data.max(),
                    "mean": col_data.mean(),
                    "median": col_data.median(),
                    "std": col_data.std(),
                    "quantiles": {
                        "25%": col_data.quantile(0.25),
                        "50%": col_data.quantile(0.50),
                        "75%": col_data.quantile(0.75)
                    }
                })
            else:
                analysis.update({
                    "most_common": col_data.value_counts().head().to_dict(),
                    "sample_values": col_data.dropna().unique()[:10].tolist()
                })
            
            message = f"Analysis complete for column '{column}'. "
            message += f"Data type: {analysis['data_type']}, "
            message += f"Unique values: {analysis['unique_values']}, "
            message += f"Missing values: {analysis['null_values']}"
            
            return ActionResult(
                success=True,
                action_type=ActionType.ANALYZE_COLUMN,
                data=analysis,
                message=message,
                suggestions=[f"Visualize {column} distribution", f"Filter data by {column} values"]
            )
            
        except Exception as e:
            logger.error(f"Error analyzing column: {str(e)}")
            return ActionResult(
                success=False,
                action_type=ActionType.ANALYZE_COLUMN,
                error=str(e)
            )

class AIAssistant:
    """Enhanced AI-powered assistant for data analysis and visualization."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the AI assistant."""
        self.config = config
        self.client = None
        self.conversation_history = []
        self.is_initialized = False
        self.dataset_context = DatasetContext()
        self.actions = DataAnalysisActions(self.dataset_context)
        
        # Action mapping
        self.action_handlers = {
            ActionType.DESCRIBE_DATA: self.actions.describe_data,
            ActionType.VISUALIZE: self.actions.visualize_data,
            ActionType.ANALYZE_COLUMN: self.actions.analyze_column,
        }
        
    def initialize(self) -> bool:
        """Initialize OpenAI client."""
        try:
            llm_config = self.config.get("llm", {})
            api_key = llm_config.get("api_key")
            
            if not api_key:
                self.is_initialized = True
                return True
            
            self.client = openai.OpenAI(api_key=api_key)
            
            if llm_config.get("org_id"):
                openai.organization = llm_config.get("org_id")
                
            self.is_initialized = True
            logger.info("AI assistant initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing AI assistant: {str(e)}")
            return False
    
    def load_dataset(self, file_path: str, name: str = "") -> ActionResult:
        """Load a dataset into the context."""
        try:
            # Support common file formats
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            elif file_path.endswith('.json'):
                df = pd.read_json(file_path)
            else:
                return ActionResult(
                    success=False,
                    action_type=ActionType.LOAD_DATA,
                    error=f"Unsupported file format: {file_path}"
                )
            
            self.dataset_context.update_from_dataframe(
                df, name or Path(file_path).stem, file_path
            )
            
            # Update actions with new context
            self.actions = DataAnalysisActions(self.dataset_context)
            
            return ActionResult(
                success=True,
                action_type=ActionType.LOAD_DATA,
                message=f"Successfully loaded dataset '{self.dataset_context.name}' with {df.shape[0]} rows and {df.shape[1]} columns",
                suggestions=["Say 'describe data' to explore the dataset", "Try 'visualize [column]' to see distributions"]
            )
            
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            return ActionResult(
                success=False,
                action_type=ActionType.LOAD_DATA,
                error=str(e)
            )
    
    def process_voice_command(self, command: str, parsed_command: Dict[str, Any] = None) -> ActionResult:
        """Process a voice command and execute the appropriate action."""
        try:
            logger.info(f"Processing voice command: {command}")
            
            # If we have a parsed command structure, use it
            if parsed_command:
                return self._execute_parsed_command(parsed_command)
            
            # Otherwise, use AI to parse and execute
            if self.client:
                return self._process_with_ai(command)
            else:
                return self._process_with_patterns(command)
                
        except Exception as e:
            logger.error(f"Error processing voice command: {str(e)}")
            return ActionResult(
                success=False,
                action_type=ActionType.UNKNOWN,
                error=str(e)
            )
    
    def _execute_parsed_command(self, parsed_command: Dict[str, Any]) -> ActionResult:
        """Execute a pre-parsed command."""
        intent = parsed_command.get("intent", "unknown")
        
        # Map intents to actions
        intent_mapping = {
            "describe": ActionType.DESCRIBE_DATA,
            "visualize": ActionType.VISUALIZE,
            "analyze": ActionType.ANALYZE_COLUMN,
            "show": ActionType.DESCRIBE_DATA
        }
        
        action_type = intent_mapping.get(intent, ActionType.UNKNOWN)
        
        if action_type in self.action_handlers:
            # Extract parameters from parsed command
            kwargs = {
                "column": parsed_command.get("column"),
                "chart_type": parsed_command.get("chart_type", "histogram"),
                "x_column": parsed_command.get("x_column"),
                "y_column": parsed_command.get("y_column")
            }
            # Remove None values
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            
            return self.action_handlers[action_type](**kwargs)
        else:
            return ActionResult(
                success=False,
                action_type=ActionType.UNKNOWN,
                error=f"Unknown intent: {intent}"
            )
    
    def _process_with_patterns(self, command: str) -> ActionResult:
        """Process command using simple pattern matching."""
        command_lower = command.lower()
        
        if any(word in command_lower for word in ["describe", "show data", "info", "summary"]):
            return self.actions.describe_data()
        elif "visualize" in command_lower or "plot" in command_lower or "chart" in command_lower:
            # Extract column name if mentioned
            words = command.split()
            column = None
            for i, word in enumerate(words):
                if word.lower() in ["column", "visualize"] and i + 1 < len(words):
                    column = words[i + 1]
                    break
            return self.actions.visualize_data(column=column)
        elif "analyze" in command_lower:
            # Extract column name
            words = command.split()
            column = None
            for i, word in enumerate(words):
                if word.lower() == "analyze" and i + 1 < len(words):
                    column = words[i + 1]
                    break
            if column:
                return self.actions.analyze_column(column)
        
        return ActionResult(
            success=False,
            action_type=ActionType.UNKNOWN,
            error=f"Could not understand command: {command}",
            suggestions=[
                "Try 'describe data' to explore the dataset",
                "Say 'visualize [column name]' to create charts",
                "Use 'analyze [column name]' for detailed column analysis"
            ]
        )
    
    def _process_with_ai(self, command: str) -> ActionResult:
        """Process command using AI when available."""
        try:
            system_prompt = self._create_system_prompt()
            
            response = self.client.chat.completions.create(
                model=self.config["llm"]["model"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"User command: {command}"}
                ],
                temperature=self.config["llm"]["temperature"],
                max_tokens=self.config["llm"]["max_tokens"]
            )
            
            ai_response = response.choices[0].message.content
            parsed_response = self._parse_ai_response(ai_response)
            
            # Execute the parsed action
            return self._execute_ai_action(parsed_response)
            
        except Exception as e:
            logger.error(f"Error processing with AI: {str(e)}")
            return self._process_with_patterns(command)  # Fallback
    
    def _create_system_prompt(self) -> str:
        """Create system prompt with current dataset context."""
        prompt = """You are a data science assistant. Analyze user commands and respond with JSON containing the action to take.

Current Dataset:
"""
        if self.dataset_context.dataframe is not None:
            context = self.dataset_context.to_dict()
            prompt += f"- Name: {context['name']}\n"
            prompt += f"- Shape: {context['shape']}\n"
            prompt += f"- Columns: {', '.join(context['columns'])}\n"
        else:
            prompt += "- No dataset loaded\n"
        
        prompt += """
Respond with JSON in this format:
{
    "action": "describe_data|visualize|analyze_column",
    "parameters": {"column": "column_name", "chart_type": "histogram"},
    "explanation": "Brief explanation"
}

Available actions:
- describe_data: Show dataset overview
- visualize: Create charts (histogram, scatter, bar, correlation)
- analyze_column: Detailed column analysis
"""
        return prompt
    
    def _parse_ai_response(self, response: str) -> Dict[str, Any]:
        """Parse AI response and extract structured information."""
        try:
            if "{" in response and "}" in response:
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                json_str = response[json_start:json_end]
                return json.loads(json_str)
        except:
            pass
        
        return {"action": "unknown", "parameters": {}, "explanation": response}
    
    def _execute_ai_action(self, parsed_response: Dict[str, Any]) -> ActionResult:
        """Execute action from AI response."""
        action = parsed_response.get("action", "unknown")
        parameters = parsed_response.get("parameters", {})
        
        action_mapping = {
            "describe_data": ActionType.DESCRIBE_DATA,
            "visualize": ActionType.VISUALIZE,
            "analyze_column": ActionType.ANALYZE_COLUMN
        }
        
        action_type = action_mapping.get(action, ActionType.UNKNOWN)
        
        if action_type in self.action_handlers:
            return self.action_handlers[action_type](**parameters)
        else:
            return ActionResult(
                success=False,
                action_type=ActionType.UNKNOWN,
                error=f"Unknown action: {action}"
            )
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history."""
        return self.conversation_history
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get current dataset information."""
        return self.dataset_context.to_dict() if self.dataset_context.dataframe is not None else {} 