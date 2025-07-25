"""
Command Parser
=============

Extracts user intent from Whisper transcription output using pattern matching
and NLP techniques. Supports data analysis commands like visualization, filtering,
sorting, and more.

Features:
- Regex-based pattern matching for fast prototyping
- Modular design for easy extension
- Structured output format
- Support for common data analysis operations
- Optional spaCy integration for advanced NLP
"""

import re
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import json

# Optional spaCy import
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

logger = logging.getLogger(__name__)

class CommandIntent(Enum):
    """Supported command intents."""
    VISUALIZE = "visualize"
    FILTER = "filter"
    SORT = "sort"
    GROUP = "group"
    AGGREGATE = "aggregate"
    SHOW = "show"
    DESCRIBE = "describe"
    UNKNOWN = "unknown"

class OperationType(Enum):
    """Supported operation types."""
    # Comparison operations
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    GREATER_EQUAL = "greater_equal"
    LESS_EQUAL = "less_equal"
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    
    # Sort operations
    ASCENDING = "ascending"
    DESCENDING = "descending"
    
    # Aggregation operations
    SUM = "sum"
    MEAN = "mean"
    COUNT = "count"
    MAX = "max"
    MIN = "min"
    
    # Visualization types
    BAR_CHART = "bar_chart"
    LINE_CHART = "line_chart"
    SCATTER_PLOT = "scatter_plot"
    HISTOGRAM = "histogram"
    PIE_CHART = "pie_chart"

@dataclass
class ParsedCommand:
    """Structured representation of a parsed command."""
    intent: CommandIntent
    column: Optional[str] = None
    operation: Optional[OperationType] = None
    value: Optional[str] = None
    chart_type: Optional[OperationType] = None
    sort_direction: Optional[OperationType] = None
    columns: Optional[List[str]] = None
    confidence: float = 0.0
    raw_text: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {
            "intent": self.intent.value,
            "raw_text": self.raw_text,
            "confidence": self.confidence
        }
        
        if self.column:
            result["column"] = self.column
        if self.operation:
            result["operation"] = self.operation.value
        if self.value:
            result["value"] = self.value
        if self.chart_type:
            result["chart_type"] = self.chart_type.value
        if self.sort_direction:
            result["sort_direction"] = self.sort_direction.value
        if self.columns:
            result["columns"] = self.columns
            
        return result

class PatternMatcher:
    """Regex-based pattern matching for command recognition."""
    
    def __init__(self):
        self.patterns = self._initialize_patterns()
    
    def _initialize_patterns(self) -> Dict[CommandIntent, List[Tuple[str, Dict[str, Any]]]]:
        """Initialize regex patterns for each command intent."""
        return {
            CommandIntent.VISUALIZE: [
                # "create bar chart for age" - more specific pattern first
                (r"create\s+((?:\w+\s+)*?\w+)\s+(?:for|of)\s+(\w+)", 
                 {"extract": ["chart_type", "column"]}),
                # "histogram of scores" - but avoid matching aggregation patterns
                (r"(histogram|bar\s+chart|line\s+chart|scatter\s+plot|pie\s+chart)\s+of\s+(\w+)", 
                 {"extract": ["chart_type", "column"]}),
                # "visualize column age", "plot age" - more general pattern last
                (r"(?:visualize|plot|show\s+(?:chart|graph|plot))\s+(?:column\s+)?(\w+)(?:\s+(?:as|with)\s+(\w+(?:\s+\w+)?))?", 
                 {"extract": ["column", "chart_type"]}),
                # "chart age" - simple chart command
                (r"^chart\s+(?:column\s+)?(\w+)(?:\s+(?:as|with)\s+(\w+(?:\s+\w+)?))?", 
                 {"extract": ["column", "chart_type"]}),
            ],
            
            CommandIntent.FILTER: [
                # "filter country equals Germany", "where country is Germany"
                (r"(?:filter|where)\s+(\w+)\s+(?:equals|is|==)\s+(['\"]?)([^'\"]+)\2", 
                 {"extract": ["column", None, "value"], "operation": OperationType.EQUALS}),
                # "filter age greater than 25"
                (r"(?:filter|where)\s+(\w+)\s+(?:greater\s+than|>)\s+(['\"]?)([^'\"]+)\2", 
                 {"extract": ["column", None, "value"], "operation": OperationType.GREATER_THAN}),
                # "filter age less than 30"
                (r"(?:filter|where)\s+(\w+)\s+(?:less\s+than|<)\s+(['\"]?)([^'\"]+)\2", 
                 {"extract": ["column", None, "value"], "operation": OperationType.LESS_THAN}),
                # "filter name contains John"
                (r"(?:filter|where)\s+(\w+)\s+contains\s+(['\"]?)([^'\"]+)\2", 
                 {"extract": ["column", None, "value"], "operation": OperationType.CONTAINS}),
                # "show only country Germany"
                (r"show\s+only\s+(\w+)\s+(['\"]?)([^'\"]+)\2", 
                 {"extract": ["column", None, "value"], "operation": OperationType.EQUALS}),
            ],
            
            CommandIntent.SORT: [
                # "sort by age descending", "order by score desc"
                (r"(?:sort|order)\s+by\s+(\w+)\s+(descending|desc|ascending|asc)", 
                 {"extract": ["column", "sort_direction"]}),
                # "sort age ascending"
                (r"sort\s+(\w+)\s+(descending|desc|ascending|asc)", 
                 {"extract": ["column", "sort_direction"]}),
                # "sort by age" (default ascending)
                (r"(?:sort|order)\s+by\s+(\w+)", 
                 {"extract": ["column"], "sort_direction": OperationType.ASCENDING}),
            ],
            
            CommandIntent.GROUP: [
                # "group by country", "group data by category"
                (r"group\s+(?:data\s+)?by\s+(\w+)", 
                 {"extract": ["column"]}),
            ],
            
            CommandIntent.AGGREGATE: [
                # "sum of sales", "average age", "count records"
                (r"(sum|average|mean|count|max|min)\s+(?:of\s+)?(\w+)", 
                 {"extract": ["operation", "column"]}),
                # "calculate sum of revenue"
                (r"calculate\s+(sum|average|mean|count|max|min)\s+of\s+(\w+)", 
                 {"extract": ["operation", "column"]}),
            ],
            
            CommandIntent.SHOW: [
                # "show data", "display table", "show first 10 rows"
                (r"(?:show|display)\s+(?:data|table|dataset)", 
                 {"extract": []}),
                (r"(?:show|display)\s+(?:first|top)\s+(\d+)\s+(?:rows|records)", 
                 {"extract": ["value"]}),
            ],
            
            CommandIntent.DESCRIBE: [
                # "describe data", "summary statistics", "info about dataset"
                (r"(?:describe|summarize)\s+(?:data|dataset)", 
                 {"extract": []}),
                (r"(?:summary|statistics|info)(?:\s+(?:about\s+)?(?:data|dataset))?", 
                 {"extract": []}),
            ],
        }
    
    def match(self, text: str) -> Optional[ParsedCommand]:
        """Match text against patterns and return parsed command."""
        text = text.lower().strip()
        
        for intent, patterns in self.patterns.items():
            for pattern, config in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return self._extract_command(text, match, intent, config)
        
        return None
    
    def _extract_command(self, text: str, match: re.Match, intent: CommandIntent, config: Dict) -> ParsedCommand:
        """Extract command details from regex match."""
        command = ParsedCommand(intent=intent, raw_text=text, confidence=0.8)
        
        # Extract matched groups
        groups = match.groups()
        extract_keys = config.get("extract", [])
        
        for i, key in enumerate(extract_keys):
            if key and i < len(groups) and groups[i]:
                value = groups[i].strip()
                
                if key == "column":
                    command.column = value
                elif key == "value":
                    command.value = value
                elif key == "chart_type":
                    command.chart_type = self._parse_chart_type(value)
                elif key == "sort_direction":
                    command.sort_direction = self._parse_sort_direction(value)
                elif key == "operation":
                    command.operation = self._parse_operation(value)
        
        # Apply fixed configurations
        if "operation" in config:
            command.operation = config["operation"]
        if "sort_direction" in config:
            command.sort_direction = config["sort_direction"]
        
        return command
    
    def _parse_chart_type(self, value: str) -> Optional[OperationType]:
        """Parse chart type from text."""
        value = value.lower()
        chart_mappings = {
            "bar": OperationType.BAR_CHART,
            "bar chart": OperationType.BAR_CHART,
            "line": OperationType.LINE_CHART,
            "line chart": OperationType.LINE_CHART,
            "scatter": OperationType.SCATTER_PLOT,
            "scatter plot": OperationType.SCATTER_PLOT,
            "histogram": OperationType.HISTOGRAM,
            "pie": OperationType.PIE_CHART,
            "pie chart": OperationType.PIE_CHART,
        }
        return chart_mappings.get(value)
    
    def _parse_sort_direction(self, value: str) -> Optional[OperationType]:
        """Parse sort direction from text."""
        value = value.lower()
        if value in ["descending", "desc"]:
            return OperationType.DESCENDING
        elif value in ["ascending", "asc"]:
            return OperationType.ASCENDING
        return None
    
    def _parse_operation(self, value: str) -> Optional[OperationType]:
        """Parse operation type from text."""
        value = value.lower()
        operation_mappings = {
            "sum": OperationType.SUM,
            "average": OperationType.MEAN,
            "mean": OperationType.MEAN,
            "count": OperationType.COUNT,
            "max": OperationType.MAX,
            "min": OperationType.MIN,
        }
        return operation_mappings.get(value)

class NLPMatcher:
    """spaCy-based NLP matcher for more sophisticated parsing."""
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        if not SPACY_AVAILABLE:
            logger.warning("spaCy not available. Install with: pip install spacy")
            self.nlp = None
            return
        
        try:
            self.nlp = spacy.load(model_name)
            logger.info(f"Loaded spaCy model: {model_name}")
        except OSError:
            logger.warning(f"spaCy model '{model_name}' not found. Install with: python -m spacy download {model_name}")
            self.nlp = None
    
    def match(self, text: str) -> Optional[ParsedCommand]:
        """Match text using NLP techniques."""
        if not self.nlp:
            return None
        
        doc = self.nlp(text.lower())
        
        # Extract entities and analyze syntax
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        tokens = [(token.text, token.pos_, token.dep_) for token in doc]
        
        # Simple NLP-based intent detection
        intent = self._detect_intent_nlp(doc)
        if intent == CommandIntent.UNKNOWN:
            return None
        
        command = ParsedCommand(intent=intent, raw_text=text, confidence=0.9)
        
        # Extract column names (likely to be nouns)
        potential_columns = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop]
        if potential_columns:
            command.column = potential_columns[0]
        
        # Extract values (numbers, quoted strings, proper nouns)
        for token in doc:
            if token.like_num or token.ent_type_ in ["PERSON", "ORG", "GPE"]:
                command.value = token.text
                break
        
        return command
    
    def _detect_intent_nlp(self, doc) -> CommandIntent:
        """Detect intent using NLP analysis."""
        text = doc.text.lower()
        
        # Intent keywords
        intent_keywords = {
            CommandIntent.VISUALIZE: ["visualize", "plot", "chart", "graph", "show"],
            CommandIntent.FILTER: ["filter", "where", "only", "select"],
            CommandIntent.SORT: ["sort", "order", "arrange"],
            CommandIntent.GROUP: ["group"],
            CommandIntent.AGGREGATE: ["sum", "count", "average", "mean", "total"],
            CommandIntent.SHOW: ["show", "display", "list"],
            CommandIntent.DESCRIBE: ["describe", "summary", "info", "statistics"],
        }
        
        for intent, keywords in intent_keywords.items():
            if any(keyword in text for keyword in keywords):
                return intent
        
        return CommandIntent.UNKNOWN

class CommandParser:
    """Main command parser that orchestrates different matching strategies."""
    
    def __init__(self, use_nlp: bool = False, nlp_model: str = "en_core_web_sm"):
        self.pattern_matcher = PatternMatcher()
        self.nlp_matcher = None
        
        if use_nlp:
            self.nlp_matcher = NLPMatcher(nlp_model)
        
        logger.info(f"CommandParser initialized (NLP: {'enabled' if use_nlp and self.nlp_matcher and self.nlp_matcher.nlp else 'disabled'})")
    
    def parse(self, text: str) -> ParsedCommand:
        """Parse command text and return structured output."""
        if not text or not text.strip():
            return ParsedCommand(intent=CommandIntent.UNKNOWN, raw_text=text, confidence=0.0)
        
        text = text.strip()
        logger.debug(f"Parsing command: '{text}'")
        
        # Try pattern matching first (faster)
        result = self.pattern_matcher.match(text)
        if result and result.confidence > 0.5:
            logger.debug(f"Pattern match successful: {result.intent.value}")
            return result
        
        # Try NLP matching if available
        if self.nlp_matcher:
            nlp_result = self.nlp_matcher.match(text)
            if nlp_result and nlp_result.confidence > 0.5:
                logger.debug(f"NLP match successful: {nlp_result.intent.value}")
                return nlp_result
        
        # Return unknown command
        logger.debug("No match found, returning unknown command")
        return ParsedCommand(intent=CommandIntent.UNKNOWN, raw_text=text, confidence=0.0)
    
    def add_custom_pattern(self, intent: CommandIntent, pattern: str, config: Dict[str, Any]):
        """Add custom regex pattern for specific intent."""
        if intent not in self.pattern_matcher.patterns:
            self.pattern_matcher.patterns[intent] = []
        
        self.pattern_matcher.patterns[intent].append((pattern, config))
        logger.info(f"Added custom pattern for {intent.value}: {pattern}")
    
    def get_supported_intents(self) -> List[str]:
        """Get list of supported command intents."""
        return [intent.value for intent in CommandIntent if intent != CommandIntent.UNKNOWN]
    
    def get_supported_operations(self) -> List[str]:
        """Get list of supported operations."""
        return [op.value for op in OperationType]

# Utility functions
def create_parser(use_nlp: bool = False, nlp_model: str = "en_core_web_sm") -> CommandParser:
    """Create and configure a command parser."""
    return CommandParser(use_nlp=use_nlp, nlp_model=nlp_model)

def parse_command(text: str, use_nlp: bool = False) -> Dict[str, Any]:
    """Quick function to parse a command and return dictionary."""
    parser = create_parser(use_nlp=use_nlp)
    result = parser.parse(text)
    return result.to_dict()

# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Test commands
    test_commands = [
        "visualize column age",
        "filter country equals Germany", 
        "sort by score descending",
        "create bar chart for sales",
        "show only status active",
        "group by category",
        "sum of revenue",
        "describe data",
        "histogram of prices",
        "filter age greater than 25",
        "sort name ascending",
    ]
    
    print("🧪 Testing Command Parser")
    print("=" * 50)
    
    # Test with regex patterns
    parser = create_parser(use_nlp=False)
    
    for command in test_commands:
        result = parser.parse(command)
        print(f"Input: '{command}'")
        print(f"Output: {json.dumps(result.to_dict(), indent=2)}")
        print("-" * 30) 