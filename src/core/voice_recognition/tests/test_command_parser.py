"""
Command Parser Tests
===================

Comprehensive tests for the command parser functionality including:
- Pattern matching tests
- Integration with voice handler
- Custom pattern addition
- Edge case handling
"""

import sys
import os
import unittest
import json
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from command_parser import (
    CommandParser, CommandIntent, OperationType, 
    create_parser, parse_command, ParsedCommand
)

class TestCommandParser(unittest.TestCase):
    """Test cases for command parser."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.parser = create_parser(use_nlp=False)
    
    def test_visualize_commands(self):
        """Test visualization command parsing."""
        test_cases = [
            {
                "input": "visualize column age",
                "expected": {
                    "intent": "visualize",
                    "column": "age"
                }
            },
            {
                "input": "create bar chart for sales",
                "expected": {
                    "intent": "visualize",
                    "column": "sales",
                    "chart_type": "bar_chart"
                }
            },
            {
                "input": "histogram of prices",
                "expected": {
                    "intent": "visualize",
                    "column": "prices",
                    "chart_type": "histogram"
                }
            },
            {
                "input": "plot revenue",
                "expected": {
                    "intent": "visualize",
                    "column": "revenue"
                }
            }
        ]
        
        for case in test_cases:
            with self.subTest(input=case["input"]):
                result = self.parser.parse(case["input"])
                self.assertEqual(result.intent, CommandIntent.VISUALIZE)
                
                for key, expected_value in case["expected"].items():
                    if key == "intent":
                        continue
                    elif key == "column":
                        self.assertEqual(result.column, expected_value)
                    elif key == "chart_type":
                        self.assertEqual(result.chart_type.value, expected_value)
    
    def test_filter_commands(self):
        """Test filter command parsing."""
        test_cases = [
            {
                "input": "filter country equals Germany",
                "expected": {
                    "intent": "filter",
                    "column": "country",
                    "operation": "equals",
                    "value": "Germany"
                }
            },
            {
                "input": "filter age greater than 25",
                "expected": {
                    "intent": "filter",
                    "column": "age", 
                    "operation": "greater_than",
                    "value": "25"
                }
            },
            {
                "input": "where status is active",
                "expected": {
                    "intent": "filter",
                    "column": "status",
                    "operation": "equals",
                    "value": "active"
                }
            },
            {
                "input": "show only category electronics",
                "expected": {
                    "intent": "filter",
                    "column": "category",
                    "operation": "equals",
                    "value": "electronics"
                }
            }
        ]
        
        for case in test_cases:
            with self.subTest(input=case["input"]):
                result = self.parser.parse(case["input"])
                self.assertEqual(result.intent, CommandIntent.FILTER)
                self.assertEqual(result.column, case["expected"]["column"])
                self.assertEqual(result.operation.value, case["expected"]["operation"])
                self.assertEqual(result.value.lower(), case["expected"]["value"].lower())
    
    def test_sort_commands(self):
        """Test sort command parsing."""
        test_cases = [
            {
                "input": "sort by age descending",
                "expected": {
                    "intent": "sort",
                    "column": "age",
                    "sort_direction": "descending"
                }
            },
            {
                "input": "order by score asc",
                "expected": {
                    "intent": "sort",
                    "column": "score",
                    "sort_direction": "ascending"
                }
            },
            {
                "input": "sort name ascending",
                "expected": {
                    "intent": "sort",
                    "column": "name",
                    "sort_direction": "ascending"
                }
            },
            {
                "input": "sort by price",  # Default ascending
                "expected": {
                    "intent": "sort",
                    "column": "price",
                    "sort_direction": "ascending"
                }
            }
        ]
        
        for case in test_cases:
            with self.subTest(input=case["input"]):
                result = self.parser.parse(case["input"])
                self.assertEqual(result.intent, CommandIntent.SORT)
                self.assertEqual(result.column, case["expected"]["column"])
                self.assertEqual(result.sort_direction.value, case["expected"]["sort_direction"])
    
    def test_aggregate_commands(self):
        """Test aggregation command parsing."""
        test_cases = [
            {
                "input": "sum of revenue",
                "expected": {
                    "intent": "aggregate",
                    "column": "revenue",
                    "operation": "sum"
                }
            },
            {
                "input": "average age",
                "expected": {
                    "intent": "aggregate",
                    "column": "age",
                    "operation": "mean"
                }
            },
            {
                "input": "count records",
                "expected": {
                    "intent": "aggregate",
                    "column": "records",
                    "operation": "count"
                }
            },
            {
                "input": "calculate max of score",
                "expected": {
                    "intent": "aggregate",
                    "column": "score",
                    "operation": "max"
                }
            }
        ]
        
        for case in test_cases:
            with self.subTest(input=case["input"]):
                result = self.parser.parse(case["input"])
                self.assertEqual(result.intent, CommandIntent.AGGREGATE)
                self.assertEqual(result.column, case["expected"]["column"])
                self.assertEqual(result.operation.value, case["expected"]["operation"])
    
    def test_group_commands(self):
        """Test group command parsing."""
        test_cases = [
            {
                "input": "group by category",
                "expected": {
                    "intent": "group",
                    "column": "category"
                }
            },
            {
                "input": "group data by region",
                "expected": {
                    "intent": "group",
                    "column": "region"
                }
            }
        ]
        
        for case in test_cases:
            with self.subTest(input=case["input"]):
                result = self.parser.parse(case["input"])
                self.assertEqual(result.intent, CommandIntent.GROUP)
                self.assertEqual(result.column, case["expected"]["column"])
    
    def test_show_describe_commands(self):
        """Test show and describe command parsing."""
        test_cases = [
            {
                "input": "show data",
                "expected": {"intent": "show"}
            },
            {
                "input": "display table",
                "expected": {"intent": "show"}
            },
            {
                "input": "describe data",
                "expected": {"intent": "describe"}
            },
            {
                "input": "summary statistics",
                "expected": {"intent": "describe"}
            }
        ]
        
        for case in test_cases:
            with self.subTest(input=case["input"]):
                result = self.parser.parse(case["input"])
                expected_intent = CommandIntent(case["expected"]["intent"])
                self.assertEqual(result.intent, expected_intent)
    
    def test_unknown_commands(self):
        """Test handling of unknown commands."""
        unknown_inputs = [
            "",
            "   ",
            "random text",
            "hello world",
            "this is not a command"
        ]
        
        for input_text in unknown_inputs:
            with self.subTest(input=input_text):
                result = self.parser.parse(input_text)
                self.assertEqual(result.intent, CommandIntent.UNKNOWN)
                self.assertEqual(result.confidence, 0.0)
    
    def test_custom_patterns(self):
        """Test adding custom patterns."""
        # Add custom pattern for export command
        custom_pattern = r"export\s+(?:data\s+)?(?:to\s+)?(\w+)"
        custom_config = {"extract": ["value"]}
        
        self.parser.add_custom_pattern(CommandIntent.SHOW, custom_pattern, custom_config)
        
        result = self.parser.parse("export data to csv")
        self.assertEqual(result.intent, CommandIntent.SHOW)
        self.assertEqual(result.value, "csv")
    
    def test_case_insensitivity(self):
        """Test that parsing is case insensitive."""
        test_cases = [
            "VISUALIZE COLUMN AGE",
            "Filter Country Equals Germany",
            "sort BY score DESCENDING"
        ]
        
        for case in test_cases:
            with self.subTest(input=case):
                result = self.parser.parse(case)
                self.assertNotEqual(result.intent, CommandIntent.UNKNOWN)
                self.assertGreater(result.confidence, 0.5)
    
    def test_to_dict_conversion(self):
        """Test ParsedCommand to dictionary conversion."""
        result = self.parser.parse("filter age greater than 25")
        result_dict = result.to_dict()
        
        expected_keys = ["intent", "raw_text", "confidence", "column", "operation", "value"]
        for key in expected_keys:
            self.assertIn(key, result_dict)
        
        self.assertEqual(result_dict["intent"], "filter")
        self.assertEqual(result_dict["column"], "age")
        self.assertEqual(result_dict["operation"], "greater_than")
        self.assertEqual(result_dict["value"], "25")
    
    def test_utility_functions(self):
        """Test utility functions."""
        # Test parse_command function
        result_dict = parse_command("visualize column sales")
        self.assertEqual(result_dict["intent"], "visualize")
        self.assertEqual(result_dict["column"], "sales")
        
        # Test supported intents and operations
        intents = self.parser.get_supported_intents()
        self.assertIn("visualize", intents)
        self.assertIn("filter", intents)
        self.assertNotIn("unknown", intents)
        
        operations = self.parser.get_supported_operations()
        self.assertIn("equals", operations)
        self.assertIn("bar_chart", operations)

class TestVoiceIntegration(unittest.TestCase):
    """Test integration with voice recognition."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.parser = create_parser()
    
    def test_whisper_like_inputs(self):
        """Test parsing of Whisper-like transcription outputs."""
        # Simulate common Whisper transcription variations
        whisper_outputs = [
            "visualize column age",  # Perfect transcription
            "visualize column eight",  # Misheard "age" as "eight"  
            "filter country equals germany",  # Lowercase country name
            "sort by score descending order",  # Extra words
            "show me the data please",  # Conversational style
            "can you filter age greater than twenty five",  # Natural language
        ]
        
        for output in whisper_outputs:
            with self.subTest(input=output):
                result = self.parser.parse(output)
                # Should at least attempt to parse, even if not perfect
                self.assertIsInstance(result, ParsedCommand)

def run_interactive_test():
    """Run interactive test for manual verification."""
    print("🎤 Interactive Command Parser Test")
    print("=" * 40)
    print("Enter commands to test (or 'quit' to exit):")
    print("Examples:")
    print("  - visualize column age")
    print("  - filter country equals Germany")
    print("  - sort by score descending")
    print()
    
    parser = create_parser()
    
    while True:
        try:
            user_input = input("Command: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                continue
            
            result = parser.parse(user_input)
            print(f"Parsed: {json.dumps(result.to_dict(), indent=2)}")
            print()
            
        except KeyboardInterrupt:
            break
    
    print("Goodbye! 👋")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test command parser")
    parser.add_argument("--interactive", "-i", action="store_true", 
                       help="Run interactive test mode")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    if args.interactive:
        run_interactive_test()
    else:
        # Run unit tests
        if args.verbose:
            unittest.main(verbosity=2)
        else:
            unittest.main() 