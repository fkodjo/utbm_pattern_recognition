"""
VA52 Pattern Recognition Module
School TP (Travaux Pratiques) - Pattern Recognition Journey
"""

__version__ = "1.0.0"
__author__ = "VA52 School TP"

from .string_patterns import StringPatternMatcher
from .sequence_patterns import SequencePatternRecognizer
from .numeric_patterns import NumericPatternAnalyzer

__all__ = [
    "StringPatternMatcher",
    "SequencePatternRecognizer",
    "NumericPatternAnalyzer",
]
