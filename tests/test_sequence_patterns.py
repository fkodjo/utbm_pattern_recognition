"""
Tests for Sequence Pattern Recognition Module
"""

import unittest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sequence_patterns import SequencePatternRecognizer


class TestSequencePatternRecognizer(unittest.TestCase):
    """Test cases for SequencePatternRecognizer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.recognizer = SequencePatternRecognizer()

    def test_is_arithmetic_sequence_true(self):
        """Test arithmetic sequence detection - positive case."""
        sequence = [2, 4, 6, 8, 10]
        is_arith, diff = self.recognizer.is_arithmetic_sequence(sequence)
        self.assertTrue(is_arith)
        self.assertEqual(diff, 2)

    def test_is_arithmetic_sequence_false(self):
        """Test arithmetic sequence detection - negative case."""
        sequence = [1, 2, 4, 8, 16]
        is_arith, diff = self.recognizer.is_arithmetic_sequence(sequence)
        self.assertFalse(is_arith)

    def test_is_geometric_sequence_true(self):
        """Test geometric sequence detection - positive case."""
        sequence = [2, 6, 18, 54]
        is_geom, ratio = self.recognizer.is_geometric_sequence(sequence)
        self.assertTrue(is_geom)
        self.assertEqual(ratio, 3)

    def test_is_geometric_sequence_false(self):
        """Test geometric sequence detection - negative case."""
        sequence = [1, 2, 3, 4, 5]
        is_geom, ratio = self.recognizer.is_geometric_sequence(sequence)
        self.assertFalse(is_geom)

    def test_find_next_in_arithmetic(self):
        """Test finding next number in arithmetic sequence."""
        sequence = [5, 10, 15, 20]
        next_num = self.recognizer.find_next_in_arithmetic(sequence)
        self.assertEqual(next_num, 25)

    def test_find_next_in_geometric(self):
        """Test finding next number in geometric sequence."""
        sequence = [3, 9, 27, 81]
        next_num = self.recognizer.find_next_in_geometric(sequence)
        self.assertEqual(next_num, 243)

    def test_detect_repeating_pattern(self):
        """Test detecting repeating patterns."""
        sequence = [1, 2, 3, 1, 2, 3, 1, 2, 3]
        pattern = self.recognizer.detect_repeating_pattern(sequence)
        self.assertEqual(pattern, [1, 2, 3])

    def test_detect_repeating_pattern_simple(self):
        """Test detecting simple repeating patterns."""
        sequence = ['a', 'b', 'a', 'b', 'a', 'b']
        pattern = self.recognizer.detect_repeating_pattern(sequence)
        self.assertEqual(pattern, ['a', 'b'])

    def test_find_missing_number(self):
        """Test finding missing number in sequence."""
        sequence = [2, 4, 8, 10]
        missing = self.recognizer.find_missing_number(sequence)
        self.assertEqual(missing, 6)

    def test_longest_increasing_subsequence(self):
        """Test finding longest increasing subsequence."""
        sequence = [10, 9, 2, 5, 3, 7, 101, 18]
        lis = self.recognizer.longest_increasing_subsequence(sequence)
        self.assertEqual(len(lis), 4)

    def test_is_fibonacci_like_true(self):
        """Test Fibonacci-like sequence detection - positive case."""
        sequence = [1, 1, 2, 3, 5, 8, 13]
        self.assertTrue(self.recognizer.is_fibonacci_like(sequence))

    def test_is_fibonacci_like_false(self):
        """Test Fibonacci-like sequence detection - negative case."""
        sequence = [1, 2, 3, 4, 5, 6]
        self.assertFalse(self.recognizer.is_fibonacci_like(sequence))

    def test_find_alternating_pattern(self):
        """Test finding alternating patterns."""
        sequence = [1, 10, 1, 10, 1, 10, 1, 10]
        result = self.recognizer.find_alternating_pattern(sequence)
        self.assertIsNotNone(result)
        self.assertEqual(result[0], [1])
        self.assertEqual(result[1], [10])


if __name__ == '__main__':
    unittest.main()
