"""
Tests for Numeric Pattern Analysis Module
"""

import unittest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from numeric_patterns import NumericPatternAnalyzer


class TestNumericPatternAnalyzer(unittest.TestCase):
    """Test cases for NumericPatternAnalyzer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = NumericPatternAnalyzer()

    def test_calculate_statistics(self):
        """Test basic statistics calculation."""
        data = [1, 2, 3, 4, 5]
        stats = self.analyzer.calculate_statistics(data)
        self.assertEqual(stats['mean'], 3.0)
        self.assertEqual(stats['median'], 3.0)
        self.assertEqual(stats['min'], 1)
        self.assertEqual(stats['max'], 5)

    def test_detect_trend_increasing(self):
        """Test trend detection - increasing."""
        data = [1, 2, 3, 4, 5, 6]
        trend = self.analyzer.detect_trend(data)
        self.assertEqual(trend, 'increasing')

    def test_detect_trend_decreasing(self):
        """Test trend detection - decreasing."""
        data = [10, 9, 8, 7, 6]
        trend = self.analyzer.detect_trend(data)
        self.assertEqual(trend, 'decreasing')

    def test_detect_trend_stable(self):
        """Test trend detection - stable."""
        data = [5, 5, 5, 5, 5]
        trend = self.analyzer.detect_trend(data)
        self.assertEqual(trend, 'stable')

    def test_find_outliers(self):
        """Test outlier detection."""
        data = [1, 2, 3, 4, 5, 100]
        outliers = self.analyzer.find_outliers(data)
        self.assertEqual(len(outliers), 1)
        self.assertEqual(outliers[0][1], 100)

    def test_detect_peaks(self):
        """Test peak detection."""
        data = [1, 3, 2, 5, 4, 6, 3]
        peaks = self.analyzer.detect_peaks(data)
        self.assertIn(1, peaks)  # Index 1 has value 3
        self.assertIn(3, peaks)  # Index 3 has value 5

    def test_detect_valleys(self):
        """Test valley detection."""
        data = [5, 2, 4, 1, 3, 2, 5]
        valleys = self.analyzer.detect_valleys(data)
        self.assertIn(1, valleys)  # Index 1 has value 2
        self.assertIn(3, valleys)  # Index 3 has value 1

    def test_calculate_moving_average(self):
        """Test moving average calculation."""
        data = [1, 2, 3, 4, 5]
        moving_avg = self.analyzer.calculate_moving_average(data, 3)
        self.assertEqual(len(moving_avg), 3)
        self.assertEqual(moving_avg[0], 2.0)  # (1+2+3)/3

    def test_find_frequency_distribution(self):
        """Test frequency distribution calculation."""
        data = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
        freq = self.analyzer.find_frequency_distribution(data, 4)
        self.assertIsInstance(freq, dict)
        self.assertEqual(sum(freq.values()), len(data))

    def test_detect_cyclic_pattern(self):
        """Test cyclic pattern detection."""
        data = [1, 2, 3, 1, 2, 3, 1, 2, 3]
        is_cyclic = self.analyzer.detect_cyclic_pattern(data, 3)
        self.assertTrue(is_cyclic)

    def test_normalize_data(self):
        """Test data normalization."""
        data = [0, 5, 10]
        normalized = self.analyzer.normalize_data(data)
        self.assertEqual(normalized[0], 0.0)
        self.assertEqual(normalized[1], 0.5)
        self.assertEqual(normalized[2], 1.0)

    def test_standardize_data(self):
        """Test data standardization."""
        data = [1, 2, 3, 4, 5]
        standardized = self.analyzer.standardize_data(data)
        # Mean should be close to 0
        mean_standardized = sum(standardized) / len(standardized)
        self.assertAlmostEqual(mean_standardized, 0.0, places=10)


if __name__ == '__main__':
    unittest.main()
