"""
Numeric Pattern Analysis Module
Provides functionality for analyzing numerical patterns and statistics
"""

from typing import List, Dict, Tuple, Optional
import statistics


class NumericPatternAnalyzer:
    """
    A class for analyzing patterns in numerical data.
    Supports statistical analysis, trend detection, and numerical pattern recognition.
    """

    def __init__(self):
        """Initialize the NumericPatternAnalyzer."""
        pass

    def calculate_statistics(self, data: List[float]) -> Dict[str, float]:
        """
        Calculate basic statistics for numerical data.
        
        Args:
            data: List of numbers
            
        Returns:
            Dictionary with mean, median, mode, stddev, variance
        """
        if not data:
            return {}

        stats = {
            'mean': statistics.mean(data),
            'median': statistics.median(data),
            'variance': statistics.variance(data) if len(data) > 1 else 0,
            'stdev': statistics.stdev(data) if len(data) > 1 else 0,
            'min': min(data),
            'max': max(data),
            'range': max(data) - min(data),
        }

        try:
            stats['mode'] = statistics.mode(data)
        except statistics.StatisticsError:
            stats['mode'] = None

        return stats

    def detect_trend(self, data: List[float]) -> str:
        """
        Detect the overall trend in numerical data.
        
        Args:
            data: List of numbers
            
        Returns:
            'increasing', 'decreasing', 'stable', or 'fluctuating'
        """
        if len(data) < 2:
            return 'stable'

        differences = [data[i + 1] - data[i] for i in range(len(data) - 1)]

        positive_count = sum(1 for d in differences if d > 0)
        negative_count = sum(1 for d in differences if d < 0)
        zero_count = sum(1 for d in differences if d == 0)

        total = len(differences)
        if positive_count / total > 0.7:
            return 'increasing'
        elif negative_count / total > 0.7:
            return 'decreasing'
        elif zero_count / total > 0.7:
            return 'stable'
        else:
            return 'fluctuating'

    def find_outliers(self, data: List[float], threshold: float = 2.0) -> List[Tuple[int, float]]:
        """
        Find outliers in numerical data using standard deviation method.
        
        Args:
            data: List of numbers
            threshold: Number of standard deviations for outlier detection
            
        Returns:
            List of (index, value) tuples for outliers
        """
        if len(data) < 3:
            return []

        mean = statistics.mean(data)
        stdev = statistics.stdev(data)

        outliers = []
        for i, value in enumerate(data):
            z_score = abs((value - mean) / stdev) if stdev > 0 else 0
            if z_score > threshold:
                outliers.append((i, value))

        return outliers

    def detect_peaks(self, data: List[float]) -> List[int]:
        """
        Detect peaks (local maxima) in numerical data.
        
        Args:
            data: List of numbers
            
        Returns:
            List of indices where peaks occur
        """
        if len(data) < 3:
            return []

        peaks = []
        for i in range(1, len(data) - 1):
            if data[i] > data[i - 1] and data[i] > data[i + 1]:
                peaks.append(i)

        return peaks

    def detect_valleys(self, data: List[float]) -> List[int]:
        """
        Detect valleys (local minima) in numerical data.
        
        Args:
            data: List of numbers
            
        Returns:
            List of indices where valleys occur
        """
        if len(data) < 3:
            return []

        valleys = []
        for i in range(1, len(data) - 1):
            if data[i] < data[i - 1] and data[i] < data[i + 1]:
                valleys.append(i)

        return valleys

    def calculate_moving_average(self, data: List[float], window_size: int) -> List[float]:
        """
        Calculate moving average of numerical data.
        
        Args:
            data: List of numbers
            window_size: Size of the moving window
            
        Returns:
            List of moving averages
        """
        if window_size <= 0 or window_size > len(data):
            return []

        moving_avg = []
        for i in range(len(data) - window_size + 1):
            window = data[i:i + window_size]
            moving_avg.append(sum(window) / window_size)

        return moving_avg

    def find_frequency_distribution(self, data: List[float], num_bins: int = 10) -> Dict[str, int]:
        """
        Calculate frequency distribution of numerical data.
        
        Args:
            data: List of numbers
            num_bins: Number of bins for distribution
            
        Returns:
            Dictionary mapping bin ranges to frequencies
        """
        if not data or num_bins <= 0:
            return {}

        min_val = min(data)
        max_val = max(data)
        bin_width = (max_val - min_val) / num_bins

        frequency = {}
        for i in range(num_bins):
            bin_start = min_val + i * bin_width
            bin_end = bin_start + bin_width
            bin_label = f"{bin_start:.2f}-{bin_end:.2f}"
            count = sum(1 for x in data if bin_start <= x < bin_end or (i == num_bins - 1 and x == bin_end))
            frequency[bin_label] = count

        return frequency

    def detect_cyclic_pattern(self, data: List[float], period: int) -> bool:
        """
        Check if data exhibits a cyclic pattern with given period.
        
        Args:
            data: List of numbers
            period: Expected period of the cycle
            
        Returns:
            True if cyclic pattern detected, False otherwise
        """
        if len(data) < 2 * period:
            return False

        # Compare segments of the data with the given period
        num_cycles = len(data) // period
        threshold = 0.2  # 20% tolerance

        for i in range(num_cycles - 1):
            segment1 = data[i * period:(i + 1) * period]
            segment2 = data[(i + 1) * period:(i + 2) * period]

            if len(segment1) != len(segment2):
                continue

            # Calculate correlation
            differences = [abs(segment1[j] - segment2[j]) for j in range(len(segment1))]
            avg_diff = sum(differences) / len(differences)
            avg_magnitude = sum(abs(x) for x in segment1) / len(segment1)

            if avg_magnitude > 0 and avg_diff / avg_magnitude > threshold:
                return False

        return True

    def normalize_data(self, data: List[float]) -> List[float]:
        """
        Normalize numerical data to range [0, 1].
        
        Args:
            data: List of numbers
            
        Returns:
            Normalized list of numbers
        """
        if not data:
            return []

        min_val = min(data)
        max_val = max(data)
        range_val = max_val - min_val

        if range_val == 0:
            return [0.5] * len(data)

        return [(x - min_val) / range_val for x in data]

    def standardize_data(self, data: List[float]) -> List[float]:
        """
        Standardize numerical data (z-score normalization).
        
        Args:
            data: List of numbers
            
        Returns:
            Standardized list of numbers
        """
        if len(data) < 2:
            return [0.0] * len(data)

        mean = statistics.mean(data)
        stdev = statistics.stdev(data)

        if stdev == 0:
            return [0.0] * len(data)

        return [(x - mean) / stdev for x in data]
