"""
Examples of using the Numeric Pattern Analysis Module
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from numeric_patterns import NumericPatternAnalyzer


def main():
    """Run numeric pattern analysis examples."""
    print("=" * 60)
    print("NUMERIC PATTERN ANALYSIS EXAMPLES")
    print("=" * 60)

    analyzer = NumericPatternAnalyzer()

    # Example 1: Basic statistics
    print("\n1. Basic statistics calculation:")
    data = [12, 15, 18, 20, 22, 25, 28, 30]
    stats = analyzer.calculate_statistics(data)
    print(f"   Data: {data}")
    print(f"   Mean: {stats['mean']:.2f}")
    print(f"   Median: {stats['median']:.2f}")
    print(f"   Std Dev: {stats['stdev']:.2f}")
    print(f"   Min: {stats['min']}, Max: {stats['max']}")

    # Example 2: Trend detection
    print("\n2. Trend detection:")
    datasets = [
        [1, 2, 3, 4, 5, 6],
        [10, 9, 8, 7, 6],
        [5, 5, 5, 5, 5],
        [1, 5, 2, 6, 3, 7]
    ]
    for data in datasets:
        trend = analyzer.detect_trend(data)
        print(f"   {data} -> Trend: {trend}")

    # Example 3: Outlier detection
    print("\n3. Outlier detection:")
    data = [10, 12, 13, 12, 11, 13, 14, 100, 12, 11]
    outliers = analyzer.find_outliers(data, threshold=2.0)
    print(f"   Data: {data}")
    print(f"   Outliers: {[(idx, val) for idx, val in outliers]}")

    # Example 4: Peak and valley detection
    print("\n4. Peak and valley detection:")
    data = [1, 3, 2, 5, 4, 6, 3, 7, 4]
    peaks = analyzer.detect_peaks(data)
    valleys = analyzer.detect_valleys(data)
    print(f"   Data: {data}")
    print(f"   Peaks at indices: {peaks} -> values: {[data[i] for i in peaks]}")
    print(f"   Valleys at indices: {valleys} -> values: {[data[i] for i in valleys]}")

    # Example 5: Moving average
    print("\n5. Moving average calculation:")
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    window_size = 3
    moving_avg = analyzer.calculate_moving_average(data, window_size)
    print(f"   Data: {data}")
    print(f"   Moving average (window={window_size}): {[f'{x:.2f}' for x in moving_avg]}")

    # Example 6: Frequency distribution
    print("\n6. Frequency distribution:")
    data = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5]
    freq = analyzer.find_frequency_distribution(data, num_bins=5)
    print(f"   Data: {data}")
    print(f"   Frequency distribution:")
    for bin_range, count in freq.items():
        print(f"      {bin_range}: {'*' * count} ({count})")

    # Example 7: Cyclic pattern detection
    print("\n7. Cyclic pattern detection:")
    datasets = [
        ([1, 2, 3, 1, 2, 3, 1, 2, 3], 3),
        ([5, 10, 5, 10, 5, 10], 2),
        ([1, 2, 3, 4, 5, 6, 7, 8], 4)
    ]
    for data, period in datasets:
        is_cyclic = analyzer.detect_cyclic_pattern(data, period)
        print(f"   Data: {data}, Period: {period} -> Cyclic: {is_cyclic}")

    # Example 8: Data normalization
    print("\n8. Data normalization:")
    data = [10, 20, 30, 40, 50]
    normalized = analyzer.normalize_data(data)
    print(f"   Original: {data}")
    print(f"   Normalized: {[f'{x:.2f}' for x in normalized]}")

    # Example 9: Data standardization
    print("\n9. Data standardization (z-score):")
    data = [10, 20, 30, 40, 50]
    standardized = analyzer.standardize_data(data)
    print(f"   Original: {data}")
    print(f"   Standardized: {[f'{x:.2f}' for x in standardized]}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
