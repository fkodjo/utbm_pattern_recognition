"""
Examples of using the Sequence Pattern Recognition Module
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sequence_patterns import SequencePatternRecognizer


def main():
    """Run sequence pattern recognition examples."""
    print("=" * 60)
    print("SEQUENCE PATTERN RECOGNITION EXAMPLES")
    print("=" * 60)

    recognizer = SequencePatternRecognizer()

    # Example 1: Arithmetic sequences
    print("\n1. Arithmetic sequence detection:")
    sequences = [
        [2, 4, 6, 8, 10],
        [5, 10, 15, 20, 25],
        [1, 2, 4, 8, 16]
    ]
    for seq in sequences:
        is_arith, diff = recognizer.is_arithmetic_sequence(seq)
        print(f"   {seq}")
        print(f"   Is arithmetic: {is_arith}, Difference: {diff}")
        if is_arith:
            next_num = recognizer.find_next_in_arithmetic(seq)
            print(f"   Next number: {next_num}")

    # Example 2: Geometric sequences
    print("\n2. Geometric sequence detection:")
    sequences = [
        [2, 6, 18, 54],
        [3, 9, 27, 81],
        [1, 2, 3, 4, 5]
    ]
    for seq in sequences:
        is_geom, ratio = recognizer.is_geometric_sequence(seq)
        print(f"   {seq}")
        print(f"   Is geometric: {is_geom}, Ratio: {ratio}")
        if is_geom:
            next_num = recognizer.find_next_in_geometric(seq)
            print(f"   Next number: {next_num}")

    # Example 3: Repeating patterns
    print("\n3. Repeating pattern detection:")
    sequences = [
        [1, 2, 3, 1, 2, 3, 1, 2, 3],
        ['A', 'B', 'A', 'B', 'A', 'B'],
        [5, 10, 5, 10, 5, 10]
    ]
    for seq in sequences:
        pattern = recognizer.detect_repeating_pattern(seq)
        print(f"   Sequence: {seq}")
        print(f"   Repeating pattern: {pattern}")

    # Example 4: Fibonacci-like sequences
    print("\n4. Fibonacci-like sequence detection:")
    sequences = [
        [1, 1, 2, 3, 5, 8, 13],
        [2, 3, 5, 8, 13, 21],
        [1, 2, 3, 4, 5, 6]
    ]
    for seq in sequences:
        is_fib = recognizer.is_fibonacci_like(seq)
        print(f"   {seq}")
        print(f"   Is Fibonacci-like: {is_fib}")

    # Example 5: Finding missing numbers
    print("\n5. Finding missing numbers:")
    sequences = [
        [2, 4, 8, 10],
        [10, 20, 40, 50],
        [1, 3, 5, 9, 11]
    ]
    for seq in sequences:
        missing = recognizer.find_missing_number(seq)
        print(f"   Sequence: {seq}")
        print(f"   Missing number: {missing}")

    # Example 6: Longest increasing subsequence
    print("\n6. Longest increasing subsequence:")
    sequences = [
        [10, 9, 2, 5, 3, 7, 101, 18],
        [0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15]
    ]
    for seq in sequences:
        lis = recognizer.longest_increasing_subsequence(seq)
        print(f"   Sequence: {seq}")
        print(f"   LIS: {lis} (length: {len(lis)})")

    # Example 7: Alternating patterns
    print("\n7. Alternating pattern detection:")
    sequences = [
        [1, 10, 2, 20, 3, 30, 4, 40],
        ['a', 'A', 'b', 'B', 'c', 'C']
    ]
    for seq in sequences:
        result = recognizer.find_alternating_pattern(seq)
        print(f"   Sequence: {seq}")
        if result:
            print(f"   Even positions: {result[0]}")
            print(f"   Odd positions: {result[1]}")
        else:
            print(f"   No alternating pattern found")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
