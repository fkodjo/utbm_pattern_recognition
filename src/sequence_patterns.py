"""
Sequence Pattern Recognition Module
Provides functionality for recognizing patterns in sequences and lists
"""

from typing import List, Any, Optional, Tuple


class SequencePatternRecognizer:
    """
    A class for recognizing patterns in sequences of data.
    Supports arithmetic, geometric, and custom sequence patterns.
    """

    def __init__(self):
        """Initialize the SequencePatternRecognizer."""
        pass

    def is_arithmetic_sequence(self, sequence: List[float]) -> Tuple[bool, Optional[float]]:
        """
        Check if a sequence follows an arithmetic pattern.
        
        Args:
            sequence: List of numbers to check
            
        Returns:
            Tuple of (is_arithmetic, common_difference)
        """
        if len(sequence) < 2:
            return False, None

        diff = sequence[1] - sequence[0]
        for i in range(2, len(sequence)):
            if abs((sequence[i] - sequence[i - 1]) - diff) > 1e-9:
                return False, None
        return True, diff

    def is_geometric_sequence(self, sequence: List[float]) -> Tuple[bool, Optional[float]]:
        """
        Check if a sequence follows a geometric pattern.
        
        Args:
            sequence: List of numbers to check
            
        Returns:
            Tuple of (is_geometric, common_ratio)
        """
        if len(sequence) < 2:
            return False, None

        if sequence[0] == 0:
            return False, None

        ratio = sequence[1] / sequence[0]
        for i in range(2, len(sequence)):
            if sequence[i - 1] == 0:
                return False, None
            if abs((sequence[i] / sequence[i - 1]) - ratio) > 1e-9:
                return False, None
        return True, ratio

    def find_next_in_arithmetic(self, sequence: List[float]) -> Optional[float]:
        """
        Find the next number in an arithmetic sequence.
        
        Args:
            sequence: List of numbers in arithmetic progression
            
        Returns:
            Next number in sequence or None if not arithmetic
        """
        is_arith, diff = self.is_arithmetic_sequence(sequence)
        if is_arith and diff is not None:
            return sequence[-1] + diff
        return None

    def find_next_in_geometric(self, sequence: List[float]) -> Optional[float]:
        """
        Find the next number in a geometric sequence.
        
        Args:
            sequence: List of numbers in geometric progression
            
        Returns:
            Next number in sequence or None if not geometric
        """
        is_geom, ratio = self.is_geometric_sequence(sequence)
        if is_geom and ratio is not None:
            return sequence[-1] * ratio
        return None

    def detect_repeating_pattern(self, sequence: List[Any]) -> Optional[List[Any]]:
        """
        Detect a repeating pattern in a sequence.
        
        Args:
            sequence: List to analyze
            
        Returns:
            Repeating pattern or None if no pattern found
        """
        n = len(sequence)
        for pattern_length in range(1, n // 2 + 1):
            pattern = sequence[:pattern_length]
            is_repeating = True

            for i in range(pattern_length, n):
                if sequence[i] != pattern[i % pattern_length]:
                    is_repeating = False
                    break

            if is_repeating:
                return pattern

        return None

    def find_missing_number(self, sequence: List[int]) -> Optional[int]:
        """
        Find a missing number in a sequence (assumes one missing).
        
        Args:
            sequence: List of integers with one missing
            
        Returns:
            Missing number or None
        """
        if len(sequence) < 2:
            return None

        sorted_seq = sorted(sequence)
        expected_diff = (sorted_seq[-1] - sorted_seq[0]) // len(sorted_seq)

        for i in range(len(sorted_seq) - 1):
            actual_diff = sorted_seq[i + 1] - sorted_seq[i]
            if actual_diff != expected_diff:
                return sorted_seq[i] + expected_diff

        return None

    def longest_increasing_subsequence(self, sequence: List[float]) -> List[float]:
        """
        Find the longest increasing subsequence.
        
        Args:
            sequence: List of numbers
            
        Returns:
            Longest increasing subsequence
        """
        if not sequence:
            return []

        n = len(sequence)
        dp = [1] * n
        parent = [-1] * n

        for i in range(1, n):
            for j in range(i):
                if sequence[j] < sequence[i] and dp[j] + 1 > dp[i]:
                    dp[i] = dp[j] + 1
                    parent[i] = j

        max_length = max(dp)
        max_index = dp.index(max_length)

        # Reconstruct the subsequence
        result = []
        current = max_index
        while current != -1:
            result.append(sequence[current])
            current = parent[current]

        return list(reversed(result))

    def is_fibonacci_like(self, sequence: List[int]) -> bool:
        """
        Check if a sequence follows Fibonacci-like pattern (each number is sum of previous two).
        
        Args:
            sequence: List of integers to check
            
        Returns:
            True if Fibonacci-like, False otherwise
        """
        if len(sequence) < 3:
            return True

        for i in range(2, len(sequence)):
            if sequence[i] != sequence[i - 1] + sequence[i - 2]:
                return False

        return True

    def find_alternating_pattern(self, sequence: List[Any]) -> Optional[Tuple[List[Any], List[Any]]]:
        """
        Find an alternating pattern in a sequence.
        
        Args:
            sequence: List to analyze
            
        Returns:
            Tuple of (even_positions_pattern, odd_positions_pattern) or None
        """
        if len(sequence) < 2:
            return None

        even_positions = [sequence[i] for i in range(0, len(sequence), 2)]
        odd_positions = [sequence[i] for i in range(1, len(sequence), 2)]

        # Check if each subsequence has a pattern
        even_pattern = self.detect_repeating_pattern(even_positions)
        odd_pattern = self.detect_repeating_pattern(odd_positions)

        if even_pattern or odd_pattern:
            return (even_pattern or even_positions, odd_pattern or odd_positions)

        return None
