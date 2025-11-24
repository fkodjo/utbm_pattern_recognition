# VA52 Pattern Recognition - School TP Journey

A comprehensive Python library for pattern recognition and analysis. This project implements various pattern recognition algorithms for educational purposes as part of the VA52 school practical work (Travaux Pratiques).

## 🎯 Features

### String Pattern Matching
- Find and count pattern occurrences in text
- Regular expression matching
- Email and URL extraction
- Palindrome detection
- Repeating substring identification
- Levenshtein distance (edit distance) calculation

### Sequence Pattern Recognition
- Arithmetic sequence detection and prediction
- Geometric sequence detection and prediction
- Repeating pattern identification
- Fibonacci-like sequence detection
- Missing number detection
- Longest increasing subsequence
- Alternating pattern analysis

### Numeric Pattern Analysis
- Statistical analysis (mean, median, mode, standard deviation)
- Trend detection (increasing, decreasing, stable, fluctuating)
- Outlier detection using statistical methods
- Peak and valley detection
- Moving average calculation
- Frequency distribution analysis
- Cyclic pattern detection
- Data normalization and standardization

## 📁 Project Structure

```
A2025_VA52_TP_A/
├── src/                          # Source code
│   ├── __init__.py              # Package initialization
│   ├── string_patterns.py       # String pattern matching module
│   ├── sequence_patterns.py     # Sequence pattern recognition module
│   └── numeric_patterns.py      # Numeric pattern analysis module
├── tests/                        # Unit tests
│   ├── __init__.py
│   ├── test_string_patterns.py
│   ├── test_sequence_patterns.py
│   └── test_numeric_patterns.py
├── examples/                     # Example scripts
│   ├── string_pattern_examples.py
│   ├── sequence_pattern_examples.py
│   └── numeric_pattern_examples.py
├── requirements.txt              # Project dependencies
└── README.md                     # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- No external dependencies required (uses only Python standard library)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/fkodjo/A2025_VA52_TP_A.git
cd A2025_VA52_TP_A
```

2. (Optional) Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies (optional, no external dependencies):
```bash
pip install -r requirements.txt
```

## 📖 Usage

### String Pattern Matching

```python
from src.string_patterns import StringPatternMatcher

matcher = StringPatternMatcher()

# Find pattern occurrences
text = "hello world hello"
indices = matcher.find_pattern(text, "hello")
print(indices)  # [0, 12]

# Extract emails
text = "Contact us at info@example.com"
emails = matcher.extract_emails(text)
print(emails)  # ['info@example.com']

# Check palindrome
is_palindrome = matcher.is_palindrome("racecar")
print(is_palindrome)  # True

# Calculate edit distance
distance = matcher.levenshtein_distance("kitten", "sitting")
print(distance)  # 3
```

### Sequence Pattern Recognition

```python
from src.sequence_patterns import SequencePatternRecognizer

recognizer = SequencePatternRecognizer()

# Detect arithmetic sequence
sequence = [2, 4, 6, 8, 10]
is_arith, diff = recognizer.is_arithmetic_sequence(sequence)
print(f"Arithmetic: {is_arith}, Difference: {diff}")  # True, 2

# Find next number
next_num = recognizer.find_next_in_arithmetic(sequence)
print(next_num)  # 12

# Detect repeating pattern
sequence = [1, 2, 3, 1, 2, 3, 1, 2, 3]
pattern = recognizer.detect_repeating_pattern(sequence)
print(pattern)  # [1, 2, 3]

# Check Fibonacci-like sequence
sequence = [1, 1, 2, 3, 5, 8, 13]
is_fib = recognizer.is_fibonacci_like(sequence)
print(is_fib)  # True
```

### Numeric Pattern Analysis

```python
from src.numeric_patterns import NumericPatternAnalyzer

analyzer = NumericPatternAnalyzer()

# Calculate statistics
data = [1, 2, 3, 4, 5]
stats = analyzer.calculate_statistics(data)
print(stats)  # {'mean': 3.0, 'median': 3.0, ...}

# Detect trend
data = [1, 2, 3, 4, 5, 6]
trend = analyzer.detect_trend(data)
print(trend)  # 'increasing'

# Find outliers
data = [10, 12, 13, 12, 11, 13, 14, 100, 12, 11]
outliers = analyzer.find_outliers(data)
print(outliers)  # [(7, 100)]

# Detect peaks
data = [1, 3, 2, 5, 4, 6, 3]
peaks = analyzer.detect_peaks(data)
print(peaks)  # [1, 3, 5]
```

## 🧪 Running Tests

Run all unit tests:

```bash
python -m unittest discover tests
```

Run specific test module:

```bash
python -m unittest tests.test_string_patterns
python -m unittest tests.test_sequence_patterns
python -m unittest tests.test_numeric_patterns
```

Run with verbose output:

```bash
python -m unittest discover tests -v
```

## 📚 Examples

Run the example scripts to see the modules in action:

```bash
# String pattern examples
python examples/string_pattern_examples.py

# Sequence pattern examples
python examples/sequence_pattern_examples.py

# Numeric pattern examples
python examples/numeric_pattern_examples.py
```

## 🎓 Educational Purpose

This project is designed for educational purposes as part of the VA52 school TP (Travaux Pratiques). It demonstrates:

- Object-oriented programming in Python
- Algorithm implementation
- Pattern recognition techniques
- Statistical analysis
- Unit testing best practices
- Code documentation
- Project organization

## 🤝 Contributing

This is an educational project. Feel free to fork and experiment with the code!

## 📝 License

This project is created for educational purposes as part of VA52 school work.

## 👨‍💻 Author

VA52 School TP - Pattern Recognition Journey

## 🔗 Repository

[https://github.com/fkodjo/A2025_VA52_TP_A](https://github.com/fkodjo/A2025_VA52_TP_A)