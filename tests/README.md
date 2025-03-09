# TunedIn Tests

This directory contains tests for the TunedIn music recommendation system.

## Running Tests

To run all tests:

```bash
python -m unittest discover tests
```

To run a specific test file:

```bash
python -m unittest tests.test_models
```

## Test Files

- `test_models.py`: Tests for the GNN models (GCN, GAT, LightGCN, GraphSAGE)

## Adding New Tests

When adding new tests, follow these guidelines:

1. Create a new test file with the prefix `test_`.
2. Extend `unittest.TestCase` for your test class.
3. Write test methods with the prefix `test_`.
4. Use descriptive method names that explain what is being tested.
5. Add appropriate assertions to verify expected behavior.

Example:

```python
import unittest

class TestExample(unittest.TestCase):
    def test_something(self):
        # Test code here
        self.assertEqual(1 + 1, 2)
```
