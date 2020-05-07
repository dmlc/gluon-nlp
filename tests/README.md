# Unit Tests

To run the unittests, use the following command

```bash
pytest .
```

To test for certain file, e.g., the `test_models_transformer.py`, use the following command

```bash
pytest test_models_transformer
```

Refer to the [official guide of pytest](https://docs.pytest.org/en/latest/) for more details.

# Naming Convention
The naming convention of the tests are `test_{directory_name}_{module_name}.py`. 
For example, the test of [models/transformer.py](../src/gluonnlp/models/transformer.py) will be in 
`test_models_transformer.py`.
