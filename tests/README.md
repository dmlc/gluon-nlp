# Unit Tests

To run the unittests, use the following command

```bash
python3 -m pytest --device="cpu" .
```

To test for certain file, e.g., the `test_models_transformer.py`, use the following command

```bash
python3 -m pytest --device="cpu" test_models_transformer.py
```

To test only for gpu device, use the following command

```bash
python3 -m pytest --device="gpu" test_models_transformer.py
```

To test both for cpu and gpu device, use the following command

```bash
python3 -m pytest --device="cpu" --device="gpu" test_models_transformer.py
```

In addition, to run all the tests, you should add the `--runslow` flag

```bash
python3 -m pytest --device="gpu" --runslow test_models.py
```

Refer to the [official guide of pytest](https://docs.pytest.org/en/latest/) for more details.

# Naming Convention

The naming convention of the tests are `test_{module_name}.py`. 
For example, the test of [models/transformer.py](../src/gluonnlp/models/transformer.py) will be in 
`test_models_transformer.py`. The test of [models/__init__.py](../src/gluonnlp/models/__init__.py) 
is `test_models.py`. 

Also, we include the scheduled testing scripts for `nlp_process` in [processing](processing), 
and 'nlp_data' in [data](data).

