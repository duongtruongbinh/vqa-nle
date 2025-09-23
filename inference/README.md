## Qwen 2.5 VL

```bash
pip install git+https://github.com/huggingface/transformers accelerate
```

## Molmo

```bash
pip install einops torchvision
```

Replace

```python
cache_name, cache = super()._extract_past_from_model_output(outputs)
model_kwargs[cache_name] = cache
```

with

```python
model_kwargs["past_key_values"] = outputs.past_key_values
```
