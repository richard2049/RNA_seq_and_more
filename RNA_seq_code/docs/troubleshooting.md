# Troubleshooting

This doc is written to match the current repo state: a single Python script using Scanpy + scvi-tools
and some manual steps. It focuses on two pain points: **manual annotation**
and **data size / matrix blow-ups**.

---

## 1) GPU vs CPU (scvi-tools training)

### Symptom
You run the script and get errors like:
- `RuntimeError: CUDA error...`
- `AssertionError: Torch not compiled with CUDA enabled`
- `ValueError: You requested accelerator='gpu' but no GPUs are available`

### Cause
The script currently does:
```python
model.train(accelerator="gpu", devices=1)
