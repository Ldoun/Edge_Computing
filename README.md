- Pruning

Dense Model training

```python main.py --pruning_ratio 0.0 --scheduler cosine --lr 0.001```

Sparse Model training

```python main.py --pruning_ratio 0.98 --scheduler cosine --lr 0.001 --dense_model result/Dense1 --prune_type structured```