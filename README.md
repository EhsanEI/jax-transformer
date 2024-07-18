# jax-transformer
German to English translation with Seq2Seq Transformer [1] in JAX.

## Usage

1. Install the requirements.
2. Set the checkpoint directory in the config file.
3. Run train.py to train a model and save a checkpoint.
4. Run translate.py to load the checkpoint for translation.

## Notes

1. The code is tested on Python 3.11.
2. This is an educational repo. The dataset is small and the translation quality is not great.
3. The encoder part loosely follows [2] but is not the same code and is not guaranteed to produce the same results.

## References

[1] Vaswani, Ashish, et al. "Attention is all you need." NeurIPS, (2017).

[2] https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial6/Transformers_and_MHAttention.html
