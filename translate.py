from jax import random
import jax.numpy as jnp
from data import make_dataset, SpecialTokens
from jax_transformer import make_transformer, load_state
import yaml

with open('config_de.yml', 'r') as f:
    config = yaml.safe_load(f)

dataset, text_transform = make_dataset()

config['src_vocab_size'] = len(dataset.vocab_transform['src'])
config['tgt_vocab_size'] = len(dataset.vocab_transform['tgt'])

transformer, state, generate_mask, pad_to_seq_len = make_transformer(config)
state, checkpoint_manager, save_args, step = load_state(config, state)
main_rng = random.PRNGKey(config['seed'])


def greedy_decode(state, transformer, rng, src_sentence, initial_input=""):
    src_item = pad_to_seq_len(text_transform['src'](src_sentence))
    src = jnp.array([src_item])

    tgt_item = pad_to_seq_len(text_transform['tgt'](initial_input))
    tgt_input = jnp.array([tgt_item])

    tgt_input = tgt_input.at[0, 1:].set(SpecialTokens.PAD)
    src_mask, tgt_mask = generate_mask(src, tgt_input)

    rng, dropout_apply_rng = random.split(rng)

    encoder_output = state.apply_fn(
        {'params': state.params},
        src=src, src_mask=src_mask,
        train=False, method=transformer.encode,
        rngs={'dropout': dropout_apply_rng})

    translation = []

    for length in range(1, config['seq_len']):
        rng, dropout_apply_rng = random.split(rng)

        tgt_output = state.apply_fn(
            {'params': state.params},
            encoder_output=encoder_output, tgt=tgt_input, src_mask=src_mask, tgt_mask=tgt_mask,
            train=False, method=transformer.decode,
            rngs={'dropout': dropout_apply_rng})
        tgt_output = tgt_output.argmax(axis=-1)

        new_word = tgt_output[0, length - 1]
        tgt_input = tgt_input.at[0, length].set(new_word)

        src_mask, tgt_mask = generate_mask(src, tgt_input)

        if new_word == SpecialTokens.EOS:
            break
        else:
            translation.append(new_word)

    return " ".join(dataset.vocab_transform['tgt'].lookup_tokens(translation))


main_rng, rng = random.split(main_rng)
while True:
    src_sentence = input("Enter German text to translate or leave blank to exit:\n")
    if not src_sentence:
        break
    print('Translation:', greedy_decode(state, transformer, rng, src_sentence), '\n')