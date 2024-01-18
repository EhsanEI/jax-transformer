import jax
from jax import random
import jax.numpy as jnp
from data import make_dataset, SpecialTokens
from tqdm import tqdm
from jax_transformer import make_transformer, load_state
from torch.utils.data import DataLoader
import optax
from timeit import default_timer as timer
import yaml

with open('config_de.yml', 'r') as f:
    config = yaml.safe_load(f)

dataset, text_transform = make_dataset()

config['src_vocab_size'] = len(dataset.vocab_transform['src'])
config['tgt_vocab_size'] = len(dataset.vocab_transform['tgt'])

transformer, state, generate_mask, pad_to_seq_len = make_transformer(config)
state, checkpoint_manager, save_args, step = load_state(config, state)
main_rng = random.PRNGKey(config['seed'])


def collate_fn(batch):
    src_batch, tgt_batch, label_batch = [], [], []
    for src_sample, tgt_sample in batch:
        src_item = pad_to_seq_len(text_transform['src'](src_sample.rstrip("\n")))
        tgt_item = pad_to_seq_len(text_transform['tgt'](tgt_sample.rstrip("\n")))
        label_item = pad_to_seq_len(text_transform['tgt'](tgt_sample.rstrip("\n"))[1:])
        src_batch.append(src_item)
        tgt_batch.append(tgt_item)
        label_batch.append(label_item)

    src_batch = jnp.array(src_batch)
    tgt_batch = jnp.array(tgt_batch)
    label_batch = jnp.array(label_batch)

    return src_batch, tgt_batch, label_batch


def calculate_loss(params, apply_fn, rng, batch, train=True):
    src, tgt, src_mask, tgt_mask, label = batch
    rng, dropout_apply_rng = random.split(rng)
    logits = apply_fn(
        {'params': params},
        src=src, tgt=tgt, src_mask=src_mask, tgt_mask=tgt_mask,
        train=train, 
        rngs={'dropout': dropout_apply_rng})

    loss = optax.softmax_cross_entropy_with_integer_labels(logits, label)
    loss = jnp.where(label == SpecialTokens.PAD, 0., loss)
    loss = loss.mean()
    return loss, rng


@jax.jit
def train_step(state, rng, batch):
    def jax_loss_fn(params):
        return calculate_loss(params, state.apply_fn, rng, batch, train=True)

    ret, grads = jax.value_and_grad(jax_loss_fn, has_aux=True)(state.params)
    loss, rng = ret[0], ret[1]
    state = state.apply_gradients(grads=grads)
    return loss, state, rng


@jax.jit
def eval_step(state, rng, batch):
    loss, rng = calculate_loss(state.params, state.apply_fn, rng, batch, train=False)
    return loss, rng


def train_epoch(state, rng):
    losses = 0
    train_iter = dataset.get_pair_iter('train')
    train_dataloader = DataLoader(
        list(train_iter), batch_size=config['batch_size'], collate_fn=collate_fn)

    for src, tgt_input, tgt_out in tqdm(train_dataloader):

        src_mask, tgt_mask = generate_mask(src, tgt_input)

        batch = (src, tgt_input,
                 src_mask, tgt_mask, tgt_out)

        loss, state, rng = train_step(state, rng, batch)
        losses += loss.item()
    return losses / len(list(train_dataloader)), state, rng


def eval_epoch(state, rng):
    losses = 0
    val_iter = dataset.get_pair_iter('val')
    val_dataloader = DataLoader(
        list(val_iter), batch_size=config['batch_size'], collate_fn=collate_fn)

    for src, tgt_input, tgt_out in tqdm(val_dataloader):

        src_mask, tgt_mask = generate_mask(src, tgt_input)

        batch = (src, tgt_input,
                 src_mask, tgt_mask, tgt_out)

        loss, rng = eval_step(state, rng, batch)
        losses += loss.item()
    return losses / len(list(val_dataloader)), rng


main_rng, rng = random.split(main_rng)
for epoch in range(step, config['num_epochs'] + step):
    start_time = timer()
    train_loss, state, rng = train_epoch(state, rng)
    end_time = timer()
    val_loss, rng = eval_epoch(state, rng)
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
    checkpoint_manager.save(epoch, items=state, save_kwargs={'save_args': save_args})
