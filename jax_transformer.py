import numpy as np
import math
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from jax import random
from flax.training import train_state
from data import SpecialTokens
import os
import orbax
from flax.training import orbax_utils
import torch


def dotprod_attention(q, k, v, mask=None):
    dot_prod = (q @ jnp.swapaxes(k, -2, -1)) / \
        (k.shape[-1]**0.5)  # (b, h, sl, sl)
    if mask is not None:
        dot_prod = jnp.where(mask == 0, -9e15, dot_prod)
    attention = jax.nn.softmax(dot_prod, axis=-1)
    values = attention @ v
    return values, attention


class MultiheadSelfAttention(nn.Module):
    dim: int  # d
    num_heads: int  # h

    def setup(self):
        self.qkv_layer = nn.Dense(features=3*self.dim,
                                  kernel_init=nn.initializers.xavier_uniform(),
                                  bias_init=nn.initializers.zeros)
        self.output_layer = nn.Dense(features=self.dim,
                                     kernel_init=nn.initializers.xavier_uniform(),
                                     bias_init=nn.initializers.zeros)

    def __call__(self, x, mask=None):
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        # x: (b, sl, dx)
        qkv = self.qkv_layer(x)  # (b, sl, 3d)
        qkv = qkv.reshape(batch_size, seq_len,
                          self.num_heads, -1)  # (b, sl, h, 3d/h)
        qkv = qkv.transpose(0, 2, 1, 3)  # (b, h, sl, 3d/h)
        q, k, v = jnp.array_split(qkv, 3, axis=-1)  # (b, h, sl, d/h)

        values, attention = dotprod_attention(
            q, k, v, mask=mask)  # values: (b, h, sl, d/h)
        values = values.transpose(0, 2, 1, 3)  # (b, sl, h, d/h)
        values = values.reshape(batch_size, seq_len, self.dim)
        out = self.output_layer(values)

        return out, attention


class MultiheadAttention(nn.Module):
    dim: int  # d
    num_heads: int  # h

    def setup(self):
        self.q_layer = nn.Dense(features=self.dim,
                                kernel_init=nn.initializers.xavier_uniform(),
                                bias_init=nn.initializers.zeros)
        self.k_layer = nn.Dense(features=self.dim,
                                kernel_init=nn.initializers.xavier_uniform(),
                                bias_init=nn.initializers.zeros)
        self.v_layer = nn.Dense(features=self.dim,
                                kernel_init=nn.initializers.xavier_uniform(),
                                bias_init=nn.initializers.zeros)
        self.output_layer = nn.Dense(features=self.dim,
                                     kernel_init=nn.initializers.xavier_uniform(),
                                     bias_init=nn.initializers.zeros)

    def __call__(self, q, k, v, mask=None):
        batch_size = q.shape[0]
        seq_len = q.shape[1]

        # x: (b, sl, dx)
        q = self.q_layer(q)  # (b, sl, d)
        k = self.k_layer(k)  # (b, sl, d)
        v = self.v_layer(v)  # (b, sl, d)
        qkv = jnp.concatenate([q, k, v], axis=2)  # (b, sl, 3d)
        qkv = qkv.reshape((batch_size, seq_len,
                          self.num_heads, -1), order='F')  # (b, sl, h, 3d/h)
        qkv = qkv.transpose(0, 2, 1, 3)  # (b, h, sl, 3d/h)
        q, k, v = jnp.array_split(qkv, 3, axis=-1)  # (b, h, sl, d/h)

        values, attention = dotprod_attention(
            q, k, v, mask=mask)  # values: (b, h, sl, d/h)
        values = values.transpose(0, 2, 1, 3)  # (b, sl, h, d/h)
        values = values.reshape(batch_size, seq_len, self.dim)
        out = self.output_layer(values)

        return out, attention


class EncoderBlock(nn.Module):
    dim: int
    num_heads: int
    feed_forward_dim: int
    dropout_prob: float

    def setup(self):
        self.mh_attn = MultiheadSelfAttention(
            dim=self.dim, num_heads=self.num_heads)
        self.norm1 = nn.LayerNorm()

        self.feed_forward = [
            nn.Dense(self.feed_forward_dim),
            nn.Dropout(self.dropout_prob),
            nn.relu,
            nn.Dense(self.dim)
        ]
        self.norm2 = nn.LayerNorm()
        self.dropout = nn.Dropout(self.dropout_prob)

    def __call__(self, x, mask=None, train=True):
        attn_out, _ = self.mh_attn(x, mask=mask)
        x = x + self.dropout(attn_out, deterministic=not train)
        x = self.norm1(x)

        feed_forward_out = x
        for layer in self.feed_forward:
            feed_forward_out = layer(feed_forward_out) if not isinstance(layer, nn.Dropout) \
                else layer(feed_forward_out, deterministic=not train)
        x = x + self.dropout(feed_forward_out, deterministic=not train)
        x = self.norm2(x)

        return x


class Encoder(nn.Module):
    num_blocks: int
    dim: int
    num_heads: int
    feed_forward_dim: int
    dropout_prob: float

    def setup(self):
        self.blocks = [EncoderBlock(dim=self.dim, num_heads=self.num_heads,
                                    feed_forward_dim=self.feed_forward_dim, dropout_prob=self.dropout_prob)
                       for _ in range(self.num_blocks)]

    def __call__(self, x, mask=None, train=True):
        for block in self.blocks:
            x = block(x, mask=mask, train=train)
        return x


class DecoderBlock(nn.Module):
    dim: int
    num_heads: int
    feed_forward_dim: int
    dropout_prob: float

    def setup(self):
        self.m_mh_attn = MultiheadSelfAttention(
            dim=self.dim, num_heads=self.num_heads)
        self.norm1 = nn.LayerNorm()

        self.e_mh_attn = MultiheadAttention(
            dim=self.dim, num_heads=self.num_heads)
        self.norm2 = nn.LayerNorm()

        self.feed_forward = [
            nn.Dense(self.feed_forward_dim),
            nn.Dropout(self.dropout_prob),
            nn.relu,
            nn.Dense(self.dim)
        ]
        self.norm3 = nn.LayerNorm()
        self.dropout = nn.Dropout(self.dropout_prob)

    def __call__(self, x, encoder_output, src_mask=None, tgt_mask=None, train=True):
        attn_out, _ = self.m_mh_attn(x, mask=tgt_mask)
        x = x + self.dropout(attn_out, deterministic=not train)
        x = self.norm1(x)

        attn_out, _ = self.e_mh_attn(x, encoder_output, encoder_output, mask=src_mask)
        x = x + self.dropout(attn_out, deterministic=not train)
        x = self.norm2(x)

        feed_forward_out = x
        for layer in self.feed_forward:
            feed_forward_out = layer(feed_forward_out) if not isinstance(layer, nn.Dropout) \
                else layer(feed_forward_out, deterministic=not train)
        x = x + self.dropout(feed_forward_out, deterministic=not train)
        x = self.norm3(x)

        return x


class Decoder(nn.Module):
    num_blocks: int
    dim: int
    num_heads: int
    feed_forward_dim: int
    dropout_prob: float

    def setup(self):
        self.blocks = [DecoderBlock(dim=self.dim, num_heads=self.num_heads,
                                    feed_forward_dim=self.feed_forward_dim, dropout_prob=self.dropout_prob)
                       for _ in range(self.num_blocks)]

    def __call__(self, x, encoder_output, src_mask=None, tgt_mask=None, train=True):
        for block in self.blocks:
            x = block(x, encoder_output, src_mask=src_mask, tgt_mask=tgt_mask, train=train)
        return x


class PositionalEncoding(nn.Module):
    dim: int
    max_len: int = 500

    def setup(self):
        pe = np.zeros((self.max_len, self.dim))
        pos = np.arange(self.max_len)
        den = 1/np.exp(np.arange(0, self.dim, 2) * (-math.log(10000.0) / self.dim))
        pe[:, ::2] = np.sin(np.expand_dims(pos, axis=1) / np.expand_dims(den, axis=0))
        pe[:, 1::2] = np.cos(np.expand_dims(pos, axis=1) / np.expand_dims(den, axis=0))
        pe = np.expand_dims(pe, axis=0)
        self.pe = jax.device_put(pe)

    def __call__(self, x):
        x = x + self.pe[:, :x.shape[1]]
        return x


class Transformer(nn.Module):
    src_vocab_size: int
    tgt_vocab_size: int
    num_blocks: int
    dim: int
    num_heads: int
    feed_forward_dim: int
    dropout_prob: float

    def setup(self):
        self.src_emb = nn.Embed(num_embeddings=self.src_vocab_size, features=self.dim)
        self.tgt_emb = nn.Embed(num_embeddings=self.tgt_vocab_size, features=self.dim)
        self.src_pos = PositionalEncoding(dim=self.dim)
        self.tgt_pos = PositionalEncoding(dim=self.dim)

        self.encoder = Encoder(
            num_blocks=self.num_blocks,
            dim=self.dim,
            num_heads=self.num_heads,
            feed_forward_dim=self.feed_forward_dim,
            dropout_prob=self.dropout_prob)

        self.decoder = Decoder(
            num_blocks=self.num_blocks,
            dim=self.dim,
            num_heads=self.num_heads,
            feed_forward_dim=self.feed_forward_dim,
            dropout_prob=self.dropout_prob)

        self.output_net = [
            nn.Dense(self.dim),
            nn.LayerNorm(),
            nn.relu,
            nn.Dropout(self.dropout_prob),
            nn.Dense(self.tgt_vocab_size)
        ]

    def encode(self, src, src_mask=None, train=True):
        src = self.src_emb(src)
        src = self.src_pos(src)
        src = self.encoder(src, mask=src_mask, train=train)
        return src

    def decode(self, encoder_output, tgt, src_mask=None, tgt_mask=None, train=True):
        tgt = self.tgt_emb(tgt)
        tgt = self.tgt_pos(tgt)
        tgt = self.decoder(tgt, encoder_output, src_mask=src_mask, tgt_mask=tgt_mask, train=train)
        return tgt

    def project(self, x, train=True):
        for layer in self.output_net:
            x = layer(x) if not isinstance(layer, nn.Dropout) else layer(x, deterministic=not train)
        return x

    def __call__(self, src=None, tgt=None, src_mask=None, tgt_mask=None, 
                 train=True, encoder_output=None, encode=True, decode=True):
        assert encode or decode
        if not decode:
            out = self.encode(src, src_mask, train=train)
        elif not encode:
            decoder_output = self.decode(encoder_output, tgt, src_mask, tgt_mask, train=train)
            out = self.project(decoder_output, train=train)
        else:
            encoder_output = self.encode(src, src_mask, train=train)
            decoder_output = self.decode(encoder_output, tgt, src_mask, tgt_mask, train=train)
            out = self.project(decoder_output, train=train)
        return out


def make_transformer(config):
    transformer = Transformer(src_vocab_size=config['src_vocab_size'],
                              tgt_vocab_size=config['tgt_vocab_size'],
                              num_blocks=config['num_blocks'],
                              dim=config['emb_size'],
                              num_heads=config['num_heads'],
                              feed_forward_dim=config['ffn_hid_dim'],
                              dropout_prob=config['dropout_prob'])

    main_rng = random.PRNGKey(config['seed'])
    main_rng, rng = random.split(main_rng)
    src = random.randint(rng, (1, config['seq_len']), 0, config['src_vocab_size'])
    main_rng, rng = random.split(main_rng)
    tgt = random.randint(rng, (1, config['seq_len']), 0, config['tgt_vocab_size'])

    main_rng, init_rng, dropout_init_rng = random.split(main_rng, 3)
    params = transformer.init(
        {'params': init_rng, 'dropout': dropout_init_rng}, src, tgt, train=True)['params']

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(1e-4)
    )

    state = train_state.TrainState.create(apply_fn=transformer.apply, params=params, tx=optimizer)

    # 0, 1. (0 means masked)
    # (sl) -> (sl, sl)
    def generate_mask(src, tgt):

        src_mask = (src != SpecialTokens.PAD)[:, None]  # (sl, 1)
        src_mask = src_mask @ src_mask.transpose(1, 0)  # (sl, sl)

        tgt_padding_mask = (tgt != SpecialTokens.PAD)[:, None]  # (sl, 1)
        tgt_padding_mask = tgt_padding_mask @ tgt_padding_mask.transpose(1, 0)  # (sl, sl)

        tgt_causal_mask = (jnp.triu(jnp.ones((config['seq_len'], config['seq_len'])))).transpose(1, 0) > 0.5  # (sl, sl)
        tgt_mask = tgt_padding_mask & tgt_causal_mask

        src_mask = src_mask[None, :, :].repeat(config['num_heads'], axis=0)
        tgt_mask = tgt_mask[None, :, :].repeat(config['num_heads'], axis=0)

        return src_mask, tgt_mask

    generate_mask = jax.vmap(jax.jit(generate_mask))

    def pad_to_seq_len(item):
        pad_len = config['seq_len'] - item.shape[0]
        assert pad_len > 0, f"sentence too long {item.shape[0]} > {config['seq_len']}"
        return torch.cat((item, torch.Tensor([SpecialTokens.PAD] * pad_len))).int()

    return transformer, state, generate_mask, pad_to_seq_len


def load_state(config, state):
    step = 1
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(state)
    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=1, create=True)
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        os.path.join(config['ckpt_dir'], 'managed'), orbax_checkpointer, options)

    try:
        step = checkpoint_manager.latest_step()
        state = checkpoint_manager.restore(step, items=state)
        print(f'Loaded checkpoint from epoch {step}')
    except Exception as e:
        step = 1
        print('Could not load checkpoint. Training from scratch.')
        print(e)

    return state, checkpoint_manager, save_args, step