import numpy as np

from t1_reference_common import (Safetensors, bf16_round_f32, bf16_to_f32,
                                 define_float, define_uint, f32_to_bf16_u16,
                                 rmsnorm, sigmoid)

PREFIX = "model.language_model.layers."
EMBED = "model.language_model.embed_tokens.weight"
FINAL_NORM = "model.language_model.norm.weight"
LM_HEAD = "lm_head.weight"
ATTENTION_SCALE = 0.08838834764831845
LINEAR_BLOCK_ROWS = 4096

DEFINES_VS_CONFIG = [
    ("HIDDEN_DIMENSION", "hidden_size", "uint"),
    ("LAYER_COUNT", "num_hidden_layers", "uint"),
    ("OUTPUT_VOCAB_COUNT", "vocab_size", "uint"),
    ("RMS_NORM_EPSILON", "rms_norm_eps", "float"),
    ("POST_NORM_EPSILON", "post_norm_eps", "float"),
    ("INTERMEDIATE_DIMENSION", "intermediate_size", "uint"),
    ("ATTENTION_HEAD_COUNT", "num_attention_heads", "uint"),
    ("KV_HEAD_COUNT", "num_key_value_heads", "uint"),
    ("HEAD_DIMENSION", "head_dim", "uint"),
    ("SLIDING_WINDOW", "sliding_window", "uint"),
    ("QK_SCALE_FACTOR", "qk_scale_factor", "float"),
    ("OUTPUT_MULTIPLIER", "output_multiplier", "float"),
    ("FINAL_LOGIT_SOFTCAP", "final_logit_softcapping", "float"),
    ("END_OF_TEXT_TOKEN_ID", "eos_token_id", "uint"),
]
DEFINES_RECORDED_ONLY = [
    ("MAXIMUM_CONTEXT_TOKENS", "max_position_embeddings", "uint"),
    ("BOS_TOKEN_ID", "bos_token_id", "uint"),
]


class MuseConfigError(ValueError):
    pass


def define_of(kind, defines, dname):
    return define_uint(defines, dname) if kind == "uint" \
        else define_float(defines, dname)


def scalar_of(kind, value):
    got = value[0] if isinstance(value, list) else value
    return float(got)


def cross_check(defines, config):
    mismatches = []
    for dname, cname, kind in DEFINES_VS_CONFIG + DEFINES_RECORDED_ONLY:
        if cname not in config:
            raise MuseConfigError(f"config key {cname} missing for "
                                  f"SPARK_LLM_{dname}")
        want = define_of(kind, defines, dname)
        got = scalar_of(kind, config[cname])
        if abs(want - got) > 0:
            mismatches.append({"define": f"SPARK_LLM_{dname}", "value": want,
                               "config_key": cname, "config_value": got})
    for dname, _, _ in DEFINES_VS_CONFIG:
        for row in mismatches:
            if row["define"] == f"SPARK_LLM_{dname}":
                raise MuseConfigError(
                    f"{row['define']}={row['value']} disagrees with "
                    f"{row['config_key']}={row['config_value']}")
    return mismatches


class MuseEngine:
    def __init__(self, checkpoint_dir, defines, config):
        self.mismatches = cross_check(defines, config)
        self.st = Safetensors(checkpoint_dir)
        self.hidden = int(config["hidden_size"])
        self.layers = int(config["num_hidden_layers"])
        self.vocab = int(config["vocab_size"])
        self.eps = float(config["rms_norm_eps"])
        self.post_eps = float(config["post_norm_eps"])
        self.intermediate = int(config["intermediate_size"])
        self.heads = int(config["num_attention_heads"])
        self.kv_heads = int(config["num_key_value_heads"])
        self.head_dim = int(config["head_dim"])
        self.window = int(config["sliding_window"])
        self.theta = float(config["rope_parameters"]["rope_theta"])
        self.qk_scale = float(config["qk_scale_factor"])
        self.output_multiplier = float(config["output_multiplier"])
        self.softcap = float(config["final_logit_softcapping"])
        self.layer_types = list(config["layer_types"])
        self.layer_rope_theta = [float(t) for t in config["layer_rope_theta"]]
        self.eot = int(config["eos_token_id"][0] if isinstance(
            config["eos_token_id"], list) else config["eos_token_id"])
        self.attn_group = self.heads // self.kv_heads
        half = self.head_dim // 2
        exponents = np.arange(0, self.head_dim, 2, dtype=np.float32) \
            / np.float32(self.head_dim)
        self.rope_inv_freq = np.float32(1.0) \
            / (np.float32(self.theta) ** exponents)
        self.rope_pairs = half
        if len(self.layer_types) != self.layers \
                or len(self.layer_rope_theta) != self.layers:
            raise MuseConfigError("layer type lists disagree with layer count")
        fulls = [i for i, kind in enumerate(self.layer_types)
                 if kind == "full_attention"]
        slidings = [i for i, kind in enumerate(self.layer_types)
                    if kind == "sliding_attention"]
        if fulls != list(range(3, self.layers, 4)) \
                or len(slidings) != self.layers - len(fulls):
            raise MuseConfigError("layer_types disagree with the period-4 "
                                  "full-attention phase-3 pattern")
        for i, theta in enumerate(self.layer_rope_theta):
            if (theta == 0.0) != (self.layer_types[i] == "full_attention"):
                raise MuseConfigError(
                    f"layer_rope_theta[{i}]={theta} disagrees with "
                    f"layer_types[{i}]={self.layer_types[i]}")
            if theta != 0.0 and theta != self.theta:
                raise MuseConfigError(
                    f"layer_rope_theta[{i}]={theta} disagrees with "
                    f"rope_parameters.rope_theta={self.theta}")
        self.check_shape("q_proj", self.heads * self.head_dim)
        self.check_shape("k_proj", self.kv_heads * self.head_dim)
        self.check_shape("v_proj", self.kv_heads * self.head_dim)
        self.check_shape("gate_proj", self.heads * self.head_dim)
        embed = self.st.entry(EMBED)
        if embed["shape"] != [self.vocab, self.hidden]:
            raise MuseConfigError(f"embed shape {embed['shape']} disagrees "
                                  f"with vocab {self.vocab} hidden "
                                  f"{self.hidden}")
        head = self.st.entry(LM_HEAD)
        if head["shape"] != [self.vocab, self.hidden]:
            raise MuseConfigError(f"lm_head shape {head['shape']} disagrees "
                                  f"with vocab {self.vocab} hidden "
                                  f"{self.hidden}")

    def check_shape(self, name, rows):
        entry = self.st.entry(f"{PREFIX}0.self_attn.{name}.weight")
        if entry["shape"][0] != rows:
            raise MuseConfigError(
                f"{PREFIX}0.self_attn.{name}.weight rows "
                f"{entry['shape'][0]} disagree with {rows}")

    def tensor(self, name):
        raw = self.st.raw(name)
        if raw.dtype == np.uint16:
            return bf16_to_f32(raw)
        raise ValueError(f"reference tensor {name} must be BF16")

    def linear(self, x, name):
        weight = name + ".weight"
        shape = self.st.entry(weight)["shape"]
        rows = shape[0]
        if shape[1] != x.shape[0]:
            raise ValueError(f"weight {weight} shape {shape} disagrees with "
                             f"activation width {x.shape[0]}")
        out = np.empty(rows, dtype=np.float32)
        for start in range(0, rows, LINEAR_BLOCK_ROWS):
            count = min(LINEAR_BLOCK_ROWS, rows - start)
            slab = self.st.raw_rows(weight, start, count)
            out[start:start + count] = bf16_to_f32(slab) @ x
        return bf16_round_f32(out)

    def embed(self, token_id):
        if token_id < 0 or token_id >= self.vocab:
            raise ValueError(f"token {token_id} outside vocabulary {self.vocab}")
        raw = self.st.raw_rows(EMBED, token_id, 1)
        if raw.dtype != np.uint16:
            raise ValueError("reference embedding must be BF16")
        row = bf16_to_f32(raw[0])
        return bf16_round_f32(scaleless_rms(row, self.eps))

    def centered_norm(self, x, name, epsilon):
        weight = self.tensor(name)
        normed = scaleless_rms(x, epsilon)
        return bf16_round_f32(normed * (np.float32(1.0) + weight))

    def rope_cos_sin(self, position):
        angles = np.float32(position) * self.rope_inv_freq
        cos = bf16_round_f32(np.cos(angles))
        sin = bf16_round_f32(np.sin(angles))
        return np.concatenate([cos, cos]), np.concatenate([sin, sin])

    def rotate_half(self, rows):
        half = self.rope_pairs
        return np.concatenate([-rows[:, half:], rows[:, :half]], axis=1)

    def apply_rope(self, rows, position):
        cos, sin = self.rope_cos_sin(position)
        turned = bf16_round_f32(self.rotate_half(rows))
        left = bf16_round_f32(rows * cos[None, :])
        right = bf16_round_f32(turned * sin[None, :])
        return bf16_round_f32(left + right)

    def attention(self, prefix, x, cache, index, position):
        q = self.linear(x, prefix + "self_attn.q_proj").reshape(
            self.heads, self.head_dim)
        k = self.linear(x, prefix + "self_attn.k_proj").reshape(
            self.kv_heads, self.head_dim)
        v = self.linear(x, prefix + "self_attn.v_proj").reshape(
            self.kv_heads, self.head_dim)
        q = bf16_round_f32(scaleless_rms(q, self.eps))
        k = bf16_round_f32(scaleless_rms(k, self.eps))
        q = bf16_round_f32(q * np.float32(self.qk_scale))
        if self.layer_types[index] == "sliding_attention":
            q = self.apply_rope(q, position)
            k = self.apply_rope(k, position)
        elif self.layer_types[index] != "full_attention":
            raise ValueError(f"unsupported layer type "
                             f"{self.layer_types[index]}")
        cache.append((f32_to_bf16_u16(k.reshape(-1)),
                      f32_to_bf16_u16(v.reshape(-1))))
        first = 0 if self.layer_types[index] == "full_attention" \
            else max(0, position - self.window + 1)
        keys = np.stack([bf16_to_f32(entry[0]).reshape(self.kv_heads,
                                                       self.head_dim)
                         for entry in cache[first:]])
        values = np.stack([bf16_to_f32(entry[1]).reshape(self.kv_heads,
                                                         self.head_dim)
                           for entry in cache[first:]])
        out = np.empty((self.heads, self.head_dim), dtype=np.float32)
        for h in range(self.heads):
            kvh = h // self.attn_group
            scores = (keys[:, kvh, :] @ q[h]) * np.float32(ATTENTION_SCALE)
            peak = scores.max()
            weights = np.exp(scores - peak)
            weights = weights / weights.sum()
            out[h] = weights @ values[:, kvh, :]
        gated_attn = bf16_round_f32(out.reshape(-1))
        gate = self.linear(x, prefix + "self_attn.gate_proj")
        gate = bf16_round_f32(sigmoid(gate))
        gated = bf16_round_f32(gated_attn * gate)
        return self.linear(gated, prefix + "self_attn.o_proj")

    def mlp(self, prefix, x):
        gate = self.linear(x, prefix + "mlp.gate_proj")
        up = self.linear(x, prefix + "mlp.up_proj")
        activated = bf16_round_f32(gate * sigmoid(gate))
        activated = bf16_round_f32(activated * up)
        return self.linear(activated, prefix + "mlp.down_proj")

    def forward_layer(self, index, streams, caches, position):
        prefix = PREFIX + str(index) + "."
        x = self.centered_norm(streams, prefix + "input_layernorm.weight",
                               self.eps)
        attention = self.attention(prefix, x, caches[index], index, position)
        normed = self.centered_norm(attention,
                                    prefix + "post_attention_layernorm.weight",
                                    self.post_eps)
        streams = bf16_round_f32(streams + normed)
        x = self.centered_norm(streams,
                               prefix + "pre_feedforward_layernorm.weight",
                               self.eps)
        normed = self.centered_norm(self.mlp(prefix, x),
                                    prefix + "post_feedforward_layernorm.weight",
                                    self.post_eps)
        return bf16_round_f32(streams + normed)

    def decode_step(self, token_id, position, states, caches, capture,
                    capture_streams=None):
        del states, capture
        if token_id < 0 or token_id >= self.vocab:
            raise ValueError(f"token {token_id} outside vocabulary {self.vocab}")
        if position == 0:
            for i in range(self.layers):
                caches[i] = []
        streams = self.embed(token_id)
        for i in range(self.layers):
            streams = self.forward_layer(i, streams, caches, position)
            if capture_streams is not None:
                capture_streams(i, streams)
            if not np.isfinite(streams).all():
                raise ValueError(f"nonfinite reference state at layer {i}")
        return streams

    def softcapped(self, raw):
        scaled = bf16_round_f32(raw * np.float32(self.output_multiplier))
        divided = bf16_round_f32(scaled / np.float32(self.softcap))
        tamed = bf16_round_f32(np.tanh(divided))
        return float(bf16_round_f32(tamed * np.float32(self.softcap)))

    def logits(self, streams, chunk=LINEAR_BLOCK_ROWS):
        norm = bf16_round_f32(rmsnorm(
            streams, self.tensor(FINAL_NORM), self.eps))
        head = self.st.entry(LM_HEAD)
        if head["shape"] != [self.vocab, self.hidden]:
            raise MuseConfigError(f"lm_head shape {head['shape']} disagrees "
                                  f"with vocab {self.vocab} hidden "
                                  f"{self.hidden}")
        best = -np.inf
        best_token = -1
        for start in range(0, self.vocab, chunk):
            count = min(chunk, self.vocab - start)
            slab = self.st.raw_rows(LM_HEAD, start, count)
            scores = bf16_to_f32(slab) @ norm
            i = int(np.argmax(scores))
            if float(scores[i]) > best:
                best = float(scores[i])
                best_token = start + i
        return best_token, self.softcapped(best)


def scaleless_rms(x, epsilon):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True)
                       + np.float32(epsilon))


ENGINE_CLASS = MuseEngine
