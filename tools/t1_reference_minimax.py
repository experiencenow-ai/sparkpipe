import numpy as np

from t1_reference_common import (Safetensors, bf16_round_f32, bf16_to_f32,
                                 define_float, define_uint, f32_to_bf16_u16,
                                 rmsnorm, sigmoid)

PREFIX = "model.language_model.layers."
EMBED_NAME = "model.language_model.embed_tokens.weight"
FINAL_NORM_NAME = "model.language_model.norm.weight"
LM_HEAD_NAME = "lm_head.weight"


class MiniMaxConfigError(ValueError):
    pass


def swish(x):
    return x * sigmoid(x)


def cross_check(defines, config):
    checks = [
        ("HIDDEN_DIMENSION", "hidden_size", "uint"),
        ("LAYER_COUNT", "num_hidden_layers", "uint"),
        ("OUTPUT_VOCAB_COUNT", "vocab_size", "uint"),
        ("RMS_NORM_EPSILON", "rms_norm_eps", "float"),
        ("ATTENTION_HEAD_COUNT", "num_attention_heads", "uint"),
        ("KV_HEAD_COUNT", "num_key_value_heads", "uint"),
        ("HEAD_DIMENSION", "head_dim", "uint"),
        ("INTERMEDIATE_DIMENSION", "intermediate_size", "uint"),
        ("MAXIMUM_CONTEXT_TOKENS", "max_position_embeddings", "uint"),
        ("END_OF_TEXT_TOKEN_ID", "eos_token_id", "uint"),
        ("ROPE_THETA", "rope_theta", "float"),
    ]
    for dname, cname, kind in checks:
        if cname not in config:
            raise MiniMaxConfigError(
                f"config key {cname} missing for SPARK_LLM_{dname}")
        want = define_uint(defines, dname) if kind == "uint" \
            else define_float(defines, dname)
        got = config[cname]
        if isinstance(got, list):
            got = got[0]
        if abs(float(want) - float(got)) > 0:
            raise MiniMaxConfigError(
                f"SPARK_LLM_{dname}={want} disagrees with config "
                f"{cname}={got}")
    scaling = config.get("rope_scaling")
    if not isinstance(scaling, dict):
        raise MiniMaxConfigError("config key rope_scaling missing")
    section = scaling.get("mrope_section")
    if not isinstance(section, list) or len(section) != 3:
        raise MiniMaxConfigError("rope_scaling.mrope_section must have 3 parts")
    for dname, got in (("MROPE_SECTION_TEMPORAL", section[0]),
                       ("MROPE_SECTION_HEIGHT", section[1]),
                       ("MROPE_SECTION_WIDTH", section[2])):
        if define_uint(defines, dname) != int(got):
            raise MiniMaxConfigError(
                f"SPARK_LLM_{dname}={define_uint(defines, dname)} disagrees "
                f"with config rope_scaling.mrope_section value {got}")
    interleaved = scaling.get("mrope_interleaved")
    if not isinstance(interleaved, bool):
        raise MiniMaxConfigError(
            "rope_scaling.mrope_interleaved missing from config")
    if define_uint(defines, "MROPE_INTERLEAVED") != int(interleaved):
        raise MiniMaxConfigError(
            f"SPARK_LLM_MROPE_INTERLEAVED="
            f"{define_uint(defines, 'MROPE_INTERLEAVED')} disagrees with "
            f"config rope_scaling.mrope_interleaved={interleaved}")
    return []


class MiniMaxEngine:
    def __init__(self, checkpoint_dir, defines, config):
        self.mismatches = cross_check(defines, config)
        self.st = Safetensors(checkpoint_dir)
        self.hidden = int(config["hidden_size"])
        self.layers = int(config["num_hidden_layers"])
        self.vocab = int(config["vocab_size"])
        self.eps = float(config["rms_norm_eps"])
        self.heads = int(config["num_attention_heads"])
        self.kv_heads = int(config["num_key_value_heads"])
        self.head_dim = int(config["head_dim"])
        self.intermediate = int(config["intermediate_size"])
        self.rope_theta = float(config["rope_theta"])
        self.max_context = int(config["max_position_embeddings"])
        self.eot = int(config["eos_token_id"] if not isinstance(
            config["eos_token_id"], list) else config["eos_token_id"][0])
        section = config["rope_scaling"]["mrope_section"]
        if sum(int(v) for v in section) != self.head_dim // 2:
            raise MiniMaxConfigError(
                f"mrope_section {section} does not tile head_dim/2 "
                f"{self.head_dim // 2}; text-only rope collapse is invalid")
        self.q_dim = self.heads * self.head_dim
        self.kv_dim = self.kv_heads * self.head_dim
        self.attn_group = self.heads // self.kv_heads
        if self.heads % self.kv_heads != 0:
            raise MiniMaxConfigError(
                f"head count {self.heads} not divisible by kv heads "
                f"{self.kv_heads}")
        self._check_shapes(0)
        self._check_shapes(self.layers - 1)
        self._check_vocab_tensors()
        self._half = self.head_dim // 2
        self._rope_freq = np.exp2(
            -(2.0 * np.arange(self._half, dtype=np.float64)
              / self.head_dim) * np.log2(self.rope_theta)).astype(np.float32)
        self._lm_head_u16 = None

    def _entry_shape(self, name):
        entry = self.st.entry(name)
        return tuple(entry["shape"]), entry["dtype"]

    def _check_shapes(self, layer):
        p = PREFIX + str(layer) + "."
        expected = {
            p + "self_attn.q_proj.weight": (self.q_dim, self.hidden),
            p + "self_attn.k_proj.weight": (self.kv_dim, self.hidden),
            p + "self_attn.v_proj.weight": (self.kv_dim, self.hidden),
            p + "self_attn.o_proj.weight": (self.hidden, self.q_dim),
            p + "self_attn.q_norm.weight": (self.head_dim,),
            p + "self_attn.k_norm.weight": (self.head_dim,),
            p + "mlp.gate_proj.weight": (self.intermediate, self.hidden),
            p + "mlp.up_proj.weight": (self.intermediate, self.hidden),
            p + "mlp.down_proj.weight": (self.hidden, self.intermediate),
            p + "input_layernorm.weight": (self.hidden,),
            p + "post_attention_layernorm.weight": (self.hidden,),
        }
        for name, shape in expected.items():
            got, _ = self._entry_shape(name)
            if got != shape:
                raise MiniMaxConfigError(
                    f"tensor {name} shape {got} disagrees with config-derived "
                    f"{shape}")

    def _check_vocab_tensors(self):
        embed_shape, embed_dtype = self._entry_shape(EMBED_NAME)
        if embed_shape != (self.vocab, self.hidden):
            raise MiniMaxConfigError(
                f"tensor {EMBED_NAME} shape {embed_shape} disagrees with "
                f"config-derived {(self.vocab, self.hidden)}")
        if embed_dtype != "BF16":
            raise MiniMaxConfigError(
                f"reference embedding must be BF16, got {embed_dtype}")
        head_shape, head_dtype = self._entry_shape(LM_HEAD_NAME)
        if head_shape != (self.vocab, self.hidden):
            raise MiniMaxConfigError(
                f"tensor {LM_HEAD_NAME} shape {head_shape} disagrees with "
                f"config-derived {(self.vocab, self.hidden)}; untied "
                "language modeling head expected")
        if head_dtype != "BF16":
            raise MiniMaxConfigError(
                f"reference lm_head must be BF16, got {head_dtype}")

    def tensor(self, name):
        raw = self.st.raw(name)
        if raw.dtype == np.uint16:
            return bf16_to_f32(raw)
        return raw.astype(np.float32)

    def linear(self, x, name):
        return bf16_round_f32(self.tensor(name + ".weight") @ x)

    def embed(self, token_id):
        if token_id < 0 or token_id >= self.vocab:
            raise ValueError(f"token {token_id} outside vocabulary {self.vocab}")
        return bf16_to_f32(self.st.raw_rows(EMBED_NAME, token_id, 1)[0])

    def rope(self, rows, position):
        if position < 0 or position >= self.max_context:
            raise ValueError(f"position {position} outside modeled context "
                             f"{self.max_context}")
        angle = np.float32(position) * self._rope_freq
        cos = np.cos(angle)
        sin = np.sin(angle)
        half = self._half
        real = rows[:, 0:half].copy()
        imag = rows[:, half:self.head_dim].copy()
        rows[:, 0:half] = real * cos - imag * sin
        rows[:, half:self.head_dim] = imag * cos + real * sin
        return rows

    def head_rmsnorm(self, rows, gain):
        variance = (rows * rows).sum(axis=1) / self.head_dim
        return rows / np.sqrt(variance + self.eps)[:, None] * gain[None, :]

    def attention(self, prefix, x, cache, position):
        q = bf16_round_f32(self.tensor(
            prefix + "self_attn.q_proj.weight") @ x).reshape(
            self.heads, self.head_dim)
        q = self.head_rmsnorm(q, self.tensor(
            prefix + "self_attn.q_norm.weight").reshape(-1))
        q = self.rope(q, position)
        k = bf16_round_f32(self.tensor(
            prefix + "self_attn.k_proj.weight") @ x).reshape(
            self.kv_heads, self.head_dim)
        k = self.head_rmsnorm(k, self.tensor(
            prefix + "self_attn.k_norm.weight").reshape(-1))
        k = self.rope(k, position)
        v = bf16_round_f32(self.tensor(
            prefix + "self_attn.v_proj.weight") @ x).reshape(
            self.kv_heads, self.head_dim)
        cache.append((f32_to_bf16_u16(k.reshape(-1)),
                      f32_to_bf16_u16(v.reshape(-1))))
        keys = bf16_to_f32(np.stack([row[0] for row in cache])).reshape(
            len(cache), self.kv_heads, self.head_dim)
        values = bf16_to_f32(np.stack([row[1] for row in cache])).reshape(
            len(cache), self.kv_heads, self.head_dim)
        out = np.empty((self.heads, self.head_dim), dtype=np.float32)
        scale = np.float32(1.0) / np.sqrt(np.float32(self.head_dim))
        for h in range(self.heads):
            kvh = h // self.attn_group
            scores = (keys[:, kvh, :] @ q[h]) * scale
            weights = np.exp(scores - scores.max())
            weights = weights / weights.sum()
            out[h] = weights @ values[:, kvh, :]
        attended = bf16_round_f32(out.reshape(-1))
        return self.linear(attended, prefix + "self_attn.o_proj")

    def mlp(self, prefix, x):
        gate = self.linear(x, prefix + "mlp.gate_proj")
        up = self.linear(x, prefix + "mlp.up_proj")
        activated = bf16_round_f32(swish(gate) * up)
        return self.linear(activated, prefix + "mlp.down_proj")

    def forward_layer(self, index, streams, caches, position):
        prefix = PREFIX + str(index) + "."
        x = bf16_round_f32(rmsnorm(
            streams, self.tensor(prefix + "input_layernorm.weight"), self.eps))
        attention = self.attention(prefix, x, caches[index], position)
        streams = bf16_round_f32(streams + attention)
        x = bf16_round_f32(rmsnorm(
            streams, self.tensor(prefix + "post_attention_layernorm.weight"),
            self.eps))
        return bf16_round_f32(streams + self.mlp(prefix, x))

    def decode_step(self, token_id, position, states, caches, capture,
                    capture_streams=None):
        if token_id < 0 or token_id >= self.vocab:
            raise ValueError(f"token {token_id} outside vocabulary {self.vocab}")
        del states, capture
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

    def logits(self, streams, chunk=4096):
        norm = bf16_round_f32(rmsnorm(
            streams, self.tensor(FINAL_NORM_NAME), self.eps))
        if self._lm_head_u16 is None:
            self._lm_head_u16 = self.st.raw(LM_HEAD_NAME)
            if self._lm_head_u16.dtype != np.uint16:
                raise MiniMaxConfigError("reference lm_head must be BF16")
        lm = self._lm_head_u16
        best = -np.inf
        best_token = -1
        for start in range(0, lm.shape[0], chunk):
            scores = bf16_to_f32(lm[start:start + chunk]) @ norm
            i = int(np.argmax(scores))
            if float(scores[i]) > best:
                best = float(scores[i])
                best_token = start + i
        return best_token, best


ENGINE_CLASS = MiniMaxEngine
