import argparse
import importlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t1_reference_common import parse_llm_defines


def load_config(checkpoint):
    config = json.load(open(os.path.join(checkpoint, "config.json")))
    return config["text_config"] if "text_config" in config else config


def run_position(engine, token_id, position, states, caches):
    capture = {}
    streams = engine.decode_step(token_id, position, states, caches, capture)
    token, score = engine.logits(streams)
    return streams, token, score, capture


def compare_position(slow, fast, token_id, position, slow_state, fast_state,
                     slow_cache, fast_cache):
    s_streams, s_token, s_score, s_capture = run_position(
        slow, token_id, position, slow_state, slow_cache)
    f_streams, f_token, f_score, f_capture = run_position(
        fast, token_id, position, fast_state, fast_cache)
    if s_capture.keys() != f_capture.keys():
        raise SystemExit(json.dumps(
            {"position": position, "kind": "capture_keys",
             "slow": sorted(map(str, s_capture)),
             "fast": sorted(map(str, f_capture))}))
    for key in s_capture:
        s_ids, s_weights = s_capture[key]
        f_ids, f_weights = f_capture[key]
        if not np.array_equal(s_ids, f_ids):
            raise SystemExit(json.dumps(
                {"position": position, "layer": key[1], "kind": "route_ids",
                 "slow": s_ids.tolist(), "fast": f_ids.tolist()}))
        if s_weights.tobytes() != f_weights.tobytes():
            raise SystemExit(json.dumps(
                {"position": position, "layer": key[1],
                 "kind": "route_weights"}))
    if s_streams.tobytes() != f_streams.tobytes():
        bad = int(np.argmax(s_streams.view(np.uint16)
                            != f_streams.view(np.uint16)))
        raise SystemExit(json.dumps(
            {"position": position, "kind": "streams", "index": bad}))
    if s_token != f_token:
        raise SystemExit(json.dumps(
            {"position": position, "kind": "top1_token",
             "slow": s_token, "fast": f_token}))
    if np.float32(s_score).tobytes() != np.float32(f_score).tobytes():
        raise SystemExit(json.dumps(
            {"position": position, "kind": "top1_score",
             "slow": s_score, "fast": f_score}))
    print(json.dumps({"position": position, "layers": len(s_capture),
                      "verdict": "bitwise-equal"}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--header", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--positions", type=int, default=2)
    arguments = parser.parse_args()
    module = importlib.import_module(f"t1_reference_{arguments.family}")
    defines = parse_llm_defines(arguments.header)
    config = load_config(arguments.checkpoint)
    slow = module.Glm53FullEngine(arguments.checkpoint, defines, config)
    os.environ["SPARK_T1_GFULL_FAST"] = "1"
    fast = module.Glm53FullFastEngine(arguments.checkpoint, defines, config)
    os.environ["SPARK_T1_GFULL_FAST"] = "0"
    prompts = json.load(open(arguments.prompts))["prompts"]
    for spec in prompts:
        ids = [int(t) for t in spec["prompt_token_ids"]]
        slow_state, fast_state = {}, {}
        slow_cache, fast_cache = {}, {}
        for position in range(min(arguments.positions, len(ids))):
            token = ids[position]
            compare_position(slow, fast, token, position, slow_state,
                             fast_state, slow_cache, fast_cache)
    print(json.dumps({"verdict": "PASS", "family": arguments.family,
                      "positions_per_prompt": arguments.positions,
                      "prompts": len(prompts)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
