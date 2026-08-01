#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def require(text: str, needle: str, label: str) -> None:
    if needle not in text:
        raise SystemExit(f"missing {label}: {needle}")


def reject(text: str, needle: str, label: str) -> None:
    if needle in text:
        raise SystemExit(f"forbidden {label}: {needle}")


def main() -> None:
    route = read("inference/kernels/route.cuh")
    require(route, "packed_rows != expected_packed_rows", "route cardinality validation")
    require(route, "LmLaunchGroupedTileM(rows,top_k,EXPERTS)", "token-priced grouped tile")

    contracts = {
        "k3": (
            "inference/llms/kimi_k3/layer.cuh",
            "LM_TOPK_SCORE_SIGMOID",
            "K3_ROUTED_SCALE",
        ),
        "glm52": (
            "inference/llms/glm5_2/layer.cuh",
            "LM_TOPK_SCORE_SIGMOID",
            "GLM52_ROUTED_SCALE",
        ),
        "dsv4": (
            "inference/llms/deepseek_v4/layer.cuh",
            "LM_TOPK_SCORE_SQRT_SOFTPLUS",
            "DSV4_ROUTED_SCALE",
        ),
        "mimo25": (
            "inference/llms/mimo_2_5/layer.cuh",
            "LM_TOPK_SCORE_IDENTITY",
            "1.0f",
        ),
    }
    for family, (path, transform, scale) in contracts.items():
        text = read(path)
        require(text, "LmRouteBuild<", f"{family} device route build")
        require(text, "group_tile_prefix_w1", f"{family} W1 prefix")
        require(text, "group_tile_prefix_w2", f"{family} W2 prefix")
        require(text, "prefix_built = 1u", f"{family} prebuilt prefix")
        require(text, transform, f"{family} router transform")
        require(text, scale, f"{family} router scale")
        reject(text, "LmLaunchGroupedTileM(packed_rows", f"{family} route-priced tile")

    k3 = read("inference/llms/kimi_k3/layer.cuh")
    require(k3, "b->expert_w1_weight,packed_rows,rows,", "K3 W1 token count")
    require(k3, "b->expert_w2_weight,packed_rows,rows,", "K3 W2 token count")

    dsv4 = read("inference/llms/deepseek_v4/layer.cuh")
    mimo = read("inference/llms/mimo_2_5/layer.cuh")
    require(dsv4, "float *router_logits;", "DSV4 FP32 router output")
    require(mimo, "float *router_logits;", "MiMo FP32 router output")
    reject(dsv4, "(uint32_t *)b->head_candidate_score", "DSV4 scratch alias")

    model = read("tests/studies/sparkpipe_glm52_batchplane_model.c")
    require(model, "expert_sweeps_per_active_expert = 1.0", "one expert sweep model")
    require(model, "replay/chunk expert-sweep multiplier: 1.0", "removed replay multiplier")
    reject(model, "BP_LAYERS * (BP_EXPERTS", "old queue-depth divisor")

    queue_header = read(
        "model-families/glm52/include/sparkpipe/spark_glm52_expert_queue.h"
    )
    queue_source = read(
        "model-families/glm52/src/spark_glm52_expert_queue.c"
    )
    require(queue_header, "SPARK_GLM52_EXPERT_QUEUE_MODE_SEALED_BATCH", "sealed mode")
    require(queue_header, "SparkGlm52ExpertQueueSealLayer", "seal API")
    require(queue_source, "queue->layer_sealed[layer_index]", "sealed layer state")

    # -- the kernel half of the gather deletion (route.cuh's contract) --------
    # The driver half (layer.cuh) belongs to another wave; these pins hold the
    # kernel side it wires against: the row-map words, the ragged-tail clamp,
    # source-following scales, and byte-identical barrier accounting between
    # the TMA and the indirect staging paths.
    gemm = read("inference/kernels/gemm.cuh")
    tile = read("inference/kernels/tile.cuh")
    tma = read("inference/kernels/tma.cuh")
    require(gemm, "const uint32_t *activation_row_index;", "indirect-A row map word")
    require(gemm, "const void *activation_source;", "indirect-A source base word")
    require(gemm, "LmRouteSourceRow(activation_row_index", "scale follows the source row")
    require(tile, "LmPipelineProduceIndirectA(", "indirect A staging path")
    require(tile, "packed >= row_limit", "ragged tail clamp")
    require(tile, "LmRouteSourceRow(row_index,packed)", "staged row through the route map")
    require(tma, "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes",
            "tx-accounted bulk chunk copy")
    # both staging paths declare the same bytes through the one helper, or the
    # barrier's expected count can drift between them and deadlock one path
    produce = tile[tile.index("static __device__ __forceinline__ void LmPipelineProduce("):]
    indirect = tile[tile.index("LmPipelineProduceIndirectA("):]
    for name, body in (("dense", produce), ("indirect", indirect)):
        require(body[:body.index("\n}\n")], "LmPipelineProduceWeight(",
                f"{name} path shares the expect+weight helper")

    print("PASS grouped-MoE source contracts")


if __name__ == "__main__":
    main()
