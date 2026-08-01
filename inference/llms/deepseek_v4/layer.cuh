#pragma once

// One DeepSeek V4 decode layer.
//
// The third sequence, and the one that shares most with GLM 5.2 - a latent cache
// and sparse selection over it. What differs is what makes it worth writing
// separately rather than parameterising glm5_2's:
//
//     six experts per token, not eight
//     a SHARED expert every token passes through, added not weighted
//     a sliding window on top of the sparse selection
//     YaRN rope, because the context was extended past training
//     a low-rank query path at rank 1024
//
// The first of those I got wrong in config.h before writing any of this, by
// carrying GLM 5.2's eight across. A constant that looks like another model's
// and is not is the failure this whole llms/ arrangement exists to make visible.

#include "runtime/gemm.cuh"
#include "inference/kernels/norm.cuh"
#include "inference/kernels/attn.cuh"
#include "inference/kernels/topk.cuh"
#include "inference/kernels/route.cuh"
#include "inference/kernels/project.cuh"
#include "inference/kernels/head.cuh"
#include "inference/kernels/formats/fp8.cuh"
#include "inference/kernels/formats/mxfp4.cuh"
#include "inference/llms/deepseek_v4/config.h"

using Dsv4Kv = LmKvLatent<DSV4_KV_BITS, DSV4_HEAD_DIM, DSV4_ROPE_DIM, DSV4_KV_PAGE_SLOTS>;

// Overridable because the host harnesses instantiate the layer at one thread:
// the CPU shim's sequential schedule (tests/host_cuda/lm_host_cuda.cuh) is
// only a valid execution of these kernels when the template width IS one.
// Device builds take the default and nothing changes.
#ifndef DSV4_LAYER_THREADS
#define DSV4_LAYER_THREADS 256u
#endif
#define DSV4_LAYER_TILE_N 128u
#define DSV4_LAYER_STAGES 2u
#define DSV4_LAYER_WARPS 8u

static LmScaleTensor Dsv4Fp8ActivationScale(
    const void *scale_data,
    uint32_t row_count,
    uint32_t input_dimension)
{
    return LmScaleTensorRowsUe4m3(
        scale_data,
        row_count,
        input_dimension,
        LmFp8::kScaleGroup);
}

static LmScaleTensor Dsv4Fp8WeightScale(
    const void *scale_data,
    uint32_t group_count,
    uint32_t output_dimension,
    uint32_t input_dimension)
{
    return LmScaleTensorBlockF32(
        scale_data,
        group_count,
        output_dimension,
        input_dimension,
        128u,
        128u);
}

static LmScaleTensor Dsv4CheckpointFp4WeightScale(
    const void *scale_data,
    uint32_t group_count,
    uint32_t output_dimension,
    uint32_t input_dimension)
{
    return LmScaleTensorBlockUe8m0(
        scale_data,
        group_count,
        output_dimension,
        input_dimension,
        1u,
        LmMxfp4::kScaleGroup);
}

struct Dsv4LayerBuffers
{
	const void *attn_norm_weight;
	LmLowRankWeights query;
	LmLowRankScratch query_scratch;
	const void *kv_latent_weight;
	const void *kv_latent_scale;
	// Grouped low-rank o_proj (contract output_group_count x output_lora_rank):
	// the down factor is block-diagonal per group and ships BF16 - the tree's
	// only block-diagonal projection is LmPerHeadProjectKernel, which is BF16,
	// and the tensor-map GEMM cannot stride an activation by group, so an FP8
	// down is a new kernel and a later wave. The up factor is FP8 like every
	// non-expert weight.
	const void *output_down_weight;
	const void *output_up_weight;
	const void *output_up_scale;
	const void *mlp_norm_weight;
	const void *router_weight;
	const void *expert_w1_weight;
	const void *expert_w1_scale;
	const void *expert_w2_weight;
	const void *expert_w2_scale;
	// The shared expert is a dense MLP with its own weights, not expert zero of
	// the routed set.
	const void *shared_w1_weight;
	const void *shared_w1_scale;
	const void *shared_w2_weight;
	const void *shared_w2_scale;

	uint16_t *hidden_bf16;
	uint16_t *residual_bf16;
	uint16_t *normed_bf16;
	uint16_t *query_bf16;
	uint16_t *kv_slot_bf16;
	uint16_t *attention_latent_bf16;
	uint16_t *attention_out_bf16;
	// The concatenated per-group down projections, rows x (groups * rank): the
	// up GEMM reads it as one K dimension because sum_g Up_g d_g is
	// [d_0|...|d_G] @ [U_0;...;U_G].
	uint16_t *output_down_bf16;
	uint8_t *packed_activation;
	uint8_t *packed_scale;
	uint16_t *gate_up_bf16;
	uint16_t *intermediate_bf16;
	uint16_t *expert_out_bf16;
	uint16_t *shared_out_bf16;
	float *router_logits;
	uint32_t *route_expert;
	float *route_weight;
	uint32_t *route_source_token;
	uint32_t *route_packed_row;
	const uint32_t *dense_row_offset;
	uint32_t *dense_tile_prefix;
	uint32_t *group_row_offset;
	uint32_t *group_tile_prefix_w1;
	uint32_t *group_tile_prefix_w2;

	LmKvView cache;
	const uint32_t *sequence_of_row;
	const uint32_t *context_length;
	const uint32_t *positions;
	uint32_t *selected_positions;
	float *selection_scores;
	float *head_candidate_score;
	uint32_t *head_candidate_token;
	uint32_t *output_token;
	float *output_score;
};

// -- exact attention weight bytes, replacing the roadmap's approximation ---------
//
// docs/PERF_ROADMAP_2026-08-01.md prices this line at "~195 M params/layer" for
// Flash and "~632 M" for Pro, and flags +-15% because the grouped low-rank
// output projection (output_lora_rank, output_group_count) was approximated as
// full-width. The GEMMs below are the ground truth; every factor is config
// geometry and tests/test_dsv4_driver_source_contracts.py recomputes the
// figures from config.h and the contracts, so a stale number fails a gate:
//
//     Flash, as coded (config.h, contract o_proj at 8 groups x rank 1024):
//         q_down 4096x1024 + q_up 1024x32768 + kv_latent 4096x576
//         + o_down 8x(4096x1024) + o_up (8x1024)x4096 = 107216896 weights/layer
//         140771328 bytes - the down factor ships BF16 (see the o_proj below)
//         -> 6.05 GB/token over 43 layers    (full-width o_proj priced 7.50 GB)
//     Pro, as coded (dsv4_pro.json geometry, contract o_proj 16 x 1024):
//         q_down 7168x1536 + q_up 1536x65536 + kv_latent 7168x576
//         + o_down 16x(4096x1024) + o_up (16x1024)x7168 = 300351488 weights/layer
//         367460352 bytes with the BF16 down factor
//         -> 22.42 GB/token over 61 layers   (full-width priced 35.72 GB)
//         The o_proj factors alone are 184549376 weights; all-FP8 they would
//         price 18.32 GB/token - the BF16 down factor pays double until a
//         block-diagonal FP8 kernel exists. That is the remaining third of the
//         lever: the full attention line is 22.42 + 14.1 experts + 1.9 head
//         + 0.34 router = 38.76 GB/token, ~74 tok/s at the roadmap's 80% eta,
//         from ~55 tok/s at the 52.1 GB/token full-width coding.
//
// The router is separate and BF16 (hidden x experts x 2 B per layer, read
// every step): 90 MB/token Flash, 336 MB/token Pro, absent from the roadmap's
// Pro line. Index projections are absent from this driver entirely - the
// sparse score below reads the query and the latent cache directly - so they
// are not priced here. Block-128x128 FP8 weight scales add 0.024%: noise.
template<class Format>
static int32_t Dsv4LayerAttention(const Dsv4LayerBuffers *b, uint32_t rows, uint32_t context, uint32_t sms, cudaStream_t stream)
{
	LmGemmArguments gemm;
	int32_t status;
	// The sliding window bounds the selection rather than replacing it: DeepSeek
	// V4 selects 512 positions AND restricts to a 128-token window, so the
	// effective set is whichever is smaller. Treating them as alternatives would
	// attend outside the window on any context longer than 512.
	uint32_t budget = context < DSV4_SLIDING_WINDOW ? context : DSV4_SLIDING_WINDOW;
	// WINDOW: the attended set is the selection clamped to the trailing
	// 128-token window, per the budget comment above - the effective set is
	// the smaller of the two. The score kernel fails every position older
	// than the window, and the top-k and the attention run at the clamped
	// budget, so nothing outside the window can be selected or read. The
	// first version attended the full top-k with no clamp.
	uint32_t selected = DSV4_INDEX_TOP_K < DSV4_SLIDING_WINDOW
		? DSV4_INDEX_TOP_K : DSV4_SLIDING_WINDOW;
	// CAPTURE: this host branch on a runtime context length changes the launch
	// sequence, so a CUDA graph of this layer is valid only for the (rows,
	// sparse) shape it was captured at - the engine must key graphs on both.
	bool sparse = context > DSV4_INDEX_TOP_K;
	LM_LAUNCH((LmFusedResidualRmsNormKernel<DSV4_LAYER_THREADS,uint16_t>), rows, DSV4_LAYER_THREADS, (DSV4_HIDDEN + 8u) * sizeof(float), stream,
		b->hidden_bf16,b->residual_bf16,(const uint16_t *)b->attn_norm_weight, b->residual_bf16,b->normed_bf16,DSV4_HIDDEN,DSV4_HIDDEN,DSV4_RMS_EPSILON);
	// Query through the low-rank path: hidden -> 1024 -> norm -> heads.
	//
	// WIDTH: the up projection's output width is binder-set, and the buffer
	// it writes is DSV4_QUERY_ROW wide - heads x (latent + rope), with each
	// head's rope span at the tail of its own stride. config.h names that row
	// now, so the binder has a constant to be right with and is checked
	// before anything writes the buffer.
	if ( b->query.output_dimension != DSV4_QUERY_ROW )
		return(LM_LAUNCH_ERR_SHAPE);
	status = LmLowRankProject<Format>(&b->query,&b->query_scratch,b->normed_bf16,
		b->query_bf16,rows,DSV4_LAYER_THREADS,sms,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	// THE QUERY'S ROPE. Omitted in a first version of this file, which the
	// coverage gate caught by noticing DSV4_ROPE_THETA was declared and unused.
	// Query and key must be rotated into the same frame or their dot product
	// measures the angle between frames rather than between contents - the
	// scores stay finite and plausible and the attention is meaningless.
	//
	// The query uses DSV4_ROPE_THETA and the compressed KV path uses
	// DSV4_COMPRESS_ROPE_THETA. Two thetas is not a mistake in the config: the
	// model trains them separately because the compressed path sees a different
	// position distribution.
	//
	// SPAN: the query row is heads x (latent + rope) with each head's rope
	// span at the tail of its own stride, so YaRN launches per (row, head)
	// and rotates all 64 spans. The first version launched one block per row
	// over a row priced at heads * head_dim: one head of sixty-four rotated,
	// and at the wrong offset for every row after the first.
	LM_LAUNCH((LmRopeYarnKernel<DSV4_LAYER_THREADS,LM_ROPE_INTERLEAVED>), dim3(rows,DSV4_ATTN_HEADS), DSV4_LAYER_THREADS, 0, stream,
		b->query_bf16,b->positions,DSV4_ATTN_HEADS,DSV4_QUERY_HEAD_STRIDE,DSV4_ROPE_DIM, DSV4_ROPE_THETA,(float)DSV4_YARN_FACTOR, (float)DSV4_YARN_ORIGINAL_POSITIONS,1.0f,32.0f);
	// NO SECOND QUANTISE OF THE NORMED ROWS. LmLowRankProject already ran
	// LmQuantiseRowsKernel over this exact buffer into the query scratch - same
	// kernel, same grid, same row-UE4M3 scale layout - so the KV latent GEMM
	// reads that copy. The launch this replaces cost one kernel per layer per
	// token plus a full re-read and re-write of the normed rows: 61 launches
	// and ~1.35 GB of activation traffic per step at B1024 on Pro. The guard
	// is what makes the reuse a contract rather than a coincidence: the codes
	// are only the normed rows if the down projection consumes the hidden.
	static_assert(Format::kScaleGroup > 0u,
		"the KV latent GEMM reuses the low-rank path's quantised input, which exists only for a quantised format");
	if ( b->query.input_dimension != DSV4_HIDDEN )
		return(LM_LAUNCH_ERR_SHAPE);
	memset(&gemm,0,sizeof(gemm));
	gemm.scale_a = Dsv4Fp8ActivationScale(
		b->query_scratch.input_scales,
		rows,
		DSV4_HIDDEN);
	gemm.scale_b = Dsv4Fp8WeightScale(
		b->kv_latent_scale,
		1u,
		DSV4_HEAD_DIM + DSV4_ROPE_DIM,
		DSV4_HIDDEN);
	gemm.group_row_offset = b->dense_row_offset;
	gemm.group_tile_prefix = b->dense_tile_prefix;
	gemm.output_bf16 = b->kv_slot_bf16;
	status = LmGemmLaunch<Format,DSV4_LAYER_TILE_N,Format::kTileK,DSV4_LAYER_STAGES,DSV4_LAYER_WARPS>(
		&gemm,b->query_scratch.input_codes,b->kv_latent_weight,rows,rows,DSV4_TOP_K,1u,
		DSV4_HIDDEN,DSV4_HEAD_DIM + DSV4_ROPE_DIM,sms,false,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	// YaRN, not plain rope. The compressed path uses its own theta, which is why
	// the constant is separate rather than shared with the query's. The latent
	// is one rope span per row - the per-head kernel at heads 1.
	LM_LAUNCH((LmRopeYarnKernel<DSV4_LAYER_THREADS,LM_ROPE_INTERLEAVED>), dim3(rows,1u), DSV4_LAYER_THREADS, 0, stream,
		b->kv_slot_bf16,b->positions,1u,DSV4_HEAD_DIM + DSV4_ROPE_DIM,DSV4_ROPE_DIM,DSV4_COMPRESS_ROPE_THETA,(float)DSV4_YARN_FACTOR, (float)DSV4_YARN_ORIGINAL_POSITIONS,1.0f,32.0f);
	LM_LAUNCH((LmKvStoreKernel<Dsv4Kv,DSV4_LAYER_THREADS>), rows, DSV4_LAYER_THREADS, 0, stream,
		b->cache,b->kv_slot_bf16,b->sequence_of_row,b->positions,rows, DSV4_HEAD_DIM + DSV4_ROPE_DIM);
	if ( sparse )
	{
		// GRID LIMIT, REMOVED: position rides grid.x now, not grid.y - the y
		// axis tops out at 65,535 blocks on every CUDA arch and this grid
		// spans the context, which the model sells at a million tokens. The
		// row count is the batch and never nears the ceiling.
		LM_LAUNCH((LmSparseScoreKernel<Dsv4Kv,DSV4_LAYER_THREADS,DSV4_INDEX_DIM>), dim3(context,rows), DSV4_LAYER_THREADS, 0, stream,
		b->query_bf16,b->cache,b->sequence_of_row,b->context_length, DSV4_INDEX_HEADS,DSV4_SLIDING_WINDOW,b->selection_scores);
		LM_LAUNCH((LmTopkHistogramKernel<DSV4_LAYER_THREADS>), rows, DSV4_LAYER_THREADS, 0, stream,
		b->selection_scores,context,selected,b->head_candidate_token);
		LM_LAUNCH((LmTopkGatherKernel<DSV4_LAYER_THREADS>), rows, DSV4_LAYER_THREADS, 0, stream,
		b->selection_scores,context,selected,b->head_candidate_token,b->selected_positions,0);
	}
	LM_LAUNCH((LmAttentionDecodeKernel<Dsv4Kv,DSV4_LAYER_THREADS,DSV4_HEAD_DIM,DSV4_ROPE_DIM>), dim3(rows,DSV4_ATTN_HEADS), DSV4_LAYER_THREADS, 0, stream,
		b->query_bf16,b->query_bf16,b->cache,b->sequence_of_row,b->context_length, sparse ? b->selected_positions : 0,sparse ? selected : budget, DSV4_ATTN_HEADS,rsqrtf((float)DSV4_HEAD_DIM),b->attention_latent_bf16, 0);
	// THE GROUPED LOW-RANK O_PROJ. out = sum_g Up_g (Down_g x_g) over
	// DSV4_OUTPUT_GROUP_COUNT rank-DSV4_OUTPUT_LORA_RANK factors, x_g the g-th
	// group_dim slice of the attention output - the contract's form
	// (dsv4_flash.json and dsv4_pro.json both name output_group_count and
	// output_lora_rank), at groups*(group_dim*rank + rank*hidden) weights
	// against heads*dim x hidden full-width. The down half is block-diagonal
	// per group, LmPerHeadProjectKernel's exact shape (k3 uses it for the MLA
	// value path), and BF16 for want of a block-diagonal FP8 kernel; the up
	// half is one dense FP8 GEMM over the concatenated ranks.
	static_assert((DSV4_ATTN_HEADS * DSV4_HEAD_DIM) % DSV4_OUTPUT_GROUP_COUNT == 0u,
		"the attention width must split into whole o_proj groups");
	static_assert(DSV4_OUTPUT_RANK_WIDTH % Format::kScaleGroup == 0u,
		"the concatenated o_proj ranks must fill whole scale groups");
	LM_LAUNCH((LmPerHeadProjectKernel<DSV4_LAYER_THREADS,DSV4_OUTPUT_GROUP_DIM,DSV4_OUTPUT_LORA_RANK>), dim3(rows,DSV4_OUTPUT_GROUP_COUNT), DSV4_LAYER_THREADS, 0, stream,
		b->attention_latent_bf16,(const uint16_t *)b->output_down_weight,b->output_down_bf16,DSV4_OUTPUT_GROUP_COUNT,rows);
	LM_LAUNCH((LmQuantiseRowsKernel<Format,DSV4_LAYER_THREADS>), dim3(rows,DSV4_OUTPUT_RANK_WIDTH / Format::kScaleGroup), DSV4_LAYER_THREADS, (Format::kScaleGroup + 8u) * sizeof(float), stream,
		b->output_down_bf16,0,b->packed_activation,b->packed_scale, rows,DSV4_OUTPUT_RANK_WIDTH);
	gemm.scale_a = Dsv4Fp8ActivationScale(
		b->packed_scale,
		rows,
		DSV4_OUTPUT_RANK_WIDTH);
	gemm.scale_b = Dsv4Fp8WeightScale(
		b->output_up_scale,
		1u,
		DSV4_HIDDEN,
		DSV4_OUTPUT_RANK_WIDTH);
	gemm.output_bf16 = b->attention_out_bf16;
	return(LmGemmLaunch<Format,DSV4_LAYER_TILE_N,Format::kTileK,DSV4_LAYER_STAGES,DSV4_LAYER_WARPS>(
		&gemm,b->packed_activation,b->output_up_weight,rows,rows,DSV4_TOP_K,1u,
		DSV4_OUTPUT_RANK_WIDTH,DSV4_HIDDEN,sms,false,stream));
}

// The MLP half, with the shared expert.
//
// Every token passes through the shared expert IN ADDITION to its six routed
// ones, and its output is added rather than weighted - it has no gate and is not
// part of the top-k. Omitting it drops a dense contribution from every token,
// which degrades quality uniformly rather than visibly, and is the kind of thing
// that reads as "the port is a bit worse" rather than as a missing term.
//
// It is computed on the same normed rows as the routed path, so the norm happens
// once and both read it.
template<class NonExpertFormat, class RoutedExpertWeightFormat>
static int32_t Dsv4LayerMoe(
    const Dsv4LayerBuffers *b,
    uint32_t rows,
    uint32_t packed_rows,
    uint32_t sms,
    cudaStream_t stream)
{
    LmGemmArguments gemm;
    int32_t status;

    static_assert(
        NonExpertFormat::kScaleGroup == 128u,
        "DeepSeek V4 dynamic activations use 128-element FP8 scale groups");
    static_assert(
        RoutedExpertWeightFormat::kScaleGroup == 32u,
        "DeepSeek V4 checkpoint FP4 experts use 32-element UE8M0 groups");

    LM_LAUNCH(
        (LmFusedResidualRmsNormKernel<DSV4_LAYER_THREADS,uint16_t>),
        rows,
        DSV4_LAYER_THREADS,
        (DSV4_HIDDEN + 8u) * sizeof(float),
        stream,
        b->attention_out_bf16,
        b->residual_bf16,
        (const uint16_t *)b->mlp_norm_weight,
        b->residual_bf16,
        b->normed_bf16,
        DSV4_HIDDEN,
        DSV4_HIDDEN,
        DSV4_RMS_EPSILON);

    memset(&gemm, 0, sizeof(gemm));
    gemm.scale_a = LmScaleTensorNone();
    gemm.scale_b = LmScaleTensorNone();
    gemm.group_row_offset = b->dense_row_offset;
    gemm.group_tile_prefix = b->dense_tile_prefix;
    gemm.output_f32 = b->router_logits;
    status = LmGemmLaunch<
        LmBf16Format,
        DSV4_LAYER_TILE_N,
        LmBf16Format::kTileK,
        DSV4_LAYER_STAGES,
        DSV4_LAYER_WARPS>(
            &gemm,
            b->normed_bf16,
            b->router_weight,
            rows,
            rows,
            1u,
            1u,
            DSV4_HIDDEN,
            DSV4_EXPERTS,
            sms,
            false,
            stream);
    if (status != LM_LAUNCH_OK)
    {
        return status;
    }

    LM_LAUNCH(
        (LmTopkSmallKernel<
            DSV4_LAYER_THREADS,
            DSV4_TOP_K,
            true,
            1u,
            1u,
            LM_TOPK_SCORE_SQRT_SOFTPLUS>),
        rows,
        DSV4_LAYER_THREADS,
        2u * LM_TOPK_SMALL_LIMIT * sizeof(uint32_t),
        stream,
        b->router_logits,
        DSV4_EXPERTS,
        b->route_expert,
        b->route_weight,
        0,
        0,
        DSV4_ROUTED_SCALE);
    status = LmRouteBuild<DSV4_LAYER_THREADS,DSV4_EXPERTS>(
        b->route_expert,
        rows,
        packed_rows,
        DSV4_TOP_K,
        b->group_row_offset,
        b->route_packed_row,
        b->route_source_token,
        DSV4_EXPERT_INTERMEDIATE * 2u,
        DSV4_HIDDEN,
        DSV4_LAYER_TILE_N,
        b->group_tile_prefix_w1,
        b->group_tile_prefix_w2,
        stream);
    if (status != LM_LAUNCH_OK)
    {
        return status;
    }

    LM_LAUNCH(
        (LmQuantiseRowsKernel<NonExpertFormat,DSV4_LAYER_THREADS>),
        dim3(rows, DSV4_HIDDEN / NonExpertFormat::kScaleGroup),
        DSV4_LAYER_THREADS,
        (NonExpertFormat::kScaleGroup + 8u) * sizeof(float),
        stream,
        b->normed_bf16,
        0,
        b->packed_activation,
        b->packed_scale,
        rows,
        DSV4_HIDDEN);
    memset(&gemm, 0, sizeof(gemm));
    gemm.scale_a = Dsv4Fp8ActivationScale(
        b->packed_scale,
        rows,
        DSV4_HIDDEN);
    gemm.scale_b = Dsv4Fp8WeightScale(
        b->shared_w1_scale,
        1u,
        DSV4_SHARED_INTERMEDIATE * 2u,
        DSV4_HIDDEN);
    gemm.group_row_offset = b->dense_row_offset;
    gemm.group_tile_prefix = b->dense_tile_prefix;
    gemm.output_bf16 = b->gate_up_bf16;
    status = LmGemmLaunch<
        NonExpertFormat,
        DSV4_LAYER_TILE_N,
        NonExpertFormat::kTileK,
        DSV4_LAYER_STAGES,
        DSV4_LAYER_WARPS>(
            &gemm,
            b->packed_activation,
            b->shared_w1_weight,
            rows,
            rows,
            1u,
            1u,
            DSV4_HIDDEN,
            DSV4_SHARED_INTERMEDIATE * 2u,
            sms,
            false,
            stream);
    if (status != LM_LAUNCH_OK)
    {
        return status;
    }

    LM_LAUNCH(
        (LmSiluMulKernel<DSV4_LAYER_THREADS>),
        rows,
        DSV4_LAYER_THREADS,
        0,
        stream,
        b->gate_up_bf16,
        b->intermediate_bf16,
        DSV4_SHARED_INTERMEDIATE,
        true);
    LM_LAUNCH(
        (LmQuantiseRowsKernel<NonExpertFormat,DSV4_LAYER_THREADS>),
        dim3(
            rows,
            DSV4_SHARED_INTERMEDIATE / NonExpertFormat::kScaleGroup),
        DSV4_LAYER_THREADS,
        (NonExpertFormat::kScaleGroup + 8u) * sizeof(float),
        stream,
        b->intermediate_bf16,
        0,
        b->packed_activation,
        b->packed_scale,
        rows,
        DSV4_SHARED_INTERMEDIATE);
    gemm.scale_a = Dsv4Fp8ActivationScale(
        b->packed_scale,
        rows,
        DSV4_SHARED_INTERMEDIATE);
    gemm.scale_b = Dsv4Fp8WeightScale(
        b->shared_w2_scale,
        1u,
        DSV4_HIDDEN,
        DSV4_SHARED_INTERMEDIATE);
    gemm.output_bf16 = b->shared_out_bf16;
    status = LmGemmLaunch<
        NonExpertFormat,
        DSV4_LAYER_TILE_N,
        NonExpertFormat::kTileK,
        DSV4_LAYER_STAGES,
        DSV4_LAYER_WARPS>(
            &gemm,
            b->packed_activation,
            b->shared_w2_weight,
            rows,
            rows,
            1u,
            1u,
            DSV4_SHARED_INTERMEDIATE,
            DSV4_HIDDEN,
            sms,
            false,
            stream);
    if (status != LM_LAUNCH_OK)
    {
        return status;
    }

    LM_LAUNCH(
        (LmQuantiseRowsKernel<NonExpertFormat,DSV4_LAYER_THREADS>),
        dim3(packed_rows, DSV4_HIDDEN / NonExpertFormat::kScaleGroup),
        DSV4_LAYER_THREADS,
        (NonExpertFormat::kScaleGroup + 8u) * sizeof(float),
        stream,
        b->normed_bf16,
        b->route_source_token,
        b->packed_activation,
        b->packed_scale,
        packed_rows,
        DSV4_HIDDEN);
    memset(&gemm, 0, sizeof(gemm));
    gemm.scale_a = Dsv4Fp8ActivationScale(
        b->packed_scale,
        packed_rows,
        DSV4_HIDDEN);
    gemm.scale_b = Dsv4CheckpointFp4WeightScale(
        b->expert_w1_scale,
        DSV4_EXPERTS,
        DSV4_EXPERT_INTERMEDIATE * 2u,
        DSV4_HIDDEN);
    gemm.group_row_offset = b->group_row_offset;
    gemm.group_tile_prefix = b->group_tile_prefix_w1;
    gemm.prefix_built = 1u;
    gemm.output_bf16 = b->gate_up_bf16;
    status = LmGemmLaunchAsymmetric<
        NonExpertFormat,
        RoutedExpertWeightFormat,
        DSV4_LAYER_TILE_N,
        NonExpertFormat::kTileK,
        DSV4_LAYER_STAGES,
        DSV4_LAYER_WARPS>(
            &gemm,
            b->packed_activation,
            b->expert_w1_weight,
            packed_rows,
            rows,
            DSV4_TOP_K,
            DSV4_EXPERTS,
            DSV4_HIDDEN,
            DSV4_EXPERT_INTERMEDIATE * 2u,
            sms,
            true,
            stream);
    if (status != LM_LAUNCH_OK)
    {
        return status;
    }

    LM_LAUNCH(
        (LmSiluMulKernel<DSV4_LAYER_THREADS>),
        packed_rows,
        DSV4_LAYER_THREADS,
        0,
        stream,
        b->gate_up_bf16,
        b->intermediate_bf16,
        DSV4_EXPERT_INTERMEDIATE,
        true);
    LM_LAUNCH(
        (LmQuantiseRowsKernel<NonExpertFormat,DSV4_LAYER_THREADS>),
        dim3(
            packed_rows,
            DSV4_EXPERT_INTERMEDIATE / NonExpertFormat::kScaleGroup),
        DSV4_LAYER_THREADS,
        (NonExpertFormat::kScaleGroup + 8u) * sizeof(float),
        stream,
        b->intermediate_bf16,
        0,
        b->packed_activation,
        b->packed_scale,
        packed_rows,
        DSV4_EXPERT_INTERMEDIATE);
    gemm.scale_a = Dsv4Fp8ActivationScale(
        b->packed_scale,
        packed_rows,
        DSV4_EXPERT_INTERMEDIATE);
    gemm.scale_b = Dsv4CheckpointFp4WeightScale(
        b->expert_w2_scale,
        DSV4_EXPERTS,
        DSV4_HIDDEN,
        DSV4_EXPERT_INTERMEDIATE);
    gemm.group_tile_prefix = b->group_tile_prefix_w2;
    gemm.output_bf16 = b->expert_out_bf16;
    status = LmGemmLaunchAsymmetric<
        NonExpertFormat,
        RoutedExpertWeightFormat,
        DSV4_LAYER_TILE_N,
        NonExpertFormat::kTileK,
        DSV4_LAYER_STAGES,
        DSV4_LAYER_WARPS>(
            &gemm,
            b->packed_activation,
            b->expert_w2_weight,
            packed_rows,
            rows,
            DSV4_TOP_K,
            DSV4_EXPERTS,
            DSV4_EXPERT_INTERMEDIATE,
            DSV4_HIDDEN,
            sms,
            true,
            stream);
    if (status != LM_LAUNCH_OK)
    {
        return status;
    }

    LM_LAUNCH(
        (LmMoeFinalizeKernel<DSV4_LAYER_THREADS>),
        dim3(
            (DSV4_HIDDEN + DSV4_LAYER_THREADS - 1u) /
                DSV4_LAYER_THREADS,
            rows),
        DSV4_LAYER_THREADS,
        0,
        stream,
        b->expert_out_bf16,
        b->route_packed_row,
        b->route_weight,
        b->hidden_bf16,
        rows,
        DSV4_TOP_K,
        DSV4_HIDDEN);
    LM_LAUNCH(
        (LmAddRowsKernel<DSV4_LAYER_THREADS>),
        dim3(
            (DSV4_HIDDEN + DSV4_LAYER_THREADS - 1u) /
                DSV4_LAYER_THREADS,
            rows),
        DSV4_LAYER_THREADS,
        0,
        stream,
        b->hidden_bf16,
        b->shared_out_bf16,
        b->hidden_bf16,
        rows,
        DSV4_HIDDEN);
    return cudaPeekAtLastError() == cudaSuccess
        ? LM_LAUNCH_OK
        : LM_LAUNCH_ERR_LAUNCH;
}

// -- output head ----------------------------------------------------------------
//
// A layer stack that never reaches a vocabulary produces no token. This model
// had every layer kind and no head, so DSV4_VOCAB was declared, carried through
// the config gate as an exemption reading "no layer sequence yet, so no head
// call", and never read by anything. The sequence is the same three kernels
// glm5_2 uses; only the widths differ.
#define DSV4_HEAD_TILE 1024u

static int32_t Dsv4Head(const Dsv4LayerBuffers *b, const void *head_norm_weight, const void *head_weight, const uint32_t *token_ids, uint32_t vocabulary, uint32_t rows, cudaStream_t stream)
{
	uint32_t tiles = (vocabulary + DSV4_HEAD_TILE - 1u) / DSV4_HEAD_TILE;
	// End of the stream, so no residual in and no residual out.
	LM_LAUNCH((LmFusedResidualRmsNormKernel<DSV4_LAYER_THREADS,uint16_t>), rows, DSV4_LAYER_THREADS, (DSV4_HIDDEN + 8u) * sizeof(float), stream,
		b->hidden_bf16,0,(const uint16_t *)head_norm_weight, 0,b->normed_bf16,DSV4_HIDDEN,DSV4_HIDDEN,DSV4_RMS_EPSILON);
	LM_LAUNCH((LmHeadCandidateKernel<DSV4_LAYER_THREADS,DSV4_HEAD_TILE>), dim3(tiles,rows), DSV4_LAYER_THREADS, 0, stream,
		b->normed_bf16,(const uint16_t *)head_weight,token_ids, b->head_candidate_score,b->head_candidate_token,rows,DSV4_HIDDEN,vocabulary);
	LM_LAUNCH((LmHeadCommitKernel<DSV4_LAYER_THREADS>), rows, DSV4_LAYER_THREADS, 0, stream,
		b->head_candidate_score,b->head_candidate_token,tiles, b->output_token,b->output_score,rows);
	return(cudaPeekAtLastError() == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
}
