#pragma once
#include <stdint.h>
#include <stddef.h>

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#include <stdio.h>
#include "sparkpipe/spark_tp_mesh_round_control.h"
#define SPARK_TP_MESH_KERNELS_MARKER "SPARK-TP-MESH-KERNELS-V5-CANCELPOLL-ORDPARITY"
#if defined(__CUDACC__)
__constant__ char SparkTpMeshKernelsBuildMarker[] =
    SPARK_TP_MESH_KERNELS_MARKER;
#endif

#define SPARK_TP_MESH_THREADS 256u

static __device__ __forceinline__ unsigned long long SparkGlm5NextGlobalTimerNs(void)
{
	unsigned long long ns;
	asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ns));
	return ns;
}


__global__ void SparkGlm5NextMeshPublishKernel(
	volatile uint64_t *entry,
	unsigned long long *seq_cell,
	const unsigned long long *epoch_cell,
	unsigned long long *round_seq,
	uint64_t bytes,
	uint64_t slot_index,
	volatile uint64_t *slot_tail)
{
	unsigned long long sequence;
	unsigned long long tag;
	if ( threadIdx.x != 0u || blockIdx.x != 0u )
		return;
	sequence = 1ull + atomicAdd((unsigned long long *)seq_cell,1ull);
	tag = (epoch_cell[0] << 32ull) | (sequence & 0xffffffffull);
	round_seq[0] = tag;
	entry[2] = slot_index;
	entry[1] = bytes;
	__threadfence_system();
	*slot_tail = tag;
	__threadfence_system();
	entry[0] = tag;
}

__global__ void SparkGlm5NextMeshGuardKernel(
    volatile unsigned long long *error_word,
    unsigned long long *output)
{
	if ( threadIdx.x != 0u || blockIdx.x != 0u )
		return;
	if ( *error_word != 0ull )
	{
		output[0] = 0xFFFFFFFFFFFFFFFFull;
		*error_word = 0ull;
		printf("MESH-GUARD-POISON\\n");
	}
}

__global__ void SparkGlm5NextMeshWaitKernel(
	volatile uint64_t *band_base,
	uint64_t slot_bytes,
	const unsigned long long *round_seq,
	uint64_t slots_per_rank,
	uint64_t ring,
	uint32_t rank,
	uint32_t degree,
    unsigned long long *error_word,
    unsigned long long deadline_ns,
    unsigned long long *diag_word,
    volatile uint64_t *cancel_cell,
    const unsigned long long *cancel_expected)
{
	uint32_t peer;
	volatile uint64_t *end_word;
	uint64_t sequence;
	unsigned long long stop_at;
	if ( threadIdx.x != 0u || blockIdx.x != 0u )
		return;
	sequence = round_seq[0];
	stop_at = SparkGlm5NextGlobalTimerNs() + deadline_ns;
	if ( cancel_cell != 0 && cancel_expected != 0 &&
	     *cancel_cell != *cancel_expected )
		return;
	{
		unsigned long long spins = 0ull;
		unsigned long long spin_cap = deadline_ns / 200ull;
		if ( spin_cap < 1000000ull )
			spin_cap = 1000000ull;
		for ( peer = 0u; peer < degree - 1u; peer++ )
		{
			uint32_t peer_rank = peer < rank ? peer : peer + 1u;
			end_word = (volatile uint64_t *)
				((uint8_t *)band_base +
				((uint64_t)peer_rank * slots_per_rank +
					(ring & (slots_per_rank - 1ull))) * slot_bytes +
				slot_bytes - 8u);
			while ( *end_word < sequence )
			{
				if ( *error_word != 0ull )
					return;
				if ( cancel_cell != 0 && cancel_expected != 0 &&
				     *cancel_cell != *cancel_expected )
					return;
				spins++;
				if ( (spins & 4095ull) == 0ull &&
				     ( spins >= spin_cap ||
				       SparkGlm5NextGlobalTimerNs() >= stop_at ) )
				{
					unsigned long long off = (unsigned long long)
						((uint8_t *)end_word - (uint8_t *)band_base);
					unsigned long long got = *end_word;
					atomicExch((unsigned long long *)diag_word,
						((unsigned long long)peer_rank << 56ull) |
						((ring & 0xffull) << 48ull) |
						((off / slot_bytes) << 32ull) |
						((sequence & 0xffffull) << 16ull) |
						(got & 0xffffull));
					atomicExch((unsigned long long *)error_word,sequence);
					return;
				}
				__nanosleep(200u);
			}
		}
	}
}


static __device__ __forceinline__ float2 SparkGlm5NextLoadBf16Pair(const void *base,uint64_t element)
{
	uint32_t packed = ((const uint32_t *)base)[element];
	float2 pair;
	pair.x = __int_as_float((int32_t)((packed & UINT32_C(0x0000ffff)) << 16u));
	pair.y = __int_as_float((int32_t)(packed & UINT32_C(0xffff0000)));
	return(pair);
}

static __device__ __forceinline__ void SparkGlm5NextStoreBf16Pair(void *base,uint64_t element,float x,float y)
{
	uint32_t packed = ((uint32_t)(__float_as_int(y) & 0xffff0000u)) |
	    (uint32_t)((__float_as_int(x) >> 16) & 0x0000ffffu);
	((uint32_t *)base)[element] = packed;
}

static_assert(sizeof(SparkTpMeshRoundControl) ==
    SPARK_TP_MESH_ROUND_CONTROL_BYTES,"round control layout drift");

__global__ void SparkGlm5NextMeshRoundLoopKernel(
	volatile uint64_t *band_base,
	uint64_t slot_bytes,
	uint64_t slots_per_rank,
	volatile uint64_t *entry,
	volatile uint32_t *shipped_cell,
	volatile uint64_t *cancel_cell,
	SparkTpMeshRoundControl *control,
	uint32_t rank,
	uint32_t degree,
	const void *local_device,
	void *full_device,
	uint64_t bytes)
{
	__shared__ uint32_t s_active;
	__shared__ uint32_t s_decision;
	__shared__ uint64_t s_slot;
	__shared__ uint64_t s_parity_slot;
	__shared__ uint64_t s_tag;
	uint32_t tid = threadIdx.x;
	uint32_t nthreads = blockDim.x;
	uint64_t cursor = 0ull;
	uint64_t prev_tag = 0ull;
	uint64_t stop_at = 0ull;
	uint64_t spin_cap;
	if ( tid == 0u )
	{
		uint64_t now = SparkGlm5NextGlobalTimerNs();
		cursor = control->slot_cursor;
		prev_tag = control->round_seq;
		stop_at = now + control->deadline_ns;
		spin_cap = control->deadline_ns / 200ull;
		if ( spin_cap < 1000000ull )
			spin_cap = 1000000ull;
	}
	for ( ;; )
	{
		if ( tid == 0u )
		{
			s_active = control->rounds_done < control->rounds_total ? 1u : 0u;
			s_decision = SPARK_TP_MESH_ROUND_LOOP_DECISION_GO;
		}
		__syncthreads();
		if ( s_active == 0u )
			break;
		if ( tid == 0u && prev_tag != 0ull )
		{
			uint64_t spins = 0ull;
			while ( *shipped_cell != (uint32_t)prev_tag )
			{
				if ( *cancel_cell != control->cancel_expected )
				{
					s_decision = SPARK_TP_MESH_ROUND_LOOP_DECISION_CANCEL;
					break;
				}
				spins++;
				if ( (spins & 4095ull) == 0ull &&
				     ( spins >= spin_cap ||
				       SparkGlm5NextGlobalTimerNs() >= stop_at ) )
				{
					control->diag_word = (0xa5ull << 56ull) |
					    ((prev_tag & 0xffffull) << 16ull) |
					    (*shipped_cell & 0xffffull);
					control->error_word = prev_tag;
					printf("MESH-ROUNDLOOP-TIMEOUT rank=%u phase=ship-ack want=%u got=%u\\n",
						rank,(uint32_t)prev_tag,*shipped_cell);
					s_decision = SPARK_TP_MESH_ROUND_LOOP_DECISION_TIMEOUT;
					break;
				}
				__nanosleep(200u);
			}
		}
		__syncthreads();
		if ( s_decision != SPARK_TP_MESH_ROUND_LOOP_DECISION_GO )
			break;
		if ( tid == 0u )
		{
			s_slot = ((uint64_t)rank * slots_per_rank +
			    (cursor & (slots_per_rank - 1ull))) * slot_bytes;
			s_parity_slot =
			    (cursor & (slots_per_rank - 1ull)) * slot_bytes;
			s_tag = (control->epoch << 32ull) |
			    ((control->seq + 1ull) & 0xffffffffull);
		}
		__syncthreads();
		{
			volatile uint64_t *destination = (volatile uint64_t *)
			    ((uint8_t *)band_base + s_slot);
			const uint64_t *source = (const uint64_t *)local_device;
			uint64_t quads = (bytes + 7ull) >> 3ull;
			uint64_t quad;
			for ( quad = tid; quad < quads; quad += nthreads )
				destination[quad] = source[quad];
			__threadfence_system();
		}
		__syncthreads();
		if ( tid == 0u )
		{
			volatile uint64_t *tail = (volatile uint64_t *)
			    ((uint8_t *)band_base + s_slot + slot_bytes - 8ull);
			entry[2] = s_slot / slot_bytes;
			entry[1] = bytes;
			control->seq = control->seq + 1ull;
			control->round_seq = s_tag;
			__threadfence_system();
			*tail = s_tag;
			__threadfence_system();
			entry[0] = s_tag;
		}
		if ( tid == 0u )
		{
			uint32_t peer;
			uint64_t spins = 0ull;
			for ( peer = 0u; peer < degree - 1u; peer++ )
			{
				uint32_t peer_rank = peer < rank ? peer : peer + 1u;
				volatile uint64_t *end_word = (volatile uint64_t *)
				    ((uint8_t *)band_base +
				    (((uint64_t)peer_rank * slots_per_rank +
				      (cursor & (slots_per_rank - 1ull))) *
					slot_bytes) + slot_bytes - 8ull);
				while ( *end_word < s_tag )
				{
					if ( *cancel_cell != control->cancel_expected )
					{
						s_decision =
						    SPARK_TP_MESH_ROUND_LOOP_DECISION_CANCEL;
						break;
					}
					spins++;
					if ( (spins & 4095ull) == 0ull &&
					     ( spins >= spin_cap ||
					       SparkGlm5NextGlobalTimerNs() >= stop_at ) )
					{
						control->diag_word =
						    ((unsigned long long)peer_rank << 56ull) |
						    ((cursor & 0xffull) << 48ull) |
						    ((unsigned long long)
							(s_slot / slot_bytes) << 32ull) |
						    ((s_tag & 0xffffull) << 16ull) |
						    (*end_word & 0xffffull);
						control->error_word = s_tag;
						printf("MESH-ROUNDLOOP-TIMEOUT rank=%u phase=peer-wait peer=%u want=%u got=%u\\n",
						    rank,peer_rank,
						    (uint32_t)s_tag,
						    (uint32_t)*end_word);
						s_decision =
						    SPARK_TP_MESH_ROUND_LOOP_DECISION_TIMEOUT;
						break;
					}
					__nanosleep(200u);
				}
				if ( s_decision != SPARK_TP_MESH_ROUND_LOOP_DECISION_GO )
					break;
			}
		}
		__syncthreads();
		if ( s_decision != SPARK_TP_MESH_ROUND_LOOP_DECISION_GO )
			break;
		__threadfence();
		{
			uint64_t pairs = bytes >> 2ull;
			uint64_t pair;
			uint32_t source;
			for ( pair = tid; pair < pairs; pair += nthreads )
			{
				float acc_x = 0.0f;
				float acc_y = 0.0f;
				for ( source = 0u; source < degree; source++ )
				{
					float2 part = SparkGlm5NextLoadBf16Pair(
					    (const void *)((uint8_t *)band_base +
					    (uint64_t)source * slots_per_rank *
					        slot_bytes + s_parity_slot),pair);
					acc_x += part.x;
					acc_y += part.y;
				}
				SparkGlm5NextStoreBf16Pair(full_device,pair,acc_x,acc_y);
			}
		}
		__syncthreads();
		if ( tid == 0u )
		{
			cursor = cursor + 1ull;
			prev_tag = s_tag;
			control->slot_cursor = cursor;
			control->rounds_done = control->rounds_done + 1ull;
		}
		__syncthreads();
	}
}

extern "C" cudaError_t SparkGlm5NextLaunchMeshRoundLoop(cudaStream_t stream,
	volatile void *band_base,uint64_t slot_bytes,uint64_t slots_per_rank,
	volatile void *entry,void *shipped_cell,volatile void *cancel_cell,
	void *round_control,uint32_t rank,uint32_t degree,
	const void *local_device,void *full_device,uint64_t bytes)
{
	SparkGlm5NextMeshRoundLoopKernel<<<1,SPARK_TP_MESH_THREADS,0u,stream>>>(
		(volatile uint64_t *)band_base,slot_bytes,slots_per_rank,
		(volatile uint64_t *)entry,(volatile uint32_t *)shipped_cell,
		(volatile uint64_t *)cancel_cell,
		(SparkTpMeshRoundControl *)round_control,rank,degree,
		local_device,full_device,bytes);
	return cudaPeekAtLastError();
}


struct SparkGlm5NextRankSources
{
	const void *pointer[16u];
};

static __global__ void SparkGlm5NextSumRanksF32Kernel(
    void *destination_bf16,
    SparkGlm5NextRankSources sources,
    uint32_t source_count,
    uint32_t pair_count)
{
	uint32_t pair;
	float2 acc,v;
	for (pair=blockIdx.x*blockDim.x+threadIdx.x; pair<pair_count; pair+=blockDim.x*gridDim.x)
	{
		uint32_t source;
		acc.x = 0.0f;
		acc.y = 0.0f;
		for ( source = 0u; source < source_count; source++ )
		{
			v = SparkGlm5NextLoadBf16Pair(sources.pointer[source],pair);
			acc.x += v.x;
			acc.y += v.y;
		}
		SparkGlm5NextStoreBf16Pair(destination_bf16,pair,acc.x,acc.y);
	}
}

extern "C" cudaError_t SparkGlm5NextLaunchSumRanksF32(cudaStream_t stream,
    void *destination,const void *const *sources,uint32_t source_count,
    uint32_t element_count)
{
	SparkGlm5NextRankSources by_value;
	uint32_t index;
	if ( destination == 0 || sources == 0 || source_count == 0u ||
	     source_count > 16u || element_count == 0u )
		return(cudaErrorInvalidValue);
	for ( index = 0u; index < source_count; index++ )
		by_value.pointer[index] = sources[index];
	{
		dim3 grid;
		uint32_t pairs = (element_count + 1u) / 2u;
		uint32_t rows = (pairs + 255u) / 256u;
		grid = dim3(rows < 1u ? 1u : rows,1u,1u);
		SparkGlm5NextSumRanksF32Kernel<<<grid,256u,0u,stream>>>(
		    destination,by_value,source_count,pairs);
	}
	return cudaPeekAtLastError();
}

static __global__ void SparkGlm5NextSeedF32Kernel(
    float *destination_f32,
    const void *source_a_bf16,
    const void *source_b_bf16,
    uint32_t pair_count)
{
	uint32_t pair;
	float2 a,b;
	for (pair=threadIdx.x; pair<pair_count; pair+=blockDim.x)
	{
		a = SparkGlm5NextLoadBf16Pair(source_a_bf16,pair);
		b = SparkGlm5NextLoadBf16Pair(source_b_bf16,pair);
		destination_f32[2u * pair] = a.x + b.x;
		destination_f32[2u * pair + 1u] = a.y + b.y;
	}
}

static __global__ void SparkGlm5NextAddF32Kernel(
    float *destination_f32,
    const void *source_bf16,
    uint32_t pair_count)
{
	uint32_t pair;
	float2 b;
	for (pair=threadIdx.x; pair<pair_count; pair+=blockDim.x)
	{
		b = SparkGlm5NextLoadBf16Pair(source_bf16,pair);
		destination_f32[2u * pair] += b.x;
		destination_f32[2u * pair + 1u] += b.y;
	}
}

static __global__ void SparkGlm5NextRoundF32Kernel(
    void *destination_bf16,
    const float *source_f32,
    uint32_t pair_count)
{
	uint32_t pair;
	float2 v;
	for (pair=threadIdx.x; pair<pair_count; pair+=blockDim.x)
	{
		v.x = source_f32[2u * pair];
		v.y = source_f32[2u * pair + 1u];
		SparkGlm5NextStoreBf16Pair(destination_bf16,pair,v.x,v.y);
	}
}

extern "C" cudaError_t SparkGlm5NextLaunchSeedF32(cudaStream_t stream,
    float *destination,const void *a,const void *b,uint32_t element_count)
{
	SparkGlm5NextSeedF32Kernel<<<1,SPARK_TP_MESH_THREADS,0u,stream>>>(
	    destination,a,b,(element_count + 1u) / 2u);
	return cudaPeekAtLastError();
}

extern "C" cudaError_t SparkGlm5NextLaunchAddF32(cudaStream_t stream,
    float *destination,const void *b,uint32_t element_count)
{
	SparkGlm5NextAddF32Kernel<<<1,SPARK_TP_MESH_THREADS,0u,stream>>>(
	    destination,b,(element_count + 1u) / 2u);
	return cudaPeekAtLastError();
}

extern "C" cudaError_t SparkGlm5NextLaunchRoundF32(cudaStream_t stream,
    void *destination,const float *source,uint32_t element_count)
{
	SparkGlm5NextRoundF32Kernel<<<1,SPARK_TP_MESH_THREADS,0u,stream>>>(
	    destination,source,(element_count + 1u) / 2u);
	return cudaPeekAtLastError();
}

static __global__ void SparkGlm5NextAccumAddKernel(
	void *destination_bf16,
	const void *source_bf16,
	uint32_t row_count,
	uint32_t width)
{
	uint32_t row = blockIdx.x,element;
	uint64_t offset = ((uint64_t)row * width) >> 1u;
	float2 destination_pair,source_pair;
	if ( row >= row_count )
		return;
	for (element=threadIdx.x; element<(width >> 1u); element+=blockDim.x)
	{
		destination_pair = SparkGlm5NextLoadBf16Pair(destination_bf16,offset + element);
		source_pair = SparkGlm5NextLoadBf16Pair(source_bf16,offset + element);
		SparkGlm5NextStoreBf16Pair(destination_bf16,offset + element,destination_pair.x + source_pair.x,destination_pair.y + source_pair.y);
	}
}

static __global__ void SparkGlm5NextAccumU64MaxKernel(
	uint64_t *destination,
	const uint64_t *source,
	uint32_t element_count)
{
	uint32_t element;
	element = blockIdx.x * blockDim.x + threadIdx.x;
	if ( element < element_count && source[element] > destination[element] )
		destination[element] = source[element];
}

extern "C" cudaError_t SparkGlm5NextLaunchAccumAdd(cudaStream_t stream,void *destination_bf16,const void *source_bf16,uint32_t row_count,uint32_t width)
{
	if ( destination_bf16 == 0 || source_bf16 == 0 || row_count == 0u || width == 0u || (width & 1u) != 0u )
		return(cudaErrorInvalidValue);
	SparkGlm5NextAccumAddKernel<<<row_count,256u,0u,stream>>>(destination_bf16,source_bf16,row_count,width);
	return(cudaPeekAtLastError());
}

extern "C" cudaError_t SparkGlm5NextLaunchAccumU64Max(cudaStream_t stream,uint64_t *destination,const uint64_t *source,uint32_t element_count)
{
	if ( destination == 0 || source == 0 || element_count == 0u )
		return(cudaErrorInvalidValue);
	SparkGlm5NextAccumU64MaxKernel<<<(element_count + 255u) / 256u,256u,0u,stream>>>(destination,source,element_count);
	return(cudaPeekAtLastError());
}

extern "C" cudaError_t SparkGlm5NextLaunchMeshGuard(cudaStream_t stream,
	volatile void *error_word,void *output)
{
	SparkGlm5NextMeshGuardKernel<<<1,32,0u,stream>>>(
		(volatile unsigned long long *)error_word,
		(unsigned long long *)output);
	return cudaPeekAtLastError();
}

__global__ void SparkGlm5NextMeshCopyDownKernel(
    volatile uint64_t *destination,
    const uint64_t *source,
    uint32_t quad_count)
{
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i < quad_count )
		destination[i] = source[i];
}

extern "C" cudaError_t SparkGlm5NextLaunchMeshCopyDown(
	cudaStream_t stream,void *destination,const void *source,
	uint64_t bytes)
{
	uint32_t quads = (uint32_t)((bytes + 7u) / 8u);
	if ( destination == 0 || source == 0 || quads == 0u )
		return(cudaErrorInvalidValue);
	SparkGlm5NextMeshCopyDownKernel<<<(quads + 255u) / 256u,256u,0u,stream>>>(
		(volatile uint64_t *)destination,
		(const uint64_t *)source,quads);
	return(cudaPeekAtLastError());
}

extern "C" cudaError_t SparkGlm5NextLaunchMeshPublish(cudaStream_t stream,
	volatile void *entry,void *seq_cell,const void *epoch_cell,
	void *round_seq,uint64_t bytes,
	uint64_t slot_index,volatile void *slot_tail)
{
	SparkGlm5NextMeshPublishKernel<<<1,32,0u,stream>>>(
		(volatile uint64_t *)entry,(unsigned long long *)seq_cell,
		(const unsigned long long *)epoch_cell,
		(unsigned long long *)round_seq,bytes,slot_index,
		(volatile uint64_t *)slot_tail);
	return(cudaPeekAtLastError());
}

extern "C" cudaError_t SparkGlm5NextLaunchMeshWait(cudaStream_t stream,
	volatile void *band_base,uint64_t slot_bytes,const void *round_seq,
	uint64_t slots_per_rank,uint64_t ring,uint32_t rank,uint32_t degree,
	void *error_word,unsigned long long deadline_ns,void *diag_word,
	volatile void *cancel_cell,const void *cancel_expected)
{
	SparkGlm5NextMeshWaitKernel<<<1,32,0u,stream>>>(
		(volatile uint64_t *)band_base,slot_bytes,
		(const unsigned long long *)round_seq,slots_per_rank,ring,rank,
		degree,(unsigned long long *)error_word,deadline_ns,
		(unsigned long long *)diag_word,
		(volatile uint64_t *)cancel_cell,
		(const unsigned long long *)cancel_expected);
	return(cudaPeekAtLastError());
}








#endif
