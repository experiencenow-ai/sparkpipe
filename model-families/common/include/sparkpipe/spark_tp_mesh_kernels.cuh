#pragma once
#include <stdint.h>
#include <stddef.h>

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#include <stdio.h>
#include "sparkpipe/spark_tp_mesh_round_control.h"
#define SPARK_TP_MESH_KERNELS_MARKER "SPARK-TP-MESH-KERNELS-V8-MASKED-FP32-TREE"
#define SPARK_TP_MESH_ERROR_PARITY_MISMATCH 0xFFFFFFFFFF000000ull
#define SPARK_TP_MESH_ERROR_CANCELLED 0xFFFFFFFFFE000000ull
#if defined(__CUDACC__)
__constant__ char SparkTpMeshKernelsBuildMarker[] =
    SPARK_TP_MESH_KERNELS_MARKER;
#endif

#define SPARK_TP_MESH_THREADS 256u

static __device__ __forceinline__ unsigned long long SparkGlm5NextLdcvU64(
    const volatile void *address)
{
	unsigned long long value;
	asm volatile("ld.global.cv.u64 %0,[%1];"
	    : "=l"(value) : "l"(address) : "memory");
	return(value);
}

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
    uint64_t slots_per_rank,
    volatile uint64_t *slot_tail,
    unsigned long long *error_word,uint32_t peer_mask)
{
	unsigned long long sequence;
	unsigned long long tag;
	uint64_t ring;
	if ( threadIdx.x != 0u || blockIdx.x != 0u )
		return;
	if ( SparkGlm5NextLdcvU64(error_word) != 0ull )
		return;
	sequence = 1ull + atomicAdd((unsigned long long *)seq_cell,1ull);
	ring = (sequence - 1ull) & (slots_per_rank - 1ull);
	if ( (slot_index & (slots_per_rank - 1ull)) != ring )
	{
		atomicExch((unsigned long long *)error_word,
		    SPARK_TP_MESH_ERROR_PARITY_MISMATCH | sequence);
		return;
	}
	tag = (epoch_cell[0] << 32ull) | (sequence & 0xffffffffull);
	round_seq[0] = tag;
	entry[2] = slot_index;
	entry[1] = bytes;
	entry[3] = peer_mask;
	__threadfence_system();
	*slot_tail = tag;
	__threadfence_system();
	entry[0] = tag;
}

__global__ void SparkGlm5NextMeshSeqPadKernel(
    unsigned long long *seq_cell)
{
	if ( threadIdx.x != 0u || blockIdx.x != 0u )
		return;
	(void)atomicAdd((unsigned long long *)seq_cell,1ull);
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
		printf("MESH-GUARD-POISON\\n");
	}
}

__global__ void SparkGlm5NextMeshWaitKernel(
    volatile uint64_t *band_base,
    uint64_t slot_bytes,
    const unsigned long long *round_seq,
    uint64_t slots_per_rank,
    uint32_t rank,
    uint32_t degree,
    unsigned long long *error_word,
    unsigned long long deadline_ns,
    unsigned long long *diag_word,
    volatile uint64_t *cancel_cell,
    const unsigned long long *cancel_expected,
    unsigned long long *arrival_ring)
{
	uint32_t peer;
	volatile uint64_t *end_word;
	uint64_t sequence;
	uint64_t ring;
	unsigned long long stop_at;
	if ( threadIdx.x != 0u || blockIdx.x != 0u )
		return;
	sequence = SparkGlm5NextLdcvU64(round_seq);
	ring = (sequence - 1ull) & (slots_per_rank - 1ull);
	stop_at = SparkGlm5NextGlobalTimerNs() + deadline_ns;
	if ( cancel_cell != 0 && cancel_expected != 0 &&
	     *cancel_cell != *cancel_expected )
	{
		atomicExch((unsigned long long *)error_word,
		    SPARK_TP_MESH_ERROR_CANCELLED | sequence);
		return;
	}
	{
		unsigned long long spins = 0ull;
		unsigned long long spin_cap = deadline_ns / 200ull;
		if ( spin_cap < 1000000ull )
			spin_cap = 1000000ull;
		for ( peer = 0u; peer < degree - 1u; peer++ )
		{
			uint32_t peer_rank = peer < rank ? peer : peer + 1u;
			uint64_t peer_epoch;
			end_word = (volatile uint64_t *)
				((uint8_t *)band_base +
				((uint64_t)peer_rank * slots_per_rank +
					ring) * slot_bytes +
				slot_bytes - 8u);
			while ( (peer_epoch = SparkGlm5NextLdcvU64(end_word) >>
			             32ull) != (sequence >> 32ull) ||
			        SparkGlm5NextLdcvU64(end_word) < sequence )
			{
				if ( SparkGlm5NextLdcvU64(error_word) != 0ull )
					return;
				if ( cancel_cell != 0 && cancel_expected != 0 &&
				     SparkGlm5NextLdcvU64(cancel_cell) !=
				     SparkGlm5NextLdcvU64(cancel_expected) )
				{
					atomicExch((unsigned long long *)error_word,
					    SPARK_TP_MESH_ERROR_CANCELLED | sequence);
					return;
				}
				spins++;
				if ( (spins & 255ull) == 0ull &&
				     ( spins >= spin_cap ||
				       SparkGlm5NextGlobalTimerNs() >= stop_at ) )
				{
					unsigned long long off = (unsigned long long)
						((uint8_t *)end_word - (uint8_t *)band_base);
					unsigned long long got = SparkGlm5NextLdcvU64(end_word);
					atomicExch((unsigned long long *)diag_word,
						((unsigned long long)peer_rank << 56ull) |
						((ring & 0xffull) << 48ull) |
						((off / slot_bytes) << 32ull) |
						((sequence & 0xffffull) << 16ull) |
						(got & 0xffffull));
					atomicExch((unsigned long long *)error_word,sequence);
					return;
				}
			}
		}
	}
	if ( arrival_ring != 0 && threadIdx.x == 0u && blockIdx.x == 0u )
		arrival_ring[sequence & 255ull] = SparkGlm5NextGlobalTimerNs();
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
	volatile uint64_t *shipped_cell,
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
			while ( *shipped_cell != prev_tag )
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
					printf("MESH-ROUNDLOOP-TIMEOUT rank=%u phase=ship-ack want=%llu got=%llu\\n",
						rank,(unsigned long long)prev_tag,(unsigned long long)*shipped_cell);
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
			uint64_t quads = bytes >> 3ull;
			uint64_t quad;
			for ( quad = tid; quad < quads; quad += nthreads )
				destination[quad] = source[quad];
            if ( tid == 0u )
                for ( uint64_t tail = quads * 8u; tail < bytes; tail++ )
                    ((volatile uint8_t *)destination)[tail] = ((const uint8_t *)source)[tail];
			__threadfence_system();
		}
		__syncthreads();
		if ( tid == 0u )
		{
			volatile uint64_t *tail = (volatile uint64_t *)
			    ((uint8_t *)band_base + s_slot + slot_bytes - 8ull);
			entry[2] = s_slot / slot_bytes;
			entry[1] = bytes;
			entry[3] = ((1u << degree) - 1u) & ~(1u << rank);
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
				while ( (*end_word >> 32ull) != (s_tag >> 32ull) ||
				        *end_word < s_tag )
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

static __device__ bool SparkGlm5NextMeshTreeWait(
    const volatile uint64_t *cell,uint64_t tag,
    const volatile uint64_t *cancel,SparkTpMeshRoundControl *control,
    uint64_t deadline)
{
    for (;;)
    {
        if ( SparkGlm5NextLdcvU64(&control->error_word) != 0u ) return false;
        if ( SparkGlm5NextLdcvU64(cancel) != control->cancel_expected )
        {
            control->error_word = SPARK_TP_MESH_ERROR_CANCELLED | tag;
            return false;
        }
        if ( tag == 0u || SparkGlm5NextLdcvU64(cell) == tag ) return true;
        if ( SparkGlm5NextGlobalTimerNs() >= deadline )
        {
            control->error_word = tag;
            control->diag_word = SparkGlm5NextLdcvU64(cell);
            return false;
        }
        __nanosleep(200u);
    }
}

__global__ void SparkGlm5NextMeshTreeKernel(
    uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,
    volatile uint64_t *entry,const volatile uint64_t *shipped,
    const volatile uint64_t *cancel,SparkTpMeshRoundControl *control,
    uint32_t rank,uint32_t degree,const void *local,void *output,void *scratch,
    uint64_t elements,uint32_t operation,uint32_t rounds,
    uint64_t timeout_ns)
{
    __shared__ uint32_t ready;
    __shared__ uint64_t tag;
    uint32_t tid = threadIdx.x;
    uint32_t levels = SparkTpMeshTreeLevels(degree);
    uint32_t width = operation == 2u ? 8u : operation == 1u ? 4u : 2u;
    uint64_t capacity = (slot_bytes - 16u) / width;
    uint64_t deadline = SparkGlm5NextGlobalTimerNs() + timeout_ns;
    for ( uint32_t round = 0u; round < rounds; round++ )
    {
        for ( uint64_t begin = 0u; begin < elements; begin += capacity )
        {
            uint64_t count = elements - begin < capacity ? elements - begin : capacity;
            for ( uint64_t i = tid; i < count; i += blockDim.x )
            {
                uint64_t index = begin + i;
                if ( operation == 1u )
                    ((float *)scratch)[i] = __uint_as_float((uint32_t)((const uint16_t *)local)[index] << 16u);
                else if ( operation == 2u )
                    ((uint64_t *)scratch)[i] = ((const uint64_t *)local)[index];
                else
                {
                    uint32_t owner = (uint32_t)(index / (elements / degree));
                    uint64_t source = index % (elements / degree);
                    ((uint16_t *)scratch)[i] = owner == rank ? ((const uint16_t *)local)[source] : 0u;
                }
            }
            __syncthreads();
            for ( uint32_t phase = 0u; phase < 2u * levels; phase++ )
            {
                uint32_t route = SparkTpMeshTreeRoute(rank,degree,phase);
                uint32_t send = route >> 16u;
                uint32_t receive = route & 0xffffu;
                if ( tid == 0u )
                {
                    ready = control->seq < UINT32_MAX && control->error_word == 0u;
                    if ( ready == 0u ) control->error_word = UINT64_MAX;
                    tag = (control->epoch << 32u) | (control->seq + 1u);
                    if ( send != 0u && ready != 0u )
                        ready = SparkGlm5NextMeshTreeWait(shipped,control->round_seq,cancel,control,deadline);
                }
                __syncthreads();
                if ( ready == 0u ) return;
                uint64_t ring = (tag - 1u) & (slots_per_rank - 1u);
                uint64_t own_slot = ((uint64_t)rank * slots_per_rank + ring) * slot_bytes;
                if ( send != 0u )
                {
                    for ( uint64_t i = tid; i < count * width; i += blockDim.x )
                        ((volatile uint8_t *)band)[own_slot + i] = ((const uint8_t *)scratch)[i];
                    __threadfence_system();
                    __syncthreads();
                    if ( tid == 0u )
                    {
                        entry[1] = count * width;
                        entry[2] = own_slot / slot_bytes;
                        entry[3] = 1u << (send - 1u);
                        control->round_seq = tag;
                        __threadfence_system();
                        *(volatile uint64_t *)(band + own_slot + slot_bytes - 8u) = tag;
                        __threadfence_system();
                        entry[0] = tag;
                    }
                }
                if ( receive != 0u )
                {
                    uint8_t *source = band + ((uint64_t)(receive - 1u) * slots_per_rank + ring) * slot_bytes;
                    if ( tid == 0u )
                        ready = SparkGlm5NextMeshTreeWait((const volatile uint64_t *)(source + slot_bytes - 8u),tag,cancel,control,deadline);
                    __syncthreads();
                    if ( ready == 0u ) return;
                    __threadfence_system();
                    for ( uint64_t i = tid; i < count; i += blockDim.x )
                    {
                        if ( operation == 1u )
                            ((float *)scratch)[i] = phase < levels ?
                                ((float *)scratch)[i] + ((const volatile float *)source)[i] : ((const volatile float *)source)[i];
                        else if ( operation == 2u )
                        {
                            uint64_t value = ((const volatile uint64_t *)source)[i];
                            if ( phase >= levels || value > ((uint64_t *)scratch)[i] ) ((uint64_t *)scratch)[i] = value;
                        }
                        else
                            ((uint16_t *)scratch)[i] = phase < levels ?
                                ((uint16_t *)scratch)[i] | ((const volatile uint16_t *)source)[i] : ((const volatile uint16_t *)source)[i];
                    }
                }
                __syncthreads();
                if ( tid == 0u ) control->seq++;
                __syncthreads();
            }
            for ( uint64_t i = tid; i < count; i += blockDim.x )
            {
                if ( operation == 1u ) ((uint16_t *)output)[begin + i] = (uint16_t)(__float_as_uint(((float *)scratch)[i]) >> 16u);
                else if ( operation == 2u ) ((uint64_t *)output)[begin + i] = ((uint64_t *)scratch)[i];
                else ((uint16_t *)output)[begin + i] = ((uint16_t *)scratch)[i];
            }
            __syncthreads();
        }
        if ( tid == 0u )
        {
            control->slot_cursor = control->seq;
            control->rounds_done++;
        }
        __syncthreads();
    }
}

extern "C" cudaError_t SparkGlm5NextLaunchMeshTree(cudaStream_t stream,
    void *band,uint64_t slot_bytes,uint64_t slots_per_rank,volatile void *entry,
    const volatile void *shipped,const volatile void *cancel,void *round_control,
    uint32_t rank,uint32_t degree,const void *local,void *output,void *scratch,
    uint64_t elements,uint32_t operation,uint32_t rounds,uint64_t timeout_ns)
{
    SparkGlm5NextMeshTreeKernel<<<1,SPARK_TP_MESH_THREADS,0u,stream>>>(
        (uint8_t *)band,slot_bytes,slots_per_rank,(volatile uint64_t *)entry,
        (const volatile uint64_t *)shipped,(const volatile uint64_t *)cancel,
        (SparkTpMeshRoundControl *)round_control,rank,degree,local,output,scratch,
        elements,operation,rounds,timeout_ns);
    return cudaPeekAtLastError();
}

extern "C" cudaError_t SparkGlm5NextLaunchMeshRoundLoop(cudaStream_t stream,
	volatile void *band_base,uint64_t slot_bytes,uint64_t slots_per_rank,
	volatile void *entry,void *shipped_cell,volatile void *cancel_cell,
	void *round_control,uint32_t rank,uint32_t degree,
	const void *local_device,void *full_device,uint64_t bytes)
{
	SparkGlm5NextMeshRoundLoopKernel<<<1,SPARK_TP_MESH_THREADS,0u,stream>>>(
		(volatile uint64_t *)band_base,slot_bytes,slots_per_rank,
		(volatile uint64_t *)entry,(volatile uint64_t *)shipped_cell,
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
    volatile uint8_t *destination,
    const uint8_t *source,
    uint64_t bytes,
    const volatile uint64_t *shipped,
    SparkTpMeshRoundControl *control,
    const volatile uint64_t *cancel,
    uint64_t timeout_ns)
{
    __shared__ uint32_t ready;
    if ( threadIdx.x == 0u )
    {
        uint64_t previous = control->round_seq;
        uint64_t deadline = SparkGlm5NextGlobalTimerNs() + timeout_ns;
        ready = 1u;
        while ( previous != 0ull && SparkGlm5NextLdcvU64(shipped) != previous )
        {
            if ( SparkGlm5NextLdcvU64(&control->error_word) != 0ull ||
                 SparkGlm5NextLdcvU64(cancel) != control->cancel_expected )
            {
                atomicExch((unsigned long long *)&control->error_word,
                    SPARK_TP_MESH_ERROR_CANCELLED | previous);
                ready = 0u;
                break;
            }
            if ( SparkGlm5NextGlobalTimerNs() >= deadline )
            {
                atomicExch((unsigned long long *)&control->error_word,previous);
                ready = 0u;
                break;
            }
            __nanosleep(200u);
        }
        if ( SparkGlm5NextLdcvU64(cancel) != control->cancel_expected )
            atomicExch((unsigned long long *)&control->error_word,
                SPARK_TP_MESH_ERROR_CANCELLED | previous);
        if ( SparkGlm5NextLdcvU64(&control->error_word) != 0ull )
            ready = 0u;
    }
    __syncthreads();
    if ( ready == 0u )
        return;
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t quads = bytes / 8u;
    if ( i < quads )
        ((volatile uint64_t *)destination)[i] = ((const uint64_t *)source)[i];
    if ( i == quads )
        for ( uint64_t tail = quads * 8u; tail < bytes; tail++ )
            destination[tail] = source[tail];
}

extern "C" cudaError_t SparkGlm5NextLaunchMeshCopyDown(
    cudaStream_t stream,void *destination,const void *source,
    uint64_t bytes,const volatile void *shipped,void *round_control,
    const volatile void *cancel,uint64_t timeout_ns)
{
    uint64_t quads = bytes / 8u + 1u;
    if ( destination == 0 || source == 0 || bytes == 0u || shipped == 0 ||
         round_control == 0 || cancel == 0 || timeout_ns == 0u )
        return(cudaErrorInvalidValue);
    SparkGlm5NextMeshCopyDownKernel<<<(quads + 255u) / 256u,256u,0u,stream>>>(
        (volatile uint8_t *)destination,(const uint8_t *)source,bytes,
        (const volatile uint64_t *)shipped,(SparkTpMeshRoundControl *)round_control,
        (const volatile uint64_t *)cancel,timeout_ns);
    return(cudaPeekAtLastError());
}

extern "C" cudaError_t SparkGlm5NextLaunchMeshPublish(cudaStream_t stream,
    volatile void *entry,void *seq_cell,const void *epoch_cell,
    void *round_seq,uint64_t bytes,
    uint64_t slot_index,uint64_t slots_per_rank,volatile void *slot_tail,
    void *error_word,uint32_t peer_mask)
{
	SparkGlm5NextMeshPublishKernel<<<1,32,0u,stream>>>(
		(volatile uint64_t *)entry,(unsigned long long *)seq_cell,
		(const unsigned long long *)epoch_cell,
		(unsigned long long *)round_seq,bytes,slot_index,slots_per_rank,
		(volatile uint64_t *)slot_tail,
		(unsigned long long *)error_word,peer_mask);
	return cudaPeekAtLastError();
}

extern "C" cudaError_t SparkGlm5NextLaunchMeshSeqPad(cudaStream_t stream,
    void *seq_cell)
{
	SparkGlm5NextMeshSeqPadKernel<<<1,32,0u,stream>>>(
		(unsigned long long *)seq_cell);
	return cudaPeekAtLastError();
}

extern "C" cudaError_t SparkGlm5NextLaunchMeshWait(cudaStream_t stream,
    volatile void *band_base,uint64_t slot_bytes,const void *round_seq,
    uint64_t slots_per_rank,uint32_t rank,uint32_t degree,
    void *error_word,unsigned long long deadline_ns,void *diag_word,
    volatile void *cancel_cell,const void *cancel_expected,
    void *arrival_ring)
{
	SparkGlm5NextMeshWaitKernel<<<1,32,0u,stream>>>(
		(volatile uint64_t *)band_base,slot_bytes,
		(const unsigned long long *)round_seq,slots_per_rank,rank,
		degree,(unsigned long long *)error_word,deadline_ns,
		(unsigned long long *)diag_word,
		(volatile uint64_t *)cancel_cell,
		(const unsigned long long *)cancel_expected,
		(unsigned long long *)arrival_ring);
	return cudaPeekAtLastError();
}








#endif
