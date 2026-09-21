#pragma once


#include <stdint.h>

template<uint32_t THREADS, uint32_t MAX_DRAFT>
__global__ __launch_bounds__(THREADS, 1)
void LmSpecVerifyKernel(const uint32_t *__restrict__ row_tokens,
	const uint32_t *__restrict__ sequence_row_begin,
	const uint32_t *__restrict__ draft_tokens,
	const uint32_t *__restrict__ draft_counts, uint32_t draft_stride,
	uint32_t *__restrict__ accepted_out, uint32_t *__restrict__ bonus_out)
{
	uint32_t sequence = blockIdx.x;
	uint32_t begin = sequence_row_begin[sequence];
	uint32_t draft_count = draft_counts[sequence];
	uint32_t accepted = 0u;
	uint32_t bonus;
	const uint32_t *draft = draft_tokens + ((uint64_t)sequence * draft_stride);
	if ( threadIdx.x != 0u )
		return;
	if ( draft_count > MAX_DRAFT )
		draft_count = MAX_DRAFT;
	while ( accepted < draft_count
		&& row_tokens[begin + accepted] == draft[accepted] )
		accepted++;
	bonus = row_tokens[begin + accepted];
	accepted_out[sequence] = accepted;
	bonus_out[sequence] = bonus;
}
