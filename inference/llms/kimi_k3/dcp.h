#pragma once


#include <math.h>
#include <stdint.h>

#define K3_DCP_MAX_DEGREE 8u

struct K3DcpPlan
{
	uint32_t degree;
	uint32_t rank;
	uint32_t begin;
	uint32_t end;
};

static void K3DcpPlanRange(struct K3DcpPlan *plan, uint32_t context_length,
	uint32_t degree, uint32_t rank)
{
	uint32_t base,extra;
	if ( plan == 0 )
		return;
	if ( degree == 0u || degree > K3_DCP_MAX_DEGREE )
		degree = 1u;
	if ( rank >= degree )
		rank = 0u;
	if ( context_length == 0u || degree == 1u )
	{
		plan->degree = 1u;
		plan->rank = 0u;
		plan->begin = 0u;
		plan->end = context_length;
		return;
	}
	base = context_length / degree;
	extra = context_length % degree;
	plan->degree = degree;
	plan->rank = rank;
	plan->begin = rank * base + (rank < extra ? rank : extra);
	plan->end = plan->begin + base + (rank < extra ? 1u : 0u);
}

static uint32_t K3DcpPlanCoverage(const struct K3DcpPlan *plans, uint32_t degree)
{
	uint32_t rank,covered;
	if ( plans == 0 || degree == 0u || degree > K3_DCP_MAX_DEGREE )
		return(0u);
	covered = 0u;
	for ( rank = 0u; rank < degree; ++rank )
	{
		if ( plans[rank].degree != degree || plans[rank].rank != rank )
			return(0u);
		if ( plans[rank].begin != covered )
			return(0u);
		if ( plans[rank].end < plans[rank].begin )
			return(0u);
		covered = plans[rank].end;
	}
	return(covered);
}

static float K3DcpMerge2(float max_a,float sum_a,float acc_a,
	float max_b,float sum_b,float acc_b,float *merged_max,
	float *merged_sum)
{
	float global_max,weight_a,weight_b,total;
	global_max = max_a > max_b ? max_a : max_b;
	weight_a = expf(max_a - global_max);
	weight_b = expf(max_b - global_max);
	total = sum_a * weight_a + sum_b * weight_b;
	*merged_max = global_max;
	*merged_sum = total;
	return acc_a * weight_a + acc_b * weight_b;
}
