#pragma once

#include "inference/kernels/scale.cuh"
#include <stdint.h>

struct LmGemmArguments
{
    LmScaleTensor scale_a;
    LmScaleTensor scale_b;
    uint32_t prefix_built;
    const uint32_t *group_row_offset;
    uint32_t *group_tile_prefix;
    // THE REAL FIELDS, COPIED. The indirect-A words the kernel contract added
    // (route.cuh): null on every dense launch, and the recorder logs them so
    // the driver wave that sets them has something a host test can see.
    const uint32_t *activation_row_index;
    const void *activation_source;
    void *output_bf16;
    void *output_f32;
    void *accumulate_bf16;
    uint32_t output_row_stride;
    uint32_t output_column_offset;
    uint32_t group_count;
    uint32_t input_dimension;
    uint32_t output_dimension;
};
