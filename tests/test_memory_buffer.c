#include <assert.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include "sparkpipe/spark_memory_buffer.h"

static SparkMemoryBuffer *expected_descriptor;
static void *expected_pointer;
static SparkMemorySpace expected_space;
static uint32_t free_count;

static void TestRelease(void *pointer,SparkMemorySpace space)
{
	assert(pointer == expected_pointer);
	assert(space == expected_space);
	if ( expected_descriptor == pointer )
	{
		assert(expected_descriptor->pointer == 0);
		assert(expected_descriptor->space == 0u);
		assert(expected_descriptor->bytes == 0u);
	}
	free_count++;
}

static void TestFree(void *pointer)
{
	TestRelease(pointer,SPARK_MEMORY_SPACE_HOST_COHERENT);
	free(pointer);
}

static cudaError_t TestCudaFreeHost(void *pointer)
{
	TestRelease(pointer,SPARK_MEMORY_SPACE_HOST_PINNED);
	return(cudaFreeHost(pointer));
}

static cudaError_t TestCudaFree(void *pointer)
{
	TestRelease(pointer,SPARK_MEMORY_SPACE_DEVICE_PRIVATE);
	return(cudaFree(pointer));
}

#define free TestFree
#define cudaFreeHost TestCudaFreeHost
#define cudaFree TestCudaFree
#include "../runtime/memory_buffer.c"
#undef cudaFree
#undef cudaFreeHost
#undef free

static void TestBufferLifetime(SparkMemorySpace space,uint32_t embedded)
{
	SparkMemoryBuffer allocation;
	SparkMemoryBuffer *descriptor = &allocation;
	assert(SparkMemoryBufferAllocate(&allocation,space,UINT64_C(1048576)) == SPARK_STATUS_OK);
	if ( embedded != 0u )
	{
		descriptor = allocation.pointer;
		*descriptor = allocation;
	}
	expected_descriptor = descriptor;
	expected_pointer = allocation.pointer;
	expected_space = space;
	uint32_t previous_count = free_count;
	SparkMemoryBufferFree(descriptor);
	assert(free_count == previous_count + 1u);
	if ( embedded == 0u )
	{
		assert(descriptor->pointer == 0);
		assert(descriptor->space == 0u);
		assert(descriptor->bytes == 0u);
		SparkMemoryBufferFree(descriptor);
		assert(free_count == previous_count + 1u);
	}
}

int main(void)
{
	SparkMemoryBufferFree(0);
	TestBufferLifetime(SPARK_MEMORY_SPACE_HOST_COHERENT,0u);
	TestBufferLifetime(SPARK_MEMORY_SPACE_HOST_COHERENT,1u);
	TestBufferLifetime(SPARK_MEMORY_SPACE_HOST_PINNED,0u);
	TestBufferLifetime(SPARK_MEMORY_SPACE_HOST_PINNED,1u);
	TestBufferLifetime(SPARK_MEMORY_SPACE_DEVICE_PRIVATE,0u);
	assert(free_count == 5u);
	return(0);
}
