#pragma once

#define SPARK_LLM_FAMILY_TAG                    qwen38_27b
#define SPARK_LLM_HIDDEN_DIMENSION              5120u
#define SPARK_LLM_LAYER_COUNT                   64u
#define SPARK_LLM_OUTPUT_VOCAB_COUNT            248320u
#define SPARK_LLM_MAXIMUM_CONTEXT_TOKENS        262144u
#define SPARK_LLM_RMS_NORM_EPSILON              1e-06f
#define SPARK_LLM_END_OF_TEXT_TOKEN_ID          248044u

#define SPARK_LLM_QUERY_HEAD_COUNT              24u
#define SPARK_LLM_KV_HEAD_COUNT                 4u
#define SPARK_LLM_HEAD_DIMENSION                256u
#define SPARK_LLM_ROPE_DIMENSION                64u
#define SPARK_LLM_ROPE_THETA                    10000000.0f

#define SPARK_LLM_ATTENTION_PERIOD              4u
#define SPARK_LLM_FULL_ATTENTION_PHASE          3u

#define SPARK_LLM_GDN_KEY_HEAD_COUNT            16u
#define SPARK_LLM_GDN_VALUE_HEAD_COUNT          48u
#define SPARK_LLM_GDN_HEAD_KEY_DIMENSION        128u
#define SPARK_LLM_GDN_HEAD_VALUE_DIMENSION      128u
#define SPARK_LLM_GDN_CONV_KERNEL               4u

#define SPARK_LLM_DENSE_INTERMEDIATE_DIMENSION  17408u
#define SPARK_LLM_MTP_LAYER_COUNT               1u
