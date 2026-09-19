#pragma once

#define SPARK_LLM_FAMILY_TAG                    minimax
#define SPARK_LLM_HIDDEN_DIMENSION              5120u
#define SPARK_LLM_LAYER_COUNT                   64u
#define SPARK_LLM_OUTPUT_VOCAB_COUNT            151936u
#define SPARK_LLM_MAXIMUM_CONTEXT_TOKENS        262144u
#define SPARK_LLM_RMS_NORM_EPSILON              1e-06f
#define SPARK_LLM_END_OF_TEXT_TOKEN_ID          151645u
#define SPARK_LLM_BEGIN_OF_TEXT_TOKEN_ID        151643u

#define SPARK_LLM_ATTENTION_HEAD_COUNT          64u
#define SPARK_LLM_KV_HEAD_COUNT                 8u
#define SPARK_LLM_HEAD_DIMENSION                128u
#define SPARK_LLM_INTERMEDIATE_DIMENSION        25600u
#define SPARK_LLM_ROPE_THETA                    5000000.0f
#define SPARK_LLM_MROPE_SECTION_TEMPORAL        24u
#define SPARK_LLM_MROPE_SECTION_HEIGHT          20u
#define SPARK_LLM_MROPE_SECTION_WIDTH           20u
#define SPARK_LLM_MROPE_INTERLEAVED             1u
#define SPARK_LLM_TIED_WORD_EMBEDDINGS          0u

#define SPARK_LLM_MODEL_SOURCE_URI              "MiniMaxAI/MiniMax-H3"
