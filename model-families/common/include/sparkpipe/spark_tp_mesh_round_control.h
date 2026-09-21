#pragma once
#include <stddef.h>
#include <stdint.h>

#define SPARK_TP_MESH_ROUND_CONTROL_WORDS 10u
#define SPARK_TP_MESH_ROUND_CONTROL_BYTES \
    (SPARK_TP_MESH_ROUND_CONTROL_WORDS * sizeof(uint64_t))

#define SPARK_TP_MESH_ROUND_CONTROL_WORD_SLOT_CURSOR 0u
#define SPARK_TP_MESH_ROUND_CONTROL_WORD_ROUNDS_TOTAL 1u
#define SPARK_TP_MESH_ROUND_CONTROL_WORD_ROUNDS_DONE 2u
#define SPARK_TP_MESH_ROUND_CONTROL_WORD_ROUND_SEQ 3u
#define SPARK_TP_MESH_ROUND_CONTROL_WORD_CANCEL_EXPECTED 4u
#define SPARK_TP_MESH_ROUND_CONTROL_WORD_DEADLINE_NS 5u
#define SPARK_TP_MESH_ROUND_CONTROL_WORD_SEQ 6u
#define SPARK_TP_MESH_ROUND_CONTROL_WORD_EPOCH 7u
#define SPARK_TP_MESH_ROUND_CONTROL_WORD_ERROR 8u
#define SPARK_TP_MESH_ROUND_CONTROL_WORD_DIAG 9u

typedef struct SparkTpMeshRoundControl
{
    uint64_t slot_cursor;
    uint64_t rounds_total;
    uint64_t rounds_done;
    uint64_t round_seq;
    uint64_t cancel_expected;
    uint64_t deadline_ns;
    uint64_t seq;
    uint64_t epoch;
    uint64_t error_word;
    uint64_t diag_word;
} SparkTpMeshRoundControl;

#define SPARK_TP_MESH_ROUND_LOOP_DECISION_GO 0u
#define SPARK_TP_MESH_ROUND_LOOP_DECISION_CANCEL 1u
#define SPARK_TP_MESH_ROUND_LOOP_DECISION_TIMEOUT 2u
