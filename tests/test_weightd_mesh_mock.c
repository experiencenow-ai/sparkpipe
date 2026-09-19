#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/stat.h>

#include <infiniband/verbs.h>

#include "sparkpipe/spark_status.h"
#include "sparkpipe/spark_weightd.h"

#ifndef SPARK_WEIGHTD_MESH_DIR
#define SPARK_WEIGHTD_MESH_DIR "/tmp/weightd-mesh"
#endif

#define TEST_MESH_PEERS (SPARK_WEIGHTD_MESH_RANKS_PER_BAND - 1u)
#define TEST_MESH_MAGIC UINT64_C(0x4d45534830303031)
#define TEST_MESH_LIVE_DIR "/tmp/weightd-mesh"

typedef struct TestMeshRecord
{
    uint64_t magic;
    uint32_t rank;
    uint32_t send_qpn[TEST_MESH_PEERS];
    uint32_t recv_qpn[TEST_MESH_PEERS];
    uint32_t rkey;
    uint64_t recv_addr;
    uint16_t lid;
    uint8_t gid[16];
    uint8_t reserved[4];
    uint64_t boot_ns;
} TestMeshRecord;

#define TEST_MESH_INTERFACE "rocep1s0f1"

SparkStatus SparkWeightdMeshInit(uint32_t rank, const char *interface_name,
    uint32_t sgid_index);
uint32_t SparkWeightdMeshReady(void);
void SparkWeightdMeshPoll(void);
uint32_t SparkWeightdMeshBroadcast(uint32_t peer_rank_mask,
    uint64_t source_offset, uint32_t length, uint64_t remote_offset,
    uint64_t seq_value, uint64_t seq_remote_offset);

#if defined(__APPLE__)
int memfd_create(const char *name, unsigned int flags)
{
    char path[128];
    int fd;
    (void)flags;
    (void)snprintf(path,sizeof(path),"/tmp/spark-mesh-mock-%s-%d-XXXXXX",
        name,(int)getpid());
    fd = mkstemp(path);
    if (fd >= 0)
        (void)unlink(path);
    return fd;
}
#endif

static uint32_t test_checks;
static uint32_t test_failures;

#define CHECK(cond, name) do { \
        test_checks++; \
        if ( !(cond) ) { \
            test_failures++; \
            fprintf(stderr,"FAIL %s:%d %s\n",__FILE__,__LINE__,name); \
        } \
    } while (0)

static uint32_t test_peer_rank(uint32_t peer, uint32_t local_rank)
{
    return peer < local_rank ? peer : peer + 1u;
}

static uint32_t test_my_index(uint32_t peer_rank, uint32_t local_rank)
{
    return local_rank < peer_rank ? local_rank : local_rank - 1u;
}

static void test_record_path(uint32_t rank, char *path, size_t path_bytes)
{
    (void)snprintf(path,path_bytes,"%s/mesh-%x.rec",
        SPARK_WEIGHTD_MESH_DIR,rank);
}

static void test_fill_record(TestMeshRecord *record, uint32_t rank,
    uint32_t generation)
{
    uint32_t index;
    memset(record,0,sizeof(*record));
    record->magic = TEST_MESH_MAGIC;
    record->rank = rank;
    for (index = 0u; index < TEST_MESH_PEERS; index++)
    {
        record->send_qpn[index] =
            0x40000u + generation * 0x20000u + rank * 0x400u + index;
        record->recv_qpn[index] = record->send_qpn[index] + 0x200u;
    }
    record->rkey = 0x9000u + generation * 0x100u + rank;
    record->recv_addr = UINT64_C(0x7f0000000000) +
        generation * UINT64_C(0x1000000) + rank * UINT64_C(0x10000);
    record->lid = (uint16_t)(0x2000u + rank);
    for (index = 0u; index < 16u; index++)
        record->gid[index] = (uint8_t)(rank * 16u + index);
    record->boot_ns = UINT64_C(0xb000000000000000) +
        generation * UINT64_C(0x100000) + rank;
}

static int test_write_fully(int fd, const void *buffer, size_t bytes)
{
    const char *cursor = (const char *)buffer;
    size_t remaining = bytes;
    while (remaining > 0u)
    {
        ssize_t written = write(fd,cursor,remaining);
        if (written <= 0)
            return -1;
        cursor += (size_t)written;
        remaining -= (size_t)written;
    }
    return 0;
}

static int test_write_record(uint32_t rank, uint32_t generation)
{
    TestMeshRecord record;
    char path[256];
    int fd;
    int result;
    test_fill_record(&record,rank,generation);
    test_record_path(rank,path,sizeof(path));
    fd = open(path,O_WRONLY | O_CREAT | O_TRUNC,0644);
    if (fd < 0)
        return -1;
    result = test_write_fully(fd,&record,sizeof(record));
    if (result == 0 && fsync(fd) != 0)
        result = -1;
    if (close(fd) != 0)
        result = -1;
    return result;
}

static int test_read_record(uint32_t rank, TestMeshRecord *record)
{
    char path[256];
    int fd;
    char *cursor;
    size_t remaining;
    test_record_path(rank,path,sizeof(path));
    fd = open(path,O_RDONLY);
    if (fd < 0)
        return -1;
    cursor = (char *)record;
    remaining = sizeof(*record);
    while (remaining > 0u)
    {
        ssize_t got = read(fd,cursor,remaining);
        if (got <= 0)
        {
            (void)close(fd);
            return -1;
        }
        cursor += (size_t)got;
        remaining -= (size_t)got;
    }
    (void)close(fd);
    return record->magic == TEST_MESH_MAGIC ? 0 : -1;
}

static void test_expect_peer_wired(const TestMeshRecord *own_record,
    uint32_t local_rank, uint32_t peer, uint32_t generation,
    const char *tag)
{
    TestMeshRecord expected;
    uint32_t rank;
    uint32_t my_index;
    char name[128];
    rank = test_peer_rank(peer,local_rank);
    my_index = test_my_index(rank,local_rank);
    test_fill_record(&expected,rank,generation);
    (void)snprintf(name,sizeof(name),"%s peer=%u send qp aimed at record",
        tag,peer);
    CHECK(spark_stub_ibv_qp_remote_qpn(own_record->send_qpn[peer]) ==
        (int)expected.recv_qpn[my_index],name);
    (void)snprintf(name,sizeof(name),"%s peer=%u recv qp aimed at record",
        tag,peer);
    CHECK(spark_stub_ibv_qp_remote_qpn(own_record->recv_qpn[peer]) ==
        (int)expected.send_qpn[my_index],name);
    (void)snprintf(name,sizeof(name),"%s peer=%u send qp rts",tag,peer);
    CHECK(spark_stub_ibv_qp_state(own_record->send_qpn[peer]) ==
        IBV_QPS_RTS,name);
    (void)snprintf(name,sizeof(name),"%s peer=%u recv qp rts",tag,peer);
    CHECK(spark_stub_ibv_qp_state(own_record->recv_qpn[peer]) ==
        IBV_QPS_RTS,name);
}

static int test_stderr_saved = -1;
static int test_stderr_capture_fd = -1;

static void test_capture_begin(const char *path)
{
    fflush(stderr);
    test_stderr_saved = dup(2);
    test_stderr_capture_fd = open(path,O_WRONLY | O_CREAT | O_TRUNC,0644);
    if (test_stderr_saved < 0 || test_stderr_capture_fd < 0)
        return;
    (void)dup2(test_stderr_capture_fd,2);
}

static void test_capture_end(void)
{
    fflush(stderr);
    if (test_stderr_saved >= 0)
    {
        (void)dup2(test_stderr_saved,2);
        (void)close(test_stderr_saved);
        test_stderr_saved = -1;
    }
    if (test_stderr_capture_fd >= 0)
    {
        (void)close(test_stderr_capture_fd);
        test_stderr_capture_fd = -1;
    }
}

static int test_file_contains(const char *path, const char *token)
{
    char buffer[65536];
    int fd;
    ssize_t got;
    int found;
    fd = open(path,O_RDONLY);
    if (fd < 0)
        return 0;
    got = read(fd,buffer,sizeof(buffer) - 1u);
    (void)close(fd);
    if (got < 0)
        return 0;
    buffer[got] = '\0';
    found = strstr(buffer,token) != 0;
    return found;
}

static void test_ready_path(char *path, size_t path_bytes)
{
    (void)snprintf(path,path_bytes,"%s/.ready",SPARK_WEIGHTD_MESH_DIR);
}

static void test_clean_dir(void)
{
    char path[256];
    uint32_t rank;
    test_ready_path(path,sizeof(path));
    (void)unlink(path);
    for (rank = 0u; rank < SPARK_WEIGHTD_MESH_RANKS_PER_BAND; rank++)
    {
        test_record_path(rank,path,sizeof(path));
        (void)unlink(path);
    }
}

static void test_sleep_ns(uint64_t ns)
{
    struct timespec pause;
    pause.tv_sec = (time_t)(ns / 1000000000ull);
    pause.tv_nsec = (long)(ns % 1000000000ull);
    (void)nanosleep(&pause,0);
}

int main(void)
{
    TestMeshRecord own_record;
    TestMeshRecord expected;
    struct stat st;
    char path[256];
    char log_path[320];
    uint32_t local_rank;
    uint32_t rank;
    uint32_t peer;
    uint32_t dead_qpn;
    uint32_t my_index;
    uint64_t modify_before;
    uint64_t failures_before;
    uint64_t post_before;
    SparkStatus status;

    if (strcmp(SPARK_WEIGHTD_MESH_DIR,TEST_MESH_LIVE_DIR) == 0)
    {
        printf("SKIP test_weightd_mesh_mock: built without a private "
            "SPARK_WEIGHTD_MESH_DIR\n");
        return 0;
    }
    test_ready_path(path,sizeof(path));
    if (stat(path,&st) == 0)
    {
        printf("SKIP test_weightd_mesh_mock: %s looks like a live mesh\n",
            SPARK_WEIGHTD_MESH_DIR);
        return 0;
    }
    local_rank = 7u; /* explicit: the rank now comes from --mesh-rank */
    test_clean_dir();
    if (mkdir(SPARK_WEIGHTD_MESH_DIR,0755) != 0 && errno != EEXIST)
    {
        printf("SKIP test_weightd_mesh_mock: cannot create %s errno=%d\n",
            SPARK_WEIGHTD_MESH_DIR,errno);
        return 0;
    }

    for (rank = 0u; rank < SPARK_WEIGHTD_MESH_RANKS_PER_BAND; rank++)
    {
        if (rank == local_rank)
            continue;
        CHECK(test_write_record(rank,1u) == 0,"case1 write peer record");
    }
    status = SparkWeightdMeshInit(local_rank,TEST_MESH_INTERFACE,3u);
    CHECK(status == SPARK_STATUS_BUSY,"case1 init publishes and defers");
    CHECK(test_read_record(local_rank,&own_record) == 0,
        "case1 own record published");
    CHECK(own_record.boot_ns != 0ull,"case1 own boot_ns nonzero");
    SparkWeightdMeshPoll();
    CHECK(SparkWeightdMeshReady() == 1u,"case1 mesh ready after cold wire");
    test_ready_path(path,sizeof(path));
    CHECK(stat(path,&st) == 0,"case1 .ready marker exists");
    for (peer = 0u; peer < TEST_MESH_PEERS; peer++)
        test_expect_peer_wired(&own_record,local_rank,peer,1u,"case1");
    CHECK(spark_stub_ibv_modify_qp_calls() ==
        (uint64_t)TEST_MESH_PEERS * 2u * 4u,
        "case1 modify_qp count = peers*2qps*4transitions");

    peer = 2u;
    rank = test_peer_rank(peer,local_rank);
    CHECK(test_write_record(rank,2u) == 0,"case2 peer record rewritten");
    modify_before = spark_stub_ibv_modify_qp_calls();
    SparkWeightdMeshPoll();
    CHECK(SparkWeightdMeshReady() == 1u,"case2 stays ready");
    CHECK(spark_stub_ibv_modify_qp_calls() - modify_before == 8ull,
        "case2 only the restarted peer is retransitioned");
    test_expect_peer_wired(&own_record,local_rank,peer,2u,"case2");
    test_expect_peer_wired(&own_record,local_rank,0u,1u,
        "case2 untouched peer keeps generation");

    modify_before = spark_stub_ibv_modify_qp_calls();
    test_sleep_ns(1100000000ull);
    SparkWeightdMeshPoll();
    CHECK(spark_stub_ibv_modify_qp_calls() == modify_before,
        "case3 unchanged records are a wiring no-op");
    CHECK(SparkWeightdMeshReady() == 1u,"case3 stays ready");

    status = SparkWeightdMeshInit(local_rank,TEST_MESH_INTERFACE,3u);
    CHECK(status == SPARK_STATUS_BUSY,"case4 init republishes");
    CHECK(test_read_record(local_rank,&own_record) == 0,
        "case4 own record republished");
    for (rank = 0u; rank < SPARK_WEIGHTD_MESH_RANKS_PER_BAND; rank++)
    {
        if (rank == local_rank)
            continue;
        CHECK(test_write_record(rank,3u) == 0,"case4 write peer record");
    }
    peer = 5u;
    rank = test_peer_rank(peer,local_rank);
    my_index = test_my_index(rank,local_rank);
    test_fill_record(&expected,rank,3u);
    dead_qpn = expected.recv_qpn[my_index];
    spark_stub_ibv_fail_modify_qp_for_qpn(dead_qpn);
    failures_before = spark_stub_ibv_modify_qp_failures();
    (void)snprintf(log_path,sizeof(log_path),"%s/capture-wire-fail.log",
        SPARK_WEIGHTD_MESH_DIR);
    test_capture_begin(log_path);
    SparkWeightdMeshPoll();
    test_capture_end();
    CHECK(SparkWeightdMeshReady() == 0u,
        "case4 dead qpn keeps mesh unready");
    test_ready_path(path,sizeof(path));
    CHECK(stat(path,&st) != 0,"case4 no false .ready marker");
    CHECK(spark_stub_ibv_modify_qp_failures() - failures_before == 1ull,
        "case4 exactly one dead-qpn modify rejected");
    CHECK(test_file_contains(log_path,"WD-WIRE-FAIL"),
        "case4 WD-WIRE-FAIL reported");
    spark_stub_ibv_fail_modify_qp_for_qpn(0u);
    modify_before = spark_stub_ibv_modify_qp_calls();
    SparkWeightdMeshPoll();
    CHECK(SparkWeightdMeshReady() == 1u,"case4 good record completes wiring");
    CHECK(stat(path,&st) == 0,"case4 .ready marker after recovery");
    CHECK(spark_stub_ibv_modify_qp_calls() - modify_before == 8ull,
        "case4 only the failed peer is retransitioned on retry");
    test_expect_peer_wired(&own_record,local_rank,peer,3u,"case4");
    test_expect_peer_wired(&own_record,local_rank,0u,3u,"case4");

    modify_before = spark_stub_ibv_modify_qp_calls();
    SparkWeightdMeshPoll();
    CHECK(spark_stub_ibv_modify_qp_calls() == modify_before,
        "case5 wired record check is a no-op");
    peer = 7u;
    rank = test_peer_rank(peer,local_rank);
    CHECK(test_write_record(rank,4u) == 0,"case5 restarted peer record");
    spark_stub_ibv_poll_cq_inject(IBV_WC_RETRY_EXC_ERR,1u);
    (void)snprintf(log_path,sizeof(log_path),"%s/capture-cqerr.log",
        SPARK_WEIGHTD_MESH_DIR);
    test_capture_begin(log_path);
    SparkWeightdMeshPoll();
    test_capture_end();
    CHECK(test_file_contains(log_path,"WD-MESH-CQERR"),
        "case5 WD-MESH-CQERR reported");
    CHECK(spark_stub_ibv_modify_qp_calls() - modify_before == 8ull,
        "case5 cqerr repair rewires the restarted peer");
    test_expect_peer_wired(&own_record,local_rank,peer,4u,"case5");
    CHECK(SparkWeightdMeshReady() == 1u,"case5 stays ready through repair");
    post_before = spark_stub_ibv_post_send_calls();
    CHECK(SparkWeightdMeshBroadcast(1u << rank,0ull,64u,0ull,0ull,0ull) == 1u,
        "case5 broadcast posts to repaired peer");
    CHECK(spark_stub_ibv_post_send_calls() - post_before == 1ull,
        "case5 post_send flows after repair");
    spark_stub_ibv_poll_cq_inject(IBV_WC_SUCCESS,1u);
    SparkWeightdMeshPoll();
    CHECK(SparkWeightdMeshReady() == 1u,"case5 clean poll after recovery");

    test_clean_dir();
    (void)snprintf(path,sizeof(path),"%s/capture-wire-fail.log",
        SPARK_WEIGHTD_MESH_DIR);
    (void)unlink(path);
    (void)snprintf(path,sizeof(path),"%s/capture-cqerr.log",
        SPARK_WEIGHTD_MESH_DIR);
    (void)unlink(path);
    (void)rmdir(SPARK_WEIGHTD_MESH_DIR);
    fprintf(stderr,"%s: %u checks, %u failures\n",
        "test_weightd_mesh_mock",test_checks,test_failures);
    return test_failures != 0u ? 1 : 0;
}
