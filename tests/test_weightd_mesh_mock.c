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

#include "../node/weightd_mesh.c"

#ifndef SPARK_WEIGHTD_MESH_DIR
#define SPARK_WEIGHTD_MESH_DIR "/tmp/weightd-mesh"
#endif

#define TEST_MESH_PEERS (SPARK_WEIGHTD_MESH_RANKS_PER_BAND - 1u)
#define TEST_MESH_MAGIC UINT64_C(0x4d45534830303033)
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
    uint32_t rank_mask;
    uint64_t boot_ns;
} TestMeshRecord;

static uint32_t test_rank_mask = 0xffffu;

#define TEST_MESH_INTERFACE "rocep1s0f1"

SparkStatus SparkWeightdMeshInit(uint32_t rank, const char *interface_name,
    uint32_t sgid_index, const char *mesh_dir, uint32_t rank_mask);
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
    record->rank_mask = test_rank_mask;
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

static uint64_t test_shipped(uint32_t band, uint32_t rank)
{
    return *(volatile uint64_t *)((uint8_t *)weightd_mesh.recv_buffer +
        SPARK_WEIGHTD_MESH_SHIPPED_ENTRY(band,rank));
}

static SparkStatus test_post_slot(uint32_t band, uint32_t rank,
    uint64_t seq, uint32_t mask)
{
    SparkStatus status;
    pthread_mutex_lock(&SparkWeightdMeshWireLock);
    status = SparkWeightdMeshPostSlot(band,rank,seq,
        (uint64_t)rank * SPARK_WEIGHTD_MESH_SLOTS_PER_RANK +
            ((seq - 1u) & (SPARK_WEIGHTD_MESH_SLOTS_PER_RANK - 1u)),64u,mask);
    pthread_mutex_unlock(&SparkWeightdMeshWireLock);
    return status;
}

static void test_complete_range(uint32_t first, uint32_t last)
{
    uint32_t i;
    for ( i = first; i < last; i++ )
    {
        SparkStubIbvPostedWork work;
        CHECK(spark_stub_ibv_posted(i,&work) == 0,"posted WR has a reproducible completion identity");
        CHECK(spark_stub_ibv_complete(work.wr_id,IBV_WC_SUCCESS) == 0,"completion queue has declared capacity");
    }
    SparkWeightdMeshDrainCq();
}

static void test_slot_lifetimes(uint32_t local_rank)
{
    uint32_t all_peers = ((1u << SPARK_WEIGHTD_MESH_RANKS_PER_BAND) - 1u) &
        ~(1u << local_rank);
    uint32_t first, last, i;
    uint64_t seq = (UINT64_C(7) << 32u) | 1u;
    uint64_t slot = (uint64_t)local_rank * SPARK_WEIGHTD_MESH_SLOTS_PER_RANK;
    volatile uint64_t *entry = (volatile uint64_t *)((uint8_t *)weightd_mesh.recv_buffer +
        SPARK_WEIGHTD_MESH_DOORBELL_ENTRY(0u,local_rank));
    uint8_t *payload = (uint8_t *)weightd_mesh.recv_buffer + slot * SPARK_WEIGHTD_MESH_SLOT_BYTES;
    SparkStubIbvPostedWork stale;
    test_complete_range(0u,spark_stub_ibv_posted_count());
    first = spark_stub_ibv_posted_count();
    memset(payload,0x5a,64u);
    entry[2] = slot;
    entry[1] = 64u;
    entry[3] = all_peers;
    __sync_synchronize();
    *(volatile uint64_t *)(payload + SPARK_WEIGHTD_MESH_SLOT_BYTES - 8u) = seq;
    __sync_synchronize();
    entry[0] = seq;
    SparkWeightdMeshDoorbellPoll();
    last = spark_stub_ibv_posted_count();
    CHECK(last - first == TEST_MESH_PEERS * 2u,"B1 posts unchanged payload and tail to every peer");
    CHECK(test_shipped(0u,local_rank) == 0u,"posting transfers does not release the source slot");
    for ( i = first; i < last; i++ )
    {
        SparkStubIbvPostedWork work;
        CHECK(spark_stub_ibv_posted(i,&work) == 0,"B1 WR captured");
        CHECK((work.flags & IBV_SEND_SIGNALED) != 0u,"every owned WR has terminal evidence");
        if ( (work.wr_id & 3u) == 3u )
            CHECK(spark_stub_ibv_complete(work.wr_id,IBV_WC_SUCCESS) == 0,"tail completion delivered before payload");
        else
        {
            CHECK(work.source == (uint64_t)(uintptr_t)payload && work.length == 64u,
                "B1 source and payload extent remain unchanged");
            CHECK(*(const uint8_t *)(uintptr_t)work.source == 0x5au,"NIC source retains original contribution until completion");
        }
    }
    SparkWeightdMeshDrainCq();
    CHECK(test_shipped(0u,local_rank) == 0u,"tails alone cannot release outstanding payload reads");
    CHECK(spark_stub_ibv_posted(first + 1u,&stale) == 0,"capture duplicate tail identity");
    CHECK(spark_stub_ibv_complete(stale.wr_id,IBV_WC_SUCCESS) == 0,"duplicate tail injected");
    SparkWeightdMeshDoorbellPoll();
    CHECK(test_shipped(0u,local_rank) == 0u,"duplicate completion does not consume another WR");
    CHECK(spark_stub_ibv_posted_count() == last,"pending doorbell is not posted a second time");
    test_complete_range(first,last - 2u);
    CHECK(test_shipped(0u,local_rank) == 0u,"last outstanding peer retains ownership");
    test_complete_range(last - 2u,last);
    CHECK(test_shipped(0u,local_rank) == seq,"all terminal WRs release the source generation");
    first = spark_stub_ibv_posted_count();
    CHECK(test_post_slot(0u,local_rank,seq + 1u,all_peers) == SPARK_STATUS_OK,
        "completed source slot admits its next generation");
    last = spark_stub_ibv_posted_count();
    CHECK(spark_stub_ibv_complete(stale.wr_id,IBV_WC_SUCCESS) == 0,"prior generation completion injected");
    SparkWeightdMeshDrainCq();
    CHECK(test_shipped(0u,local_rank) == seq,"stale completion never releases the new generation");
    test_complete_range(first,last);
    CHECK(test_shipped(0u,local_rank) == seq + 1u,"new generation releases after its own completions");
    first = spark_stub_ibv_posted_count();
    CHECK(test_post_slot(0u,local_rank,seq + (UINT64_C(1) << 32u),all_peers) == SPARK_STATUS_OK,
        "new epoch permits reused low sequence bits");
    last = spark_stub_ibv_posted_count();
    CHECK(test_shipped(0u,local_rank) == seq + 1u,
        "prior epoch acknowledgement cannot release a new epoch");
    test_complete_range(first,last);
    CHECK(test_shipped(0u,local_rank) == seq + (UINT64_C(1) << 32u),
        "completion acknowledges the full epoch and sequence tag");
    entry[0] = 0u;

    first = spark_stub_ibv_posted_count();
    spark_stub_ibv_fail_post_call(spark_stub_ibv_post_send_calls() + 2u);
    CHECK(test_post_slot(1u,local_rank,seq,all_peers) == SPARK_STATUS_IO_ERROR,
        "partial post failure reports explicit failure");
    last = spark_stub_ibv_posted_count();
    spark_stub_ibv_fail_post_call(0u);
    CHECK(last > first && test_shipped(1u,local_rank) == 0u,
        "partial failure retains successfully posted work without false ACK");
    CHECK(test_post_slot(1u,local_rank,seq + 1u,all_peers) == SPARK_STATUS_IO_ERROR,
        "partial failure fences new source ownership while draining");
    test_complete_range(first,last);
    CHECK(test_shipped(1u,local_rank) == 0u &&
        weightd_mesh.transfers[SPARK_WEIGHTD_MESH_RANKS_PER_BAND + local_rank].pending == 0u,
        "failed generation drains all accepted work but does not advertise success");

    first = spark_stub_ibv_posted_count();
    CHECK(test_post_slot(2u,local_rank,seq,all_peers) == SPARK_STATUS_OK,"CQ failure scenario posts");
    last = spark_stub_ibv_posted_count();
    CHECK(spark_stub_ibv_posted(first,&stale) == 0,"capture failing WR identity");
    CHECK(spark_stub_ibv_complete(stale.wr_id,IBV_WC_RETRY_EXC_ERR) == 0,"terminal transport error injected");
    SparkWeightdMeshDrainCq();
    test_complete_range(first + 1u,last);
    CHECK(test_shipped(2u,local_rank) == 0u &&
        weightd_mesh.transfers[2u * SPARK_WEIGHTD_MESH_RANKS_PER_BAND + local_rank].failed != 0u,
        "CQ error cannot be converted into a successful shipment");

    first = spark_stub_ibv_posted_count();
    CHECK(test_post_slot(3u,local_rank,seq,1u << 2u) == SPARK_STATUS_OK,
        "tree destination uses the common posting path");
    last = spark_stub_ibv_posted_count();
    CHECK(last - first == 2u,"masked transfer posts only payload and tail for selected peer");
    for ( i = first; i < last; i++ )
    {
        SparkStubIbvPostedWork work;
        CHECK(spark_stub_ibv_posted(i,&work) == 0 &&
            work.qp_number == weightd_mesh.send_qps[2u]->qp_num,
            "peer mask resolves the existing physical-rank mapping");
    }
    test_complete_range(first,last);
    CHECK(test_shipped(3u,local_rank) == seq,"masked transfer requires only its selected peer completions");

    first = spark_stub_ibv_posted_count();
    CHECK(test_post_slot(4u,local_rank,seq + 4u,all_peers) == SPARK_STATUS_OK,
        "sparse phase sequence publishes current payload");
    last = spark_stub_ibv_posted_count();
    CHECK(last - first == TEST_MESH_PEERS * 2u,
        "sparse phases never replay stale source slots");
    test_complete_range(first,last - 1u);
    CHECK(test_shipped(4u,local_rank) == 0u,"sparse shipment retains source until final completion");
    test_complete_range(last - 1u,last);
    CHECK(test_shipped(4u,local_rank) == seq + 4u,"sparse shipment acknowledges its actual tag");

    first = spark_stub_ibv_posted_count();
    for ( i = 0u; i < 32u; i++ )
        CHECK(test_post_slot(8u + i / 4u,i % 4u,seq + 4u,all_peers) == SPARK_STATUS_OK,
            "declared SQ capacity admits two WRs per peer for thirty-two independent entries");
    last = spark_stub_ibv_posted_count();
    CHECK(last - first == SPARK_WEIGHTD_MESH_PEERS * SPARK_WEIGHTD_MESH_SEND_CAPACITY,
        "maximum pending completions match actual configured SQ capacity");
    CHECK(test_post_slot(7u,0u,seq + 4u,all_peers) == SPARK_STATUS_BUSY,
        "capacity pressure rejects before any partial posting");
    CHECK(spark_stub_ibv_posted_count() == last,"BUSY leaves publication and completion ledgers unchanged");
    test_complete_range(first,first + TEST_MESH_PEERS * 2u);
    CHECK(test_post_slot(7u,0u,seq + 4u,all_peers) == SPARK_STATUS_OK,
        "completed work makes capacity retry useful");
    test_complete_range(first + TEST_MESH_PEERS * 2u,spark_stub_ibv_posted_count());
    for ( i = 0u; i < TEST_MESH_PEERS; i++ )
        CHECK(weightd_mesh.send_pending[i] == 0u,"all accepted SQ ownership returns after terminal completions");
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
        fprintf(stderr,"SETUP FAIL test_weightd_mesh_mock: built without a private "
            "SPARK_WEIGHTD_MESH_DIR\n");
        return 2;
    }
    test_ready_path(path,sizeof(path));
    if (stat(path,&st) == 0)
    {
        fprintf(stderr,"SETUP FAIL test_weightd_mesh_mock: %s looks like a live mesh\n",
            SPARK_WEIGHTD_MESH_DIR);
        return 2;
    }
    local_rank = 7u; /* explicit: the rank now comes from --mesh-rank */
    test_clean_dir();
    if (mkdir(SPARK_WEIGHTD_MESH_DIR,0755) != 0 && errno != EEXIST)
    {
        fprintf(stderr,"SETUP FAIL test_weightd_mesh_mock: cannot create %s errno=%d\n",
            SPARK_WEIGHTD_MESH_DIR,errno);
        return 2;
    }

    for (rank = 0u; rank < SPARK_WEIGHTD_MESH_RANKS_PER_BAND; rank++)
    {
        if (rank == local_rank)
            continue;
        CHECK(test_write_record(rank,1u) == 0,"case1 write peer record");
    }
    status = SparkWeightdMeshInit(local_rank,TEST_MESH_INTERFACE,3u,SPARK_WEIGHTD_MESH_DIR,test_rank_mask);
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

    {
        SparkWeightdMeshRecord incompatible;
        int fd;
        test_record_path(0u,path,sizeof(path));
        fd = open(path,O_WRONLY);
        CHECK(fd >= 0,"open peer record for ABI qualification");
        if ( fd >= 0 )
        {
            uint64_t old_magic = UINT64_C(0x4d45534830303031);
            CHECK(test_write_fully(fd,&old_magic,sizeof(old_magic)) == 0,
                "write previous mesh ABI magic");
            close(fd);
        }
        CHECK(SparkWeightdMeshReadPeerRecord(0u,&incompatible) == SPARK_STATUS_ABI_MISMATCH,
            "prior mesh ABI is rejected explicitly");
        SparkWeightdMeshTryWire();
        test_ready_path(path,sizeof(path));
        CHECK(SparkWeightdMeshReady() == 0u && stat(path,&st) != 0,
            "incompatible peer clears readiness and its marker");
        CHECK(test_write_record(0u,1u) == 0,"restore compatible peer record");
        SparkWeightdMeshTryWire();
        CHECK(SparkWeightdMeshReady() == 1u,"compatible records restore readiness");
        test_record_path(0u,path,sizeof(path));
        CHECK(unlink(path) == 0,"remove peer record for missing-peer qualification");
        SparkWeightdMeshTryWire();
        test_ready_path(path,sizeof(path));
        CHECK(SparkWeightdMeshReady() == 0u && stat(path,&st) != 0,
            "missing peer cannot inherit an old ready marker");
        CHECK(test_write_record(0u,1u) == 0,"restore missing peer record");
        SparkWeightdMeshTryWire();
        CHECK(SparkWeightdMeshReady() == 1u,"restored peer permits readiness");
    }

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

    status = SparkWeightdMeshInit(local_rank,TEST_MESH_INTERFACE,3u,SPARK_WEIGHTD_MESH_DIR,test_rank_mask);
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

    test_slot_lifetimes(local_rank);

    /* case 6: a second daemon instance with its own record dir must not
     * touch ours — the two-daemons-one-host separation (the fleet's
     * weightd vs the driver developers' standalone weightsd). */
    {
        char dir2[256];
        uint64_t dir1_boot = own_record.boot_ns;
        (void)snprintf(dir2,sizeof(dir2),"%s-second",SPARK_WEIGHTD_MESH_DIR);
        (void)mkdir(dir2,0755);
        status = SparkWeightdMeshInit(local_rank,TEST_MESH_INTERFACE,3u,dir2,test_rank_mask);
        CHECK(status == SPARK_STATUS_BUSY,"case6 second init publishes");
        {
            (void)snprintf(path,sizeof(path),"%s/mesh-%x.rec",dir2,local_rank);
            CHECK(stat(path,&st) == 0,"case6 record lands in the second dir");
        }
        /* our original record is untouched */
        {
            TestMeshRecord first_again;
            CHECK(test_read_record(local_rank,&first_again) == 0,
                "case6 original record still readable");
            CHECK(first_again.boot_ns == dir1_boot,
                "case6 original record not clobbered by the second instance");
        }
        {
            char cmd[512];
            (void)snprintf(cmd,sizeof(cmd),"rm -rf %s",dir2);
            (void)system(cmd);
        }
    }

    test_clean_dir();
    test_rank_mask = 0xfu;
    CHECK(SparkWeightdMeshInit(0u,TEST_MESH_INTERFACE,3u,SPARK_WEIGHTD_MESH_DIR,0u) ==
        SPARK_STATUS_INVALID_ARGUMENT,"empty participant mask rejects");
    CHECK(SparkWeightdMeshInit(4u,TEST_MESH_INTERFACE,3u,SPARK_WEIGHTD_MESH_DIR,test_rank_mask) ==
        SPARK_STATUS_INVALID_ARGUMENT,"rank outside participant mask rejects");
    CHECK(SparkWeightdMeshInit(0u,TEST_MESH_INTERFACE,3u,SPARK_WEIGHTD_MESH_DIR,test_rank_mask) ==
        SPARK_STATUS_BUSY,"TP4 explicit group initializes");
    for ( rank = 1u; rank < 4u; rank++ )
        CHECK(test_write_record(rank,9u) == 0,"TP4 required record published");
    SparkWeightdMeshTryWire();
    CHECK(SparkWeightdMeshReady() == 1u,"TP4 needs only its three configured peers");
    post_before = spark_stub_ibv_post_send_calls();
    CHECK(SparkWeightdMeshBroadcast(1u << 4u,0u,64u,0u,0u,0u) == 0u &&
        spark_stub_ibv_post_send_calls() == post_before,
        "broadcast outside configured group rejects before posting");
    CHECK(test_post_slot(0u,0u,1u,1u << 4u) == SPARK_STATUS_INVALID_ARGUMENT,
        "doorbell cannot send to an absent participant");
    {
        volatile uint64_t *entry = (volatile uint64_t *)((uint8_t *)weightd_mesh.recv_buffer +
            SPARK_WEIGHTD_MESH_DOORBELL_ENTRY(0u,0u));
        entry[2] = 0u;
        entry[1] = 64u;
        entry[3] = 0xeu;
        entry[0] = 1u;
        SparkWeightdMeshDoorbellPoll();
    }
    CHECK(spark_stub_ibv_post_send_calls() - post_before == 6u,
        "TP4 B1 posts payload and tail to three peers only");
    test_rank_mask = 0xffu;
    CHECK(test_write_record(1u,9u) == 0,"peer with inconsistent group published");
    SparkWeightdMeshTryWire();
    CHECK(SparkWeightdMeshReady() == 0u,"inconsistent participant groups cannot become ready");

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
