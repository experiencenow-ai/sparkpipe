
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <dirent.h>

#include "sparkpipe/spark_status.h"
#include "sparkpipe/spark_weightd.h"

SparkStatus SparkWeightdMeshInit(uint32_t rank, const char *interface_name,
    uint32_t sgid_index, const char *mesh_dir);
uint32_t SparkWeightdMeshReady(void);
void SparkWeightdMeshDoorbellLoop(void);

typedef struct SparkWeightdMeshLaunch
{
    uint32_t rank;
    const char *interface_name;
    uint32_t sgid_index;
    const char *mesh_dir;
} SparkWeightdMeshLaunch;

static SparkWeightdMeshLaunch weightd_mesh_launch;

static void *SparkWeightdMeshThread(void *argument)
{
    SparkWeightdMeshLaunch *launch = (SparkWeightdMeshLaunch *)argument;
    SparkStatus status;
    status = SparkWeightdMeshInit(launch->rank,launch->interface_name,
        launch->sgid_index,launch->mesh_dir);
    if (status == SPARK_STATUS_BUSY)
        SparkWeightdMeshDoorbellLoop();
    else if (status != SPARK_STATUS_OK)
        fprintf(stderr, "weightd-mesh init=%s (serving degraded)\n",
            SparkStatusToString(status));
    return 0;
}

static volatile sig_atomic_t SparkWeightdStop;

static void SparkWeightdSignal(int signal_number)
{
    (void)signal_number;
    SparkWeightdStop = 1;
}

static void SparkWeightdUsage(const char *program)
{
    fprintf(stderr,
        "usage: %s --socket <path> [--device-bytes-max <bytes>]\n"
        "  --socket <path>          unix listen path "
            "(env SPARK_WEIGHTD_SOCKET, default /tmp/spark_weightd.sock)\n"
        "  --device-bytes-max <n>   arena ceiling in bytes "
            "(env SPARK_WEIGHTD_DEVICE_BYTES_MAX, default %llu — the "
            "operator 110 GiB device law; lower it when the node is "
            "shared, never raise it)\n"
        "  --mesh-rank <n>          mesh rank 0..%u; with "
            "--mesh-interface and --mesh-sgid-index, state all three "
            "or none\n"
        "  --mesh-interface <name>  verbs device name to bind\n"
        "  --mesh-sgid-index <n>    source GID index 0..255\n"
        "  --mesh-dir <path>        record exchange dir (env SPARK_WEIGHTD_MESH_DIR, default /tmp/weightd-mesh; use a per-deployment dir when two weightd-line daemons share the host)\n",
        program,
        (unsigned long long)SPARK_WEIGHTD_DEVICE_BYTES_MAX_DEFAULT,
        (unsigned)SPARK_WEIGHTD_MESH_RANKS - 1u);
}

#define SPARK_WEIGHTD_LATCH_PORT_DEFAULT 61900u

static int spark_weightd_latch_fd = -1;

static int SparkWeightdLatchBind(uint16_t port)
{
    struct sockaddr_in address;
    int fd = socket(AF_INET,SOCK_STREAM,0);
    int on = 1;
    if ( fd < 0 )
        return(-1);
    (void)setsockopt(fd,SOL_SOCKET,SO_REUSEADDR,&on,sizeof(on));
    memset(&address,0,sizeof(address));
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = htons(port);
    if ( bind(fd,(const struct sockaddr *)&address,sizeof(address)) != 0 ||
         listen(fd,1) != 0 )
    {
        (void)close(fd);
        return(-1);
    }
    return(fd);
}

static int SparkWeightdLatchHolderAlive(uint16_t port)
{
    struct sockaddr_in address;
    int fd = socket(AF_INET,SOCK_STREAM,0);
    int alive;
    if ( fd < 0 )
        return(0);
    memset(&address,0,sizeof(address));
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = htons(port);
    alive = connect(fd,(const struct sockaddr *)&address,sizeof(address)) == 0 ? 1 : 0;
    (void)close(fd);
    return(alive);
}

static void SparkWeightdLatchKillHolder(uint16_t port)
{
    char line[512];
    char path[256];
    FILE *table = fopen("/proc/net/tcp","r");
    DIR *procs;
    struct dirent *entry;
    if ( table == 0 )
        return;
    while ( fgets(line,sizeof(line),table) != 0 )
    {
        unsigned long inode = 0ul;
        char state[4] = {0};
        char local[32] = {0};
        if ( sscanf(line,"%*s %31s %*s %3s %*s %*s %*s %*s %*s %lu",
                local,state,&inode) < 3 || inode == 0ul )
            continue;
        {
            const char *port_text = strrchr(local,':');
            unsigned long listen_port = port_text != 0 ?
                strtoul(port_text + 1u,0,16) : 0ul;
            if ( listen_port != (unsigned long)port || strcmp(state,"0A") != 0 )
                continue;
        }
        procs = opendir("/proc");
        if ( procs == 0 )
            break;
        while ( (entry = readdir(procs)) != 0 )
        {
            char link_target[192];
            int probe_fd;
            pid_t victim;
            if ( entry->d_name[0] < '0' || entry->d_name[0] > '9' )
                continue;
            snprintf(path,sizeof(path),"/proc/%.16s/fd/%lu",entry->d_name,inode);
            probe_fd = open(path,O_RDONLY);
            if ( probe_fd < 0 )
                continue;
            if ( readlink(path,link_target,sizeof(link_target) - 1u) > 0 &&
                 strncmp(link_target,"socket:[",8u) == 0 )
            {
                char exe[128];
                char exe_path[160];
                victim = (pid_t)strtol(entry->d_name,0,10);
                snprintf(exe_path,sizeof(exe_path),"/proc/%d/exe",victim);
                if ( readlink(exe_path,exe,sizeof(exe) - 1u) > 0 &&
                     strstr(exe,"sparkpipe_weightd") != 0 )
                {
                    fprintf(stderr,
                        "weightd latch: port %u held by wedged pid %d (%s); killing\n",
                        (unsigned)port,(int)victim,exe);
                    (void)kill(victim,SIGKILL);
                }
            }
            (void)close(probe_fd);
        }
        closedir(procs);
    }
    fclose(table);
}

static int SparkWeightdLatchAcquire(uint16_t port)
{
    uint32_t attempt;
    if ( port == 0u )
        return(1);
    for ( attempt = 0u; attempt < 8u; attempt++ )
    {
        spark_weightd_latch_fd = SparkWeightdLatchBind(port);
        if ( spark_weightd_latch_fd >= 0 )
        {
            fprintf(stderr,"weightd latch: acquired port %u\n",(unsigned)port);
            return(1);
        }
        if ( SparkWeightdLatchHolderAlive(port) )
        {
            fprintf(stderr,
                "weightd latch: port %u held by a live weightd; exiting 0 (idempotent)\n",
                (unsigned)port);
            return(0);
        }
        fprintf(stderr,
            "weightd latch: port %u busy but unresponsive; hunting the holder\n",
            (unsigned)port);
        SparkWeightdLatchKillHolder(port);
        sleep(1);
    }
    fprintf(stderr,
        "weightd latch: cannot acquire port %u and no holder to correct; failing loudly\n",
        (unsigned)port);
    return(-1);
}

int main(int argument_count, char **arguments)
{
    const char *socket_path = "/tmp/spark_weightd.sock";
    uint64_t device_bytes_max = SPARK_WEIGHTD_DEVICE_BYTES_MAX_DEFAULT;
    uint64_t kv_reserve_bytes = 0ull;
    uint32_t mesh_fields = 0u;
    int ceiling_set_by_flag = 0;
    SparkWeightdServerConfig config;
    SparkWeightdServer *server = 0;
    uint32_t arena_count;
    uint64_t resident_bytes;
    SparkStatus status;
    int index;

    for (index = 1; index < argument_count; index++)
    {
        if (strcmp(arguments[index], "--socket") == 0 &&
            index + 1 < argument_count)
        {
            socket_path = arguments[++index];
        }
        else if (strcmp(arguments[index], "--device-bytes-max") == 0 &&
            index + 1 < argument_count)
        {
            char *parse_end = 0;
            device_bytes_max = strtoull(arguments[index + 1], &parse_end, 10);
            if (parse_end == arguments[index + 1] || *parse_end != '\0' ||
                device_bytes_max == 0ull)
            {
                fprintf(stderr, "weightd: bad --device-bytes-max '%s'\n",
                    arguments[index + 1]);
                SparkWeightdUsage(arguments[0]);
                return 2;
            }
            ceiling_set_by_flag = 1;
            index++;
        }
        else if (strcmp(arguments[index], "--kv-reserve-bytes") == 0 &&
            index + 1 < argument_count)
        {
            char *parse_end = 0;
            kv_reserve_bytes = strtoull(arguments[index + 1], &parse_end, 10);
            if (parse_end == arguments[index + 1] || *parse_end != '\0')
            {
                fprintf(stderr, "weightd: bad --kv-reserve-bytes '%s'\n",
                    arguments[index + 1]);
                SparkWeightdUsage(arguments[0]);
                return 2;
            }
            index++;
        }
        else if (strcmp(arguments[index], "--mesh-rank") == 0 &&
            index + 1 < argument_count)
        {
            char *parse_end = 0;
            unsigned long parsed = strtoul(arguments[index + 1],
                &parse_end, 10);
            if (parse_end == arguments[index + 1] || *parse_end != '\0' ||
                parsed >= SPARK_WEIGHTD_MESH_RANKS)
            {
                fprintf(stderr,
                    "weightd: bad --mesh-rank '%s' (need 0..%u)\n",
                    arguments[index + 1],
                    (unsigned)SPARK_WEIGHTD_MESH_RANKS - 1u);
                SparkWeightdUsage(arguments[0]);
                return 2;
            }
            weightd_mesh_launch.rank = (uint32_t)parsed;
            mesh_fields++;
            index++;
        }
        else if (strcmp(arguments[index], "--mesh-interface") == 0 &&
            index + 1 < argument_count)
        {
            weightd_mesh_launch.interface_name = arguments[++index];
            if (weightd_mesh_launch.interface_name[0] == '\0')
            {
                fprintf(stderr, "weightd: bad --mesh-interface ''\n");
                SparkWeightdUsage(arguments[0]);
                return 2;
            }
            mesh_fields++;
        }
        else if (strcmp(arguments[index], "--mesh-sgid-index") == 0 &&
            index + 1 < argument_count)
        {
            char *parse_end = 0;
            unsigned long parsed = strtoul(arguments[index + 1],
                &parse_end, 10);
            if (parse_end == arguments[index + 1] || *parse_end != '\0' ||
                parsed > 255ul)
            {
                fprintf(stderr,
                    "weightd: bad --mesh-sgid-index '%s' (need 0..255)\n",
                    arguments[index + 1]);
                SparkWeightdUsage(arguments[0]);
                return 2;
            }
            weightd_mesh_launch.sgid_index = (uint32_t)parsed;
            mesh_fields++;
            index++;
        }
        else if (strcmp(arguments[index], "--mesh-dir") == 0 &&
            index + 1 < argument_count)
        {
            weightd_mesh_launch.mesh_dir = arguments[++index];
            if (weightd_mesh_launch.mesh_dir[0] == '\0')
            {
                fprintf(stderr, "weightd: bad --mesh-dir ''\n");
                SparkWeightdUsage(arguments[0]);
                return 2;
            }
        }
        else if (strcmp(arguments[index], "--help") == 0)
        {
            SparkWeightdUsage(arguments[0]);
            return 0;
        }
        else
        {
            fprintf(stderr, "weightd: unknown argument '%s'\n", arguments[index]);
            SparkWeightdUsage(arguments[0]);
            return 2;
        }
    }
    if (mesh_fields != 0u && mesh_fields != 3u)
    {
        fprintf(stderr,
            "weightd: mesh identity partially stated (%u of 3: "
            "--mesh-rank --mesh-interface --mesh-sgid-index); "
            "state all three or none\n", mesh_fields);
        return 2;
    }
    {
        const char *env_socket = getenv("SPARK_WEIGHTD_SOCKET");
        const char *env_ceiling = getenv("SPARK_WEIGHTD_DEVICE_BYTES_MAX");
        const char *env_reserve = getenv("SPARK_WEIGHTD_KV_RESERVE_BYTES");
        const char *env_mesh_dir = getenv("SPARK_WEIGHTD_MESH_DIR");
        if (weightd_mesh_launch.mesh_dir == 0 && env_mesh_dir != 0 &&
            env_mesh_dir[0] != '\0')
            weightd_mesh_launch.mesh_dir = env_mesh_dir;
        if (env_socket != 0 && env_socket[0] != '\0')
        {
            socket_path = env_socket;
        }
        if (env_ceiling != 0 && env_ceiling[0] != '\0')
        {
            char *parse_end = 0;
            uint64_t parsed = strtoull(env_ceiling, &parse_end, 10);
            if (parse_end == env_ceiling || *parse_end != '\0' ||
                parsed == 0ull)
            {
                fprintf(stderr, "weightd: bad SPARK_WEIGHTD_DEVICE_BYTES_MAX '%s'\n",
                    env_ceiling);
                return 2;
            }
            if (!ceiling_set_by_flag)
            {
                device_bytes_max = parsed;
            }
        }
        if (env_reserve != 0 && env_reserve[0] != '\0')
        {
            char *parse_end = 0;
            uint64_t parsed = strtoull(env_reserve, &parse_end, 10);
            if (parse_end == env_reserve || *parse_end != '\0')
            {
                fprintf(stderr, "weightd: bad SPARK_WEIGHTD_KV_RESERVE_BYTES '%s'\n",
                    env_reserve);
                return 2;
            }
            kv_reserve_bytes = parsed;
        }
    }

    if (kv_reserve_bytes >= device_bytes_max)
    {
        fprintf(stderr,
            "weightd: kv reserve %llu leaves no arena room under ceiling %llu\n",
            (unsigned long long)kv_reserve_bytes,
            (unsigned long long)device_bytes_max);
        return 2;
    }

    memset(&config, 0, sizeof(config));
    config.socket_path = socket_path;
    config.device_bytes_max = device_bytes_max;
    config.kv_reserve_bytes = kv_reserve_bytes;

    signal(SIGINT, SparkWeightdSignal);
    signal(SIGTERM, SparkWeightdSignal);

    {
        uint16_t latch_port = SPARK_WEIGHTD_LATCH_PORT_DEFAULT;
        const char *latch_env = getenv("SPARK_WEIGHTD_LATCH_PORT");
        int latch;
        if ( latch_env != 0 && latch_env[0] != '\0' )
            latch_port = (uint16_t)strtoul(latch_env,0,10);
        latch = SparkWeightdLatchAcquire(latch_port);
        if ( latch == 0 )
            return 0;
        if ( latch < 0 )
            return 1;
    }

    status = SparkWeightdServerCreate(&config, &server);
    if (status != SPARK_STATUS_OK)
    {
        fprintf(stderr, "weightd create=%s socket=%s\n",
            SparkStatusToString(status), socket_path);
        return 1;
    }
    printf("spark_weightd ready unix=%s ceiling=%llu\n",
        socket_path, (unsigned long long)device_bytes_max);
    fflush(stdout);

    if (mesh_fields == 3u)
    {
        static pthread_t mesh_thread;
        if (pthread_create(&mesh_thread,0,SparkWeightdMeshThread,
            &weightd_mesh_launch) != 0)
            fprintf(stderr, "weightd-mesh: thread create failed\n");
    }
    else
    {
        fprintf(stderr,
            "weightd-mesh: identity not stated; mesh disabled\n");
    }

    status = SparkWeightdServerRun(server, &SparkWeightdStop);

    arena_count = SparkWeightdServerArenaCount(server);
    resident_bytes = SparkWeightdServerResidentBytes(server);
    SparkWeightdServerDestroy(server);
    if (status != SPARK_STATUS_OK)
    {
        fprintf(stderr, "weightd run=%s\n", SparkStatusToString(status));
        return 1;
    }
    fprintf(stderr, "spark_weightd stopped arenas=%u bytes=%llu\n",
        arena_count, (unsigned long long)resident_bytes);
    return 0;
}
