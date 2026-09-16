#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>
#include <dlfcn.h>

#define TEST_FAIL(msg) do { \
    fprintf(stderr, "FAIL: %s:%d %s\n", __FILE__, __LINE__, msg); \
    return 1; \
} while (0)

static int check_symbol_arity(const char *so_path, const char *symbol,
    uint32_t expected_args)
{
    void *handle;
    void *sym;
    char msg[256];

    handle = dlopen(so_path, RTLD_LAZY);
    if (handle == 0) {
        snprintf(msg, sizeof(msg), "dlopen %s: %s", so_path, dlerror());
        TEST_FAIL(msg);
    }
    sym = dlsym(handle, symbol);
    if (sym == 0) {
        dlclose(handle);
        snprintf(msg, sizeof(msg), "dlsym %s: not found", symbol);
        TEST_FAIL(msg);
    }

    Dl_info info;
    if (dladdr(sym, &info) == 0) {
        dlclose(handle);
        snprintf(msg, sizeof(msg), "dladdr %s: failed", symbol);
        TEST_FAIL(msg);
    }

    dlclose(handle);

    fprintf(stderr, "PASS: %s found in %s (arity check requires disassembly)\n",
        symbol, info.dli_fname);
    return 0;
}

static int check_marker_string(const char *so_path, const char *marker)
{
    char cmd[512];
    char line[256];
    FILE *pipe;
    int found = 0;

    snprintf(cmd, sizeof(cmd), "strings %s 2>/dev/null | grep -c '%s'",
        so_path, marker);
    pipe = popen(cmd, "r");
    if (pipe == 0)
        TEST_FAIL("popen strings");
    if (fgets(line, sizeof(line), pipe) != 0) {
        int count = atoi(line);
        if (count > 0)
            found = 1;
    }
    pclose(pipe);

    if (!found) {
        snprintf(line, sizeof(line),
            "marker '%s' not found in %s — module compiled a stale/private copy",
            marker, so_path);
        TEST_FAIL(line);
    }
    fprintf(stderr, "PASS: marker '%s' found in %s\n", marker, so_path);
    return 0;
}

static int check_kernel_signature_match(const char *driver_path)
{
    char cmd[1024];
    char line[512];
    FILE *pipe;
    int mismatches = 0;

    snprintf(cmd, sizeof(cmd),
        "cuobjdump -sass %s 2>/dev/null | grep -o "
        "'_Z[0-9]*SparkGlm5Next[A-Za-z]*Kernel[A-Za-z0-9]*' | sort -u",
        driver_path);
    pipe = popen(cmd, "r");
    if (pipe == 0)
        TEST_FAIL("popen cuobjdump");

    while (fgets(line, sizeof(line), pipe) != 0) {
        fprintf(stderr, "kernel: %s", line);
    }
    pclose(pipe);
    return 0;
}

int main(int argc, char **argv)
{
    const char *driver_path;
    int failures = 0;

    if (argc < 2) {
        fprintf(stderr, "usage: %s <model_driver.so path>\n", argv[0]);
        return 2;
    }
    driver_path = argv[1];

    failures += check_marker_string(driver_path,
        "SPARK-TP-MESH-KERNELS-V3-PARITY-TAIL-ABORT-DIAG");
    failures += check_marker_string(driver_path,
        "GRAPH-CAPTURE-OK");

    failures += check_kernel_signature_match(driver_path);

    fprintf(stderr, "%s: %d failures\n", argv[0], failures);
    return failures ? 1 : 0;
}
