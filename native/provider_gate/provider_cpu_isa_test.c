#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int mxfp4_cpu_compatible(void);
int mxfp4_have_avx2(void);
void mxfp4_gemv(const uint8_t *packed, const uint8_t *scales,
                 const float *input, float *output, int rows, int columns);

static int expected_compatibility(void) {
#if (defined(__x86_64__) || defined(_M_X64)) \
        && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    return !!__builtin_cpu_supports("ssse3")
        && !!__builtin_cpu_supports("avx")
        && !!__builtin_cpu_supports("fma");
#else
    return 1;
#endif
}

static int expected_avx2(void) {
#if (defined(__x86_64__) || defined(_M_X64)) \
        && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    return expected_compatibility() && !!__builtin_cpu_supports("avx2");
#else
    return 0;
#endif
}

int main(void) {
    const int expected = expected_compatibility();
    const int compatible = mxfp4_cpu_compatible();
    if (compatible != expected) {
        fprintf(stderr,
                "provider_cpu_isa=FAIL: reported=%d expected=%d\n",
                compatible, expected);
        return 1;
    }
    if (!compatible) {
        printf("provider_cpu_isa=PASS compatible=0 avx2=0 kernel=SKIP\n");
        return mxfp4_have_avx2() == 0 ? 0 : 1;
    }

    const char *disable = getenv("K3_MXFP4_DISABLE_AVX2");
    const int disabled = disable != NULL && disable[0] != '\0'
        && strcmp(disable, "0") != 0;
    const int reported_avx2 = mxfp4_have_avx2();
    const int wanted_avx2 = disabled ? 0 : expected_avx2();
    if (reported_avx2 != wanted_avx2) {
        fprintf(stderr,
                "provider_cpu_isa=FAIL: avx2 reported=%d expected=%d\n",
                reported_avx2, wanted_avx2);
        return 1;
    }

    // One exactly representable 2x32 GEMV validates that the admitted path is
    // executable. Every nibble encodes +1, every scale is 2^0, and every input
    // lane is +1, so both output rows must be exactly 32.0f.
    uint8_t packed[2 * 16];
    uint8_t scales[2];
    float input[32];
    float output[2] = {0.0f, 0.0f};
    for (size_t index = 0; index < sizeof(packed); ++index) packed[index] = 0x22;
    for (size_t index = 0; index < sizeof(scales); ++index) scales[index] = 127;
    for (size_t index = 0; index < 32; ++index) input[index] = 1.0f;
    mxfp4_gemv(packed, scales, input, output, 2, 32);
    if (output[0] != 32.0f || output[1] != 32.0f) {
        fprintf(stderr,
                "provider_cpu_isa=FAIL: GEMV output=(%.9g, %.9g)\n",
                output[0], output[1]);
        return 1;
    }
    printf("provider_cpu_isa=PASS compatible=1 avx2=%d kernel=PASS\n",
           reported_avx2);
    return 0;
}
