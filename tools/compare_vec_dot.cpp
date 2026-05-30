/* Compare one Q4_K row dot product: Gemma4.c generic vs ggml CPU generic */
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {
#include "ggml-quants.h"
#include "ggml-cpu/quants.h"
}

/* Minimal copies from gemma4 cpu-blas/main.c */
#define QK_K 256

typedef struct {
    uint8_t scales[QK_K/16];
    uint8_t qs[QK_K/2];
    uint16_t d;
    uint16_t dmin;
} BlockQ4_K;

typedef struct {
    float d;
    int8_t qs[QK_K];
    int16_t bsums[QK_K/16];
} BlockQ8_K;

static inline float host_f16f32(uint16_t h) {
    uint32_t sgn = ((uint32_t)h & 0x8000u) << 16;
    int exp = (h >> 10) & 0x1F;
    uint32_t man = h & 0x3FFu;
    uint32_t f;
    if (exp == 0) {
        if (man == 0) f = sgn;
        else {
            exp = 1;
            while (!(man & 0x400)) { man <<= 1; exp--; }
            man &= 0x3FF;
            f = sgn | ((uint32_t)(exp + 112) << 23) | (man << 13);
        }
    } else if (exp == 31) {
        f = sgn | 0x7F800000u | (man << 13);
    } else {
        f = sgn | ((uint32_t)(exp + 112) << 23) | (man << 13);
    }
    float r;
    memcpy(&r, &f, 4);
    return r;
}

static void quantize_row_q8_K(const float *x, BlockQ8_K *y, int k) {
    const int nb = k / QK_K;
    for (int i = 0; i < nb; i++) {
        float maxv = 0.0f, amax = 0.0f;
        for (int j = 0; j < QK_K; ++j) {
            float ax = fabsf(x[j]);
            if (ax > amax) { amax = ax; maxv = x[j]; }
        }
        if (!amax) {
            y[i].d = 0.0f;
            memset(y[i].qs, 0, QK_K);
            memset(y[i].bsums, 0, QK_K / 16 * sizeof(int16_t));
            x += QK_K;
            continue;
        }
        const float iscale = -127.0f / maxv;
        for (int j = 0; j < QK_K; ++j) {
            int v = (int)lrintf(iscale * x[j]);
            y[i].qs[j] = (int8_t)(v > 127 ? 127 : v);
        }
        for (int j = 0; j < QK_K / 16; ++j) {
            int sum = 0;
            for (int ii = 0; ii < 16; ++ii) sum += y[i].qs[j * 16 + ii];
            y[i].bsums[j] = (int16_t)sum;
        }
        y[i].d = 1.0f / iscale;
        x += QK_K;
    }
}

static void g4_vec_dot_q4_K_q8_K(int n, const BlockQ4_K *x, const BlockQ8_K *y, float *out) {
    const int nb = n / QK_K;
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;
    uint32_t utmp[4];
    const uint8_t *scales = (const uint8_t *)&utmp[0];
    const uint8_t *mins   = (const uint8_t *)&utmp[2];
    int8_t aux8[QK_K];
    int16_t aux16[8];
    float sums[8];
    int32_t aux32[8];
    float sumf = 0.0f;
    memset(sums, 0, sizeof(sums));
    for (int i = 0; i < nb; ++i) {
        const uint8_t *q4 = x[i].qs;
        const int8_t *q8 = y[i].qs;
        memset(aux32, 0, 8 * sizeof(int32_t));
        int8_t *a = aux8;
        for (int j = 0; j < QK_K / 64; ++j) {
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
            a += 32;
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] >> 4);
            a += 32;
            q4 += 32;
        }
        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        {
            const uint32_t uaux = utmp[1] & kmask1;
            utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
            utmp[2] = uaux;
            utmp[0] &= kmask1;
        }
        int sumi = 0;
        for (int j = 0; j < QK_K / 16; ++j) sumi += y[i].bsums[j] * mins[j / 2];
        a = aux8;
        int is = 0;
        for (int j = 0; j < QK_K / 32; ++j) {
            const int32_t scale = scales[is++];
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = host_f16f32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * (float)aux32[l];
        const float dmin = host_f16f32(x[i].dmin) * y[i].d;
        sumf -= dmin * (float)sumi;
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *out = sumf;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s row.bin dim\n", argv[0]);
        return 1;
    }
    int dim = atoi(argv[2]);
    if (dim % QK_K) { fprintf(stderr, "dim must be multiple of %d\n", QK_K); return 1; }
    int nb = dim / QK_K;
    size_t row_sz = (size_t)nb * sizeof(BlockQ4_K);
    std::vector<uint8_t> row(row_sz);
    FILE *f = fopen(argv[1], "rb");
    if (!f || fread(row.data(), 1, row_sz, f) != row_sz) { perror("read"); return 1; }
    fclose(f);

    std::vector<float> x((size_t)dim);
    for (int i = 0; i < dim; i++) x[(size_t)i] = sinf(0.01f * (float)i);

    BlockQ8_K q8[(dim / QK_K)];
    quantize_row_q8_K(x.data(), q8, dim);

    float g4 = 0, gg = 0;
    g4_vec_dot_q4_K_q8_K(dim, (const BlockQ4_K *)row.data(), q8, &g4);
    ggml_vec_dot_q4_K_q8_K_generic(dim, &gg, 0, row.data(), 0, q8, 0, 1);

    printf("gemma4 generic: %.8f\n", g4);
    printf("ggml generic:   %.8f\n", gg);
    printf("diff:           %.8e\n", g4 - gg);
    return 0;
}
