#define _POSIX_C_SOURCE 200809L

/*
 * Gemma 4 E4B GGUF Inference Engine — CPU only (single-thread C).
 *
 * Target: gemma-4-E4B-it-Q4_K_M.gguf (Q4_K / Q5_K / Q6_K + BF16 + F32 norms).
 * Architecture: ISWA (interleaved sliding-window / full attention), shared KV tail,
 * per-layer embeddings (PLE), tied token_embd LM head, logit softcapping.
 *
 * Matmul / embed: K-quant blocks dequantized to stack temp[256] per block.
 * Build: make build (see Makefile in this directory).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#define GGUF_MAGIC      0x46554747u
#define QK_K            256
#define K_SCALE_SIZE    12
#define MAX_PROMPT_TOKS 8192
#define MAX_GEN_TOKS    4096
#define MAX_CHAT_TURNS  128
#define MAX_LINE        4096
#define MAX_Q_DIM       4096   /* 8 heads * 512 */
#define MAX_KV_DIM      1024   /* 2 kv_heads * 512 */
#define MAX_PLE_DIM     256
#define N_LAYER_KV      24     /* layers 0..23 store KV; tail reuses 22/23 */
#define GEMMA4_SP       "\xe2\x96\x81" /* U+2581 ▁ */

enum gguf_vtype {
    GV_U8 = 0, GV_I8, GV_U16, GV_I16, GV_U32, GV_I32, GV_F32, GV_BOOL,
    GV_STR, GV_ARR, GV_U64, GV_I64, GV_F64
};

enum ggml_dtype {
    DT_F32   = 0,
    DT_F16   = 1,
    DT_Q4_K  = 12,
    DT_Q5_K  = 13,
    DT_Q6_K  = 14,
    DT_BF16  = 30
};

#pragma pack(push, 1)
typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[K_SCALE_SIZE];
    uint8_t  qs[QK_K / 2];
} BlockQ4_K;

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[K_SCALE_SIZE];
    uint8_t  qh[QK_K / 8];
    uint8_t  qs[QK_K / 2];
} BlockQ5_K;

typedef struct {
    uint8_t  ql[QK_K / 2];
    uint8_t  qh[QK_K / 4];
    int8_t   scales[QK_K / 16];
    uint16_t d;
} BlockQ6_K;
#pragma pack(pop)

/* ================================================================
 * FP16 / BF16 helpers
 * ================================================================ */

static inline float host_f16f32(uint16_t h) {
    uint32_t sgn = ((uint32_t)(h & 0x8000)) << 16;
    int exp = (h >> 10) & 0x1F;
    uint32_t man = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (man == 0) { f = sgn; }
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

static inline float host_bf16f32(uint16_t h) {
    uint32_t f = (uint32_t)h << 16;
    float r;
    memcpy(&r, &f, 4);
    return r;
}

/* ================================================================
 * Dequantizers
 * ================================================================ */

static inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4)  | ((q[j - 0] >> 6) << 4);
    }
}

static void dequant_q4_k(const BlockQ4_K *x, float *y, int64_t nb) {
    for (int64_t i = 0; i < nb; i++) {
        const uint8_t *q = x[i].qs;
        const float d   = host_f16f32(x[i].d);
        const float min = host_f16f32(x[i].dmin);
        int is = 0;
        uint8_t sc, m;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc; const float m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc; const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l] >> 4)  - m2;
            q += 32; is += 2;
        }
    }
}

static void dequant_q5_k(const BlockQ5_K *x, float *y, int64_t nb) {
    for (int64_t i = 0; i < nb; i++) {
        const uint8_t *ql = x[i].qs;
        const uint8_t *qh = x[i].qh;
        const float d   = host_f16f32(x[i].d);
        const float min = host_f16f32(x[i].dmin);
        int is = 0;
        uint8_t sc, m;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc; const float m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc; const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * ((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * ((ql[l] >> 4)  + ((qh[l] & u2) ? 16 : 0)) - m2;
            ql += 32; is += 2;
            u1 <<= 2; u2 <<= 2;
        }
    }
}

static void dequant_q6_k(const BlockQ6_K *x, float *y, int64_t nb) {
    for (int64_t i = 0; i < nb; i++) {
        const float d = host_f16f32(x[i].d);
        const uint8_t *ql = x[i].ql;
        const uint8_t *qh = x[i].qh;
        const int8_t  *sc = x[i].scales;
        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                const int8_t q1 = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l + 0]  = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

/* ================================================================
 * Model structures
 * ================================================================ */

typedef struct {
    int dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, max_seq;
    int ple_dim, sliding_window, shared_kv_layers, swa_pattern;
    int head_dim_swa, head_dim_full, n_rot_swa, n_rot_full;
    float rope_theta, rope_theta_swa, norm_eps, logit_softcapping;
    int *swa_layers; /* per-layer: 1 = sliding window, 0 = full */
} Config;

typedef struct {
    char *name;
    int n_dims;
    uint64_t ne[4];
    int type;
    uint64_t offset;
} TensorInfo;

typedef struct {
    char **vocab;
    int *vlen;
    float *scores;
    int size, bos, eos, eot;
    int turn_start, turn_end, turn_user, turn_model, turn_system;
    int think, channel, channel_thought, channel_end;
    int *htab;
    int htab_sz;
} Tok;

enum { CHAT_USER = 0, CHAT_ASSISTANT = 1 };

typedef struct {
    int role;
    char *text;
} ChatMessage;

typedef struct {
    ChatMessage *msgs;
    int n, cap;
} ChatHistory;

typedef struct {
    int *tokens;
    int n, cap;
} TokenBuf;

typedef struct {
    void *embd;              int embd_t;
    void *ple_tok;           int ple_tok_t;
    void *ple_proj;          int ple_proj_t;
    float *ple_proj_norm;
    float *rope_freqs;
    float *norm_out;
    float **attn_norm;
    float **post_attn_norm;
    float **ffn_norm;
    float **post_ffw_norm;
    float **q_norm;
    float **k_norm;
    float **post_norm;
    float **out_scale;
    void **wq;               int *wq_t;
    void **wk;               int *wk_t;
    void **wv;               int *wv_t;
    void **wo;               int *wo_t;
    void **gate;             int *gate_t;
    void **up;               int *up_t;
    void **down;             int *down_t;
    void **inp_gate;         int *inp_gate_t;
    void **proj;             int *proj_t;
} Weights;

typedef struct {
    float *x, *xb, *xb2, *hb, *hb2;
    float *q, *k, *v, *att, *logits;
    float *q_out, *ple, *ple_ctx, *ple_gate;
    float *kc, *vc;
} State;

typedef struct {
    Config cfg;
    Weights w;
    State s;
    Tok tok;
    int fd;
    uint8_t *fdata;
    size_t fsz;
    TensorInfo *ti;
    int nti;
    uint64_t doff;
} Model;

static int is_swa_layer(const Config *c, int l) {
    if (c->swa_layers && l >= 0 && l < c->n_layers)
        return c->swa_layers[l];
    if (c->swa_pattern <= 0) return 0;
    return (l % c->swa_pattern) < (c->swa_pattern - 1);
}

static int layer_head_dim(const Config *c, int l) {
    return is_swa_layer(c, l) ? c->head_dim_swa : c->head_dim_full;
}

static int layer_q_dim(const Config *c, int l) {
    return c->n_heads * layer_head_dim(c, l);
}

static int layer_kv_dim(const Config *c, int l) {
    return c->n_kv_heads * layer_head_dim(c, l);
}

static int layer_n_rot(const Config *c, int l) {
    return is_swa_layer(c, l) ? c->n_rot_swa : c->n_rot_full;
}

static int layer_has_kv(int l) {
    return l < N_LAYER_KV;
}

static int kv_source_for_layer(const Config *c, int l) {
    if (layer_has_kv(l)) return l;
    return is_swa_layer(c, l) ? 22 : 23;
}

static size_t row_bytes_quant(int type, int n_in) {
    int nb = n_in / QK_K;
    switch (type) {
    case DT_Q4_K: return (size_t)nb * sizeof(BlockQ4_K);
    case DT_Q5_K: return (size_t)nb * sizeof(BlockQ5_K);
    case DT_Q6_K: return (size_t)nb * sizeof(BlockQ6_K);
    default: return 0;
    }
}

static size_t block_size_quant(int type) {
    switch (type) {
    case DT_Q4_K: return sizeof(BlockQ4_K);
    case DT_Q5_K: return sizeof(BlockQ5_K);
    case DT_Q6_K: return sizeof(BlockQ6_K);
    default: return 0;
    }
}

static void dequant_one_block_to(const void *blk, int type, float *dst) {
    switch (type) {
    case DT_Q4_K: dequant_q4_k((const BlockQ4_K *)blk, dst, 1); break;
    case DT_Q5_K: dequant_q5_k((const BlockQ5_K *)blk, dst, 1); break;
    case DT_Q6_K: dequant_q6_k((const BlockQ6_K *)blk, dst, 1); break;
    default:
        fprintf(stderr, "dequant_one_block_to: bad type %d\n", type);
        exit(1);
    }
}

/* ================================================================
 * GGUF reader
 * ================================================================ */

typedef struct { uint8_t *d; uint64_t p; } Rd;

static uint32_t ru32(Rd *r) { uint32_t v; memcpy(&v, r->d + r->p, 4); r->p += 4; return v; }
static uint64_t ru64(Rd *r) { uint64_t v; memcpy(&v, r->d + r->p, 8); r->p += 8; return v; }
static float    rf32(Rd *r) { float v;    memcpy(&v, r->d + r->p, 4); r->p += 4; return v; }

static char *rstr(Rd *r, int *out_len) {
    uint64_t n = ru64(r);
    char *s = (char *)malloc((size_t)n + 1);
    memcpy(s, r->d + r->p, (size_t)n);
    s[n] = 0;
    r->p += n;
    if (out_len) *out_len = (int)n;
    return s;
}

static void skip(Rd *r, uint32_t t) {
    switch (t) {
    case GV_U8: case GV_I8: case GV_BOOL: r->p++; break;
    case GV_U16: case GV_I16: r->p += 2; break;
    case GV_U32: case GV_I32: case GV_F32: r->p += 4; break;
    case GV_U64: case GV_I64: case GV_F64: r->p += 8; break;
    case GV_STR: { uint64_t n = ru64(r); r->p += n; break; }
    case GV_ARR: {
        uint32_t at = ru32(r); uint64_t al = ru64(r);
        for (uint64_t i = 0; i < al; i++) skip(r, at);
        break;
    }
    }
}

static int64_t read_int_val(Rd *r, uint32_t vt) {
    switch (vt) {
    case GV_U8: case GV_BOOL: { uint8_t v = r->d[r->p]; r->p++; return (int64_t)v; }
    case GV_I8:  { int8_t v; memcpy(&v, r->d + r->p, 1); r->p++; return (int64_t)v; }
    case GV_U16: { uint16_t v; memcpy(&v, r->d + r->p, 2); r->p += 2; return (int64_t)v; }
    case GV_I16: { int16_t v; memcpy(&v, r->d + r->p, 2); r->p += 2; return (int64_t)v; }
    case GV_U32: return (int64_t)ru32(r);
    case GV_I32: { int32_t v; memcpy(&v, r->d + r->p, 4); r->p += 4; return (int64_t)v; }
    case GV_U64: return (int64_t)ru64(r);
    case GV_I64: { int64_t v; memcpy(&v, r->d + r->p, 8); r->p += 8; return v; }
    case GV_F32: return (int64_t)rf32(r);
    case GV_F64: { double v; memcpy(&v, r->d + r->p, 8); r->p += 8; return (int64_t)v; }
    default: skip(r, vt); return 0;
    }
}

static float read_float_val(Rd *r, uint32_t vt) {
    switch (vt) {
    case GV_F32: return rf32(r);
    case GV_F64: { double v; memcpy(&v, r->d + r->p, 8); r->p += 8; return (float)v; }
    case GV_U32: return (float)ru32(r);
    case GV_I32: { int32_t v; memcpy(&v, r->d + r->p, 4); r->p += 4; return (float)v; }
    case GV_U64: return (float)ru64(r);
    case GV_I64: { int64_t v; memcpy(&v, r->d + r->p, 8); r->p += 8; return (float)v; }
    default: skip(r, vt); return 0.0f;
    }
}

static void parse_gguf(Model *m, char ***out_merges, int *out_n_merges) {
    Rd r = { m->fdata, 0 };
    if (ru32(&r) != GGUF_MAGIC) { fprintf(stderr, "Error: invalid GGUF magic\n"); exit(1); }
    uint32_t ver = ru32(&r);
    uint64_t n_tensors = ru64(&r);
    uint64_t n_kv      = ru64(&r);

    printf("GGUF v%u | %llu tensors | %llu metadata entries\n",
           ver, (unsigned long long)n_tensors, (unsigned long long)n_kv);

    Config *c = &m->cfg;
    Tok *tk = &m->tok;

    c->rope_theta = 1e6f;
    c->rope_theta_swa = 10000.0f;
    c->norm_eps = 1e-6f;
    c->max_seq = 8192;
    c->swa_pattern = 6;
    c->head_dim_swa = 256;
    c->head_dim_full = 512;
    c->n_rot_swa = 256;
    c->n_rot_full = 512;
    c->ple_dim = 256;
    c->sliding_window = 512;
    c->shared_kv_layers = 18;
    c->logit_softcapping = 30.0f;
    c->swa_layers = NULL;
    tk->bos = 2;
    tk->eos = 106;
    tk->eot = 106;

    *out_merges = NULL;
    *out_n_merges = 0;
    uint32_t alignment = 32;

    for (uint64_t i = 0; i < n_kv; i++) {
        char *key = rstr(&r, NULL);
        uint32_t vt = ru32(&r);

        if      (!strcmp(key, "general.alignment"))                         alignment = (uint32_t)read_int_val(&r, vt);
        else if (!strcmp(key, "gemma4.embedding_length"))                   c->dim = (int)read_int_val(&r, vt);
        else if (!strcmp(key, "gemma4.feed_forward_length"))                c->hidden_dim = (int)read_int_val(&r, vt);
        else if (!strcmp(key, "gemma4.block_count"))                        c->n_layers = (int)read_int_val(&r, vt);
        else if (!strcmp(key, "gemma4.attention.head_count"))               c->n_heads = (int)read_int_val(&r, vt);
        else if (!strcmp(key, "gemma4.attention.head_count_kv"))            c->n_kv_heads = (int)read_int_val(&r, vt);
        else if (!strcmp(key, "gemma4.attention.key_length_swa"))           c->head_dim_swa = (int)read_int_val(&r, vt);
        else if (!strcmp(key, "gemma4.attention.key_length"))               c->head_dim_full = (int)read_int_val(&r, vt);
        else if (!strcmp(key, "gemma4.rope.dimension_count_swa"))           c->n_rot_swa = (int)read_int_val(&r, vt);
        else if (!strcmp(key, "gemma4.rope.dimension_count"))               c->n_rot_full = (int)read_int_val(&r, vt);
        else if (!strcmp(key, "gemma4.attention.layer_norm_rms_epsilon"))   c->norm_eps = read_float_val(&r, vt);
        else if (!strcmp(key, "gemma4.rope.freq_base"))                     c->rope_theta = read_float_val(&r, vt);
        else if (!strcmp(key, "gemma4.rope.freq_base_swa"))                 c->rope_theta_swa = read_float_val(&r, vt);
        else if (!strcmp(key, "gemma4.attention.sliding_window"))             c->sliding_window = (int)read_int_val(&r, vt);
        else if (!strcmp(key, "gemma4.attention.shared_kv_layers"))           c->shared_kv_layers = (int)read_int_val(&r, vt);
        else if (!strcmp(key, "gemma4.embedding_length_per_layer_input"))   c->ple_dim = (int)read_int_val(&r, vt);
        else if (!strcmp(key, "gemma4.final_logit_softcapping"))              c->logit_softcapping = read_float_val(&r, vt);
        else if (!strcmp(key, "gemma4.attention.sliding_window_pattern")) {
            if (vt != GV_ARR) { skip(&r, vt); free(key); continue; }
            uint32_t et = ru32(&r);
            uint64_t n = ru64(&r);
            if (n > 0 && n <= 256) {
                c->swa_layers = (int *)calloc((size_t)n, sizeof(int));
                for (uint64_t j = 0; j < n; j++) {
                    if (et == GV_BOOL || et == GV_U8 || et == GV_I8) {
                        c->swa_layers[j] = read_int_val(&r, et) ? 1 : 0;
                    } else {
                        c->swa_layers[j] = read_int_val(&r, et) ? 1 : 0;
                    }
                }
            } else {
                for (uint64_t j = 0; j < n; j++) skip(&r, et);
            }
        }
        else if (!strcmp(key, "gemma4.context_length"))                       c->max_seq = (int)read_int_val(&r, vt);
        else if (!strcmp(key, "tokenizer.ggml.bos_token_id"))                 tk->bos = (int)read_int_val(&r, vt);
        else if (!strcmp(key, "tokenizer.ggml.eos_token_id")) {
            tk->eos = (int)read_int_val(&r, vt);
            tk->eot = tk->eos;
        }
        else if (!strcmp(key, "tokenizer.ggml.tokens")) {
            if (vt != GV_ARR) { skip(&r, vt); free(key); continue; }
            ru32(&r);
            uint64_t n = ru64(&r);
            tk->size = (int)n;
            c->vocab_size = (int)n;
            tk->vocab = (char **)calloc(n, sizeof(char *));
            tk->vlen  = (int *)calloc(n, sizeof(int));
            for (uint64_t j = 0; j < n; j++)
                tk->vocab[j] = rstr(&r, &tk->vlen[j]);
        }
        else if (!strcmp(key, "tokenizer.ggml.scores")) {
            if (vt != GV_ARR) { skip(&r, vt); free(key); continue; }
            ru32(&r);
            uint64_t n = ru64(&r);
            tk->scores = (float *)malloc(n * sizeof(float));
            for (uint64_t j = 0; j < n; j++) tk->scores[j] = rf32(&r);
        }
        else if (!strcmp(key, "tokenizer.ggml.merges")) {
            if (vt != GV_ARR) { skip(&r, vt); free(key); continue; }
            ru32(&r);
            uint64_t n = ru64(&r);
            *out_n_merges = (int)n;
            *out_merges = (char **)malloc(n * sizeof(char *));
            for (uint64_t j = 0; j < n; j++)
                (*out_merges)[j] = rstr(&r, NULL);
        }
        else { skip(&r, vt); }

        free(key);
    }

    m->nti = (int)n_tensors;
    m->ti  = (TensorInfo *)calloc(n_tensors, sizeof(TensorInfo));
    for (uint64_t i = 0; i < n_tensors; i++) {
        m->ti[i].name   = rstr(&r, NULL);
        m->ti[i].n_dims = (int)ru32(&r);
        for (int d = 0; d < m->ti[i].n_dims; d++)
            m->ti[i].ne[d] = ru64(&r);
        m->ti[i].type   = (int)ru32(&r);
        m->ti[i].offset = ru64(&r);
    }

    m->doff = (r.p + (alignment - 1)) & ~(uint64_t)(alignment - 1);
}

/* ================================================================
 * Tokenizer (Gemma BPE: escape spaces, split newlines, no byte fallback)
 * ================================================================ */

static unsigned int hash_bytes(const char *s, int len) {
    unsigned int h = 2166136261u;
    for (int i = 0; i < len; i++) {
        h ^= (unsigned char)s[i];
        h *= 16777619u;
    }
    return h;
}

static void tok_build_hash(Tok *tk) {
    tk->htab_sz = tk->size * 2;
    tk->htab = (int *)malloc(tk->htab_sz * sizeof(int));
    for (int i = 0; i < tk->htab_sz; i++) tk->htab[i] = -1;
    for (int i = 0; i < tk->size; i++) {
        unsigned int h = hash_bytes(tk->vocab[i], tk->vlen[i]) % tk->htab_sz;
        while (tk->htab[h] != -1) h = (h + 1) % tk->htab_sz;
        tk->htab[h] = i;
    }
}

static int tok_lookup(Tok *tk, const char *s, int len) {
    unsigned int h = hash_bytes(s, len) % tk->htab_sz;
    while (tk->htab[h] != -1) {
        int id = tk->htab[h];
        if (tk->vlen[id] == len && memcmp(tk->vocab[id], s, len) == 0) return id;
        h = (h + 1) % tk->htab_sz;
    }
    return -1;
}

static int tok_find_special(Tok *tk, const char *name) {
    int nlen = (int)strlen(name);
    for (int i = tk->size - 1; i >= 0; i--) {
        if (tk->vlen[i] == nlen && memcmp(tk->vocab[i], name, nlen) == 0) return i;
    }
    return -1;
}

static void init_tokenizer(Tok *tk, char **merges, int n_merges) {
    if (!tk->scores && tk->size > 0)
        tk->scores = (float *)calloc(tk->size, sizeof(float));
    tok_build_hash(tk);

    if (n_merges > 0 && merges) {
        for (int i = 0; i < n_merges; i++) {
            char *sp = strchr(merges[i], ' ');
            if (!sp) { free(merges[i]); continue; }
            int la = (int)(sp - merges[i]);
            int lb = (int)strlen(sp + 1);
            char *buf = (char *)malloc((size_t)la + lb + 1);
            memcpy(buf, merges[i], (size_t)la);
            memcpy(buf + la, sp + 1, (size_t)lb);
            buf[la + lb] = 0;
            int id = tok_lookup(tk, buf, la + lb);
            if (id >= 0) tk->scores[id] = (float)(n_merges - i);
            free(buf);
            free(merges[i]);
        }
        free(merges);
    }

    tk->turn_start = tok_find_special(tk, "<|turn>");
    tk->turn_end   = tok_find_special(tk, "<turn|>");
    tk->turn_user  = tok_find_special(tk, "<|turn>user\n");
    tk->turn_model = tok_find_special(tk, "<|turn>model\n");
    tk->turn_system = tok_find_special(tk, "<|turn>system\n");
    tk->think = tok_find_special(tk, "<|think|>");
    tk->channel = tok_find_special(tk, "<|channel>");
    tk->channel_thought = tok_find_special(tk, "<|channel>thought\n");
    if (tk->channel_thought < 0)
        tk->channel_thought = tok_find_special(tk, "<|channel>thought");
    tk->channel_end = tok_find_special(tk, "<channel|>");

    printf("Tokenizer: %d tokens | BOS=%d EOS=%d think=%d channel=%d end=%d\n",
           tk->size, tk->bos, tk->eos, tk->think, tk->channel, tk->channel_end);
}

static int utf8_char_len(unsigned char c) {
    if ((c & 0x80) == 0) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}

static char *gemma_escape_line(const char *line, int line_len, int *out_len) {
    int cap = line_len * 4 + 1;
    char *buf = (char *)malloc((size_t)cap);
    int o = 0;
    for (int i = 0; i < line_len; ) {
        unsigned char c = (unsigned char)line[i];
        if (c == ' ') {
            memcpy(buf + o, GEMMA4_SP, 3);
            o += 3;
            i++;
        } else {
            int n = utf8_char_len(c);
            if (i + n > line_len) n = line_len - i;
            memcpy(buf + o, line + i, (size_t)n);
            o += n;
            i += n;
        }
    }
    buf[o] = 0;
    *out_len = o;
    return buf;
}

static int *bpe_merge(Tok *tk, int *tokens, int n, int *out_n) {
    char buf[MAX_PROMPT_TOKS * 8];
    while (n > 1) {
        float best_score = -1e20f;
        int best_id = -1, best_idx = -1;

        for (int i = 0; i < n - 1; i++) {
            int total = tk->vlen[tokens[i]] + tk->vlen[tokens[i + 1]];
            if (total >= (int)sizeof(buf)) continue;
            memcpy(buf, tk->vocab[tokens[i]], (size_t)tk->vlen[tokens[i]]);
            memcpy(buf + tk->vlen[tokens[i]], tk->vocab[tokens[i + 1]], (size_t)tk->vlen[tokens[i + 1]]);
            int id = tok_lookup(tk, buf, total);
            if (id >= 0 && tk->scores[id] > best_score) {
                best_score = tk->scores[id];
                best_id = id;
                best_idx = i;
            }
        }
        if (best_idx < 0) break;
        tokens[best_idx] = best_id;
        for (int i = best_idx + 1; i < n - 1; i++) tokens[i] = tokens[i + 1];
        n--;
    }
    *out_n = n;
    return tokens;
}

static int *gemma_bpe_encode(Tok *tk, const char *text, int text_len, int *out_n) {
    int *tokens = (int *)malloc((size_t)(text_len + 8) * sizeof(int));
    int n = 0;

    for (int i = 0; i < text_len; ) {
        unsigned char c = (unsigned char)text[i];
        int clen = utf8_char_len(c);
        if (i + clen > text_len) clen = text_len - i;
        int id = tok_lookup(tk, text + i, clen);
        if (id < 0) {
            fprintf(stderr, "Warning: unknown token bytes at offset %d\n", i);
            i += clen;
            continue;
        }
        tokens[n++] = id;
        i += clen;
    }

    return bpe_merge(tk, tokens, n, out_n);
}

static void append_tokens(int *out, int *n, int *src, int ns) {
    for (int i = 0; i < ns; i++) out[(*n)++] = src[i];
}

static void append_str_tok(Tok *tk, int *out, int *n, const char *s) {
    int id = tok_find_special(tk, s);
    if (id >= 0) {
        out[(*n)++] = id;
        return;
    }
    int nt;
    int *t = gemma_bpe_encode(tk, s, (int)strlen(s), &nt);
    append_tokens(out, n, t, nt);
    free(t);
}

static void append_gemma_text(Tok *tk, int *out, int *n, const char *text) {
    const char *p = text;
    const char *line = p;
    while (*p) {
        if (*p == '\n') {
            int line_len = (int)(p - line);
            if (line_len > 0) {
                int elen;
                char *escaped = gemma_escape_line(line, line_len, &elen);
                int nt;
                int *t = gemma_bpe_encode(tk, escaped, elen, &nt);
                append_tokens(out, n, t, nt);
                free(t);
                free(escaped);
            }
            append_str_tok(tk, out, n, "\n");
            p++;
            line = p;
        } else {
            p++;
        }
    }
    if (p > line) {
        int line_len = (int)(p - line);
        int elen;
        char *escaped = gemma_escape_line(line, line_len, &elen);
        int nt;
        int *t = gemma_bpe_encode(tk, escaped, elen, &nt);
        append_tokens(out, n, t, nt);
        free(t);
        free(escaped);
    }
}

static void append_turn_end(Tok *tk, int *out, int *n) {
    if (tk->turn_end >= 0) {
        out[(*n)++] = tk->turn_end;
        append_str_tok(tk, out, n, "\n");
    } else {
        append_str_tok(tk, out, n, "<turn|>\n");
    }
}

static void append_user_turn(Tok *tk, int *out, int *n, const char *text) {
    if (tk->turn_user >= 0) {
        out[(*n)++] = tk->turn_user;
    } else {
        append_str_tok(tk, out, n, "<|turn>user\n");
    }
    append_gemma_text(tk, out, n, text);
    append_turn_end(tk, out, n);
}

static void append_model_turn(Tok *tk, int *out, int *n, const char *text) {
    if (tk->turn_model >= 0) {
        out[(*n)++] = tk->turn_model;
    } else {
        append_str_tok(tk, out, n, "<|turn>model\n");
    }
    append_gemma_text(tk, out, n, text);
    append_turn_end(tk, out, n);
}

static void append_model_prefix(Tok *tk, int *out, int *n) {
    if (tk->turn_model >= 0) {
        out[(*n)++] = tk->turn_model;
    } else {
        append_str_tok(tk, out, n, "<|turn>model\n");
    }
}

static void append_system_think_turn(Tok *tk, int *out, int *n) {
    if (tk->turn_system >= 0) {
        out[(*n)++] = tk->turn_system;
    } else {
        append_str_tok(tk, out, n, "<|turn>system\n");
    }
    if (tk->think >= 0) {
        out[(*n)++] = tk->think;
    } else {
        append_str_tok(tk, out, n, "<|think|>");
    }
    append_turn_end(tk, out, n);
}

static void chat_history_init(ChatHistory *h) {
    h->msgs = NULL;
    h->n = 0;
    h->cap = 0;
}

static void chat_history_free(ChatHistory *h) {
    for (int i = 0; i < h->n; i++) free(h->msgs[i].text);
    free(h->msgs);
    h->msgs = NULL;
    h->n = h->cap = 0;
}

static void chat_history_push(ChatHistory *h, int role, const char *text) {
    if (h->n >= MAX_CHAT_TURNS) {
        fprintf(stderr, "Error: max %d turns reached\n", MAX_CHAT_TURNS);
        exit(1);
    }
    if (h->n >= h->cap) {
        h->cap = h->cap ? h->cap * 2 : 8;
        h->msgs = (ChatMessage *)realloc(h->msgs, (size_t)h->cap * sizeof(ChatMessage));
    }
    h->msgs[h->n].role = role;
    h->msgs[h->n].text = strdup(text);
    h->n++;
}

static int chat_encode_history(Tok *tk, const ChatHistory *hist, int enable_thinking,
                               int *toks, int max_toks) {
    int n = 0;
    if (n >= max_toks) return -1;
    toks[n++] = tk->bos;

    if (enable_thinking) {
        append_system_think_turn(tk, toks, &n);
        if (n >= max_toks) return -1;
    }

    for (int i = 0; i < hist->n; i++) {
        if (hist->msgs[i].role == CHAT_USER) {
            append_user_turn(tk, toks, &n, hist->msgs[i].text);
        } else {
            append_model_turn(tk, toks, &n, hist->msgs[i].text);
        }
        if (n >= max_toks) return -1;
    }

    append_model_prefix(tk, toks, &n);
    if (n >= max_toks) return -1;
    return n;
}

static int *chat_encode(Tok *tk, const char *prompt, int enable_thinking, int *out_n) {
    ChatHistory hist;
    chat_history_init(&hist);
    chat_history_push(&hist, CHAT_USER, prompt);
    int *toks = (int *)malloc(MAX_PROMPT_TOKS * sizeof(int));
    int n = chat_encode_history(tk, &hist, enable_thinking, toks, MAX_PROMPT_TOKS);
    chat_history_free(&hist);
    if (n < 0) {
        free(toks);
        fprintf(stderr, "Error: prompt too long (max %d tokens)\n", MAX_PROMPT_TOKS);
        exit(1);
    }
    *out_n = n;
    return toks;
}

/* ================================================================
 * Weight loading
 * ================================================================ */

static void *find_tensor(Model *m, const char *name, int *out_type) {
    for (int i = 0; i < m->nti; i++) {
        if (strcmp(m->ti[i].name, name) == 0) {
            if (out_type) *out_type = m->ti[i].type;
            return m->fdata + m->doff + m->ti[i].offset;
        }
    }
    return NULL;
}

static void load_weights(Model *m) {
    Config *c = &m->cfg;
    Weights *w = &m->w;
    int L = c->n_layers;

    w->embd = find_tensor(m, "token_embd.weight", &w->embd_t);
    w->ple_tok = find_tensor(m, "per_layer_token_embd.weight", &w->ple_tok_t);
    w->ple_proj = find_tensor(m, "per_layer_model_proj.weight", &w->ple_proj_t);
    w->ple_proj_norm = (float *)find_tensor(m, "per_layer_proj_norm.weight", NULL);
    w->rope_freqs = (float *)find_tensor(m, "rope_freqs.weight", NULL);
    w->norm_out = (float *)find_tensor(m, "output_norm.weight", NULL);

    w->attn_norm = (float **)calloc(L, sizeof(float *));
    w->post_attn_norm = (float **)calloc(L, sizeof(float *));
    w->ffn_norm = (float **)calloc(L, sizeof(float *));
    w->post_ffw_norm = (float **)calloc(L, sizeof(float *));
    w->q_norm = (float **)calloc(L, sizeof(float *));
    w->k_norm = (float **)calloc(L, sizeof(float *));
    w->post_norm = (float **)calloc(L, sizeof(float *));
    w->out_scale = (float **)calloc(L, sizeof(float *));
    w->wq = (void **)calloc(L, sizeof(void *));  w->wq_t = (int *)calloc(L, sizeof(int));
    w->wk = (void **)calloc(L, sizeof(void *));  w->wk_t = (int *)calloc(L, sizeof(int));
    w->wv = (void **)calloc(L, sizeof(void *));  w->wv_t = (int *)calloc(L, sizeof(int));
    w->wo = (void **)calloc(L, sizeof(void *));  w->wo_t = (int *)calloc(L, sizeof(int));
    w->gate = (void **)calloc(L, sizeof(void *)); w->gate_t = (int *)calloc(L, sizeof(int));
    w->up = (void **)calloc(L, sizeof(void *));   w->up_t = (int *)calloc(L, sizeof(int));
    w->down = (void **)calloc(L, sizeof(void *)); w->down_t = (int *)calloc(L, sizeof(int));
    w->inp_gate = (void **)calloc(L, sizeof(void *)); w->inp_gate_t = (int *)calloc(L, sizeof(int));
    w->proj = (void **)calloc(L, sizeof(void *)); w->proj_t = (int *)calloc(L, sizeof(int));

    char name[128];
    for (int l = 0; l < L; l++) {
        sprintf(name, "blk.%d.attn_norm.weight", l);
        w->attn_norm[l] = (float *)find_tensor(m, name, NULL);
        sprintf(name, "blk.%d.post_attention_norm.weight", l);
        w->post_attn_norm[l] = (float *)find_tensor(m, name, NULL);
        sprintf(name, "blk.%d.attn_q_norm.weight", l);
        w->q_norm[l] = (float *)find_tensor(m, name, NULL);
        sprintf(name, "blk.%d.attn_k_norm.weight", l);
        w->k_norm[l] = (float *)find_tensor(m, name, NULL);
        sprintf(name, "blk.%d.attn_q.weight", l);
        w->wq[l] = find_tensor(m, name, &w->wq_t[l]);
        sprintf(name, "blk.%d.attn_k.weight", l);
        w->wk[l] = find_tensor(m, name, &w->wk_t[l]);
        sprintf(name, "blk.%d.attn_v.weight", l);
        w->wv[l] = find_tensor(m, name, &w->wv_t[l]);
        sprintf(name, "blk.%d.attn_output.weight", l);
        w->wo[l] = find_tensor(m, name, &w->wo_t[l]);
        sprintf(name, "blk.%d.ffn_norm.weight", l);
        w->ffn_norm[l] = (float *)find_tensor(m, name, NULL);
        sprintf(name, "blk.%d.post_ffw_norm.weight", l);
        w->post_ffw_norm[l] = (float *)find_tensor(m, name, NULL);
        sprintf(name, "blk.%d.ffn_gate.weight", l);
        w->gate[l] = find_tensor(m, name, &w->gate_t[l]);
        sprintf(name, "blk.%d.ffn_up.weight", l);
        w->up[l] = find_tensor(m, name, &w->up_t[l]);
        sprintf(name, "blk.%d.ffn_down.weight", l);
        w->down[l] = find_tensor(m, name, &w->down_t[l]);
        sprintf(name, "blk.%d.inp_gate.weight", l);
        w->inp_gate[l] = find_tensor(m, name, &w->inp_gate_t[l]);
        sprintf(name, "blk.%d.proj.weight", l);
        w->proj[l] = find_tensor(m, name, &w->proj_t[l]);
        sprintf(name, "blk.%d.post_norm.weight", l);
        w->post_norm[l] = (float *)find_tensor(m, name, NULL);
        sprintf(name, "blk.%d.layer_output_scale.weight", l);
        w->out_scale[l] = (float *)find_tensor(m, name, NULL);
    }

    if (!w->embd || !w->norm_out || !w->ple_tok || !w->ple_proj || !w->ple_proj_norm) {
        fprintf(stderr, "Error: missing critical global tensors\n");
        exit(1);
    }
}

static void alloc_state(State *s, Config *c) {
    int kv_slots = N_LAYER_KV;
    size_t kv_len = (size_t)kv_slots * c->max_seq * MAX_KV_DIM;
    s->x         = (float *)calloc(c->dim, sizeof(float));
    s->xb        = (float *)calloc(MAX_Q_DIM, sizeof(float));
    s->xb2       = (float *)calloc(c->dim, sizeof(float));
    s->hb        = (float *)calloc(c->hidden_dim, sizeof(float));
    s->hb2       = (float *)calloc(c->hidden_dim, sizeof(float));
    s->q         = (float *)calloc(MAX_Q_DIM, sizeof(float));
    s->k         = (float *)calloc(MAX_KV_DIM, sizeof(float));
    s->v         = (float *)calloc(MAX_KV_DIM, sizeof(float));
    s->q_out     = (float *)calloc(MAX_Q_DIM, sizeof(float));
    s->ple       = (float *)calloc((size_t)c->ple_dim * c->n_layers, sizeof(float));
    s->ple_ctx   = (float *)calloc((size_t)c->ple_dim * c->n_layers, sizeof(float));
    s->ple_gate  = (float *)calloc(c->ple_dim, sizeof(float));
    s->att       = (float *)calloc((size_t)c->n_heads * c->max_seq, sizeof(float));
    s->logits    = (float *)calloc(c->vocab_size, sizeof(float));
    s->kc        = (float *)calloc(kv_len, sizeof(float));
    s->vc        = (float *)calloc(kv_len, sizeof(float));
}

static void free_state(State *s) {
    free(s->x); free(s->xb); free(s->xb2);
    free(s->hb); free(s->hb2);
    free(s->q); free(s->k); free(s->v); free(s->q_out);
    free(s->ple); free(s->ple_ctx); free(s->ple_gate);
    free(s->att); free(s->logits);
    free(s->kc); free(s->vc);
}

static void free_weight_ptrs(Weights *w, int L) {
    free(w->attn_norm); free(w->post_attn_norm);
    free(w->ffn_norm); free(w->post_ffw_norm);
    free(w->q_norm); free(w->k_norm); free(w->post_norm); free(w->out_scale);
    free(w->wq); free(w->wq_t); free(w->wk); free(w->wk_t);
    free(w->wv); free(w->wv_t); free(w->wo); free(w->wo_t);
    free(w->gate); free(w->gate_t); free(w->up); free(w->up_t); free(w->down); free(w->down_t);
    free(w->inp_gate); free(w->inp_gate_t); free(w->proj); free(w->proj_t);
    memset(w, 0, sizeof(*w));
    (void)L;
}

/* ================================================================
 * Math primitives
 * ================================================================ */

static void rmsnorm(float *o, const float *x, const float *weight, int n, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / n + eps);
    for (int i = 0; i < n; i++) o[i] = x[i] * ss * weight[i];
}

static void rmsnorm_head_inplace(float *vec, const float *w, int n_heads, int hd, float eps) {
    for (int h = 0; h < n_heads; h++) {
        float *seg = vec + h * hd;
        float ss = 0.0f;
        for (int i = 0; i < hd; i++) ss += seg[i] * seg[i];
        ss = 1.0f / sqrtf(ss / hd + eps);
        for (int i = 0; i < hd; i++) seg[i] *= ss * w[i];
    }
}

static void rmsnorm_head_noweight_inplace(float *vec, int n_heads, int hd, float eps) {
    for (int h = 0; h < n_heads; h++) {
        float *seg = vec + h * hd;
        float ss = 0.0f;
        for (int i = 0; i < hd; i++) ss += seg[i] * seg[i];
        ss = 1.0f / sqrtf(ss / hd + eps);
        for (int i = 0; i < hd; i++) seg[i] *= ss;
    }
}

static void softmax(float *x, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++)
        if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

static float gelu_f32(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * x * (1.0f + 0.044715f * x * x)));
}

static void mm_f32(float *o, const float *x, const float *w, int n, int d) {
    for (int i = 0; i < d; i++) {
        const float *row = w + (size_t)i * n;
        float val = 0.0f;
        for (int j = 0; j < n; j++) val += x[j] * row[j];
        o[i] = val;
    }
}

static void mm_f16(float *o, const float *x, const uint16_t *w, int n, int d) {
    for (int i = 0; i < d; i++) {
        const uint16_t *row = w + (size_t)i * n;
        float val = 0.0f;
        for (int j = 0; j < n; j++) val += x[j] * host_f16f32(row[j]);
        o[i] = val;
    }
}

static void mm_bf16(float *o, const float *x, const uint16_t *w, int n, int d) {
    for (int i = 0; i < d; i++) {
        const uint16_t *row = w + (size_t)i * n;
        float val = 0.0f;
        for (int j = 0; j < n; j++) val += x[j] * host_bf16f32(row[j]);
        o[i] = val;
    }
}

static void mm_quant_rows(float *o, const float *x, const void *w, int n, int d, int type) {
    int nb = n / QK_K;
    size_t row_sz = row_bytes_quant(type, n);
    size_t bs = block_size_quant(type);
    const uint8_t *wb = (const uint8_t *)w;
    float blk[QK_K];
    for (int i = 0; i < d; i++) {
        const uint8_t *row = wb + (size_t)i * row_sz;
        float val = 0.0f;
        for (int b = 0; b < nb; b++) {
            dequant_one_block_to(row + (size_t)b * bs, type, blk);
            const float *xp = x + b * QK_K;
            for (int j = 0; j < QK_K; j++) val += xp[j] * blk[j];
        }
        o[i] = val;
    }
}

static void mm(float *o, const float *x, const void *w, int n, int d, int type) {
    switch (type) {
    case DT_F32: mm_f32(o, x, (const float *)w, n, d); break;
    case DT_F16: mm_f16(o, x, (const uint16_t *)w, n, d); break;
    case DT_BF16: mm_bf16(o, x, (const uint16_t *)w, n, d); break;
    case DT_Q4_K: case DT_Q5_K: case DT_Q6_K:
        mm_quant_rows(o, x, w, n, d, type);
        break;
    default:
        fprintf(stderr, "Unsupported tensor type %d in matmul\n", type);
        exit(1);
    }
}

static void emb_lookup(float *o, const void *w, int type, int id, int dim) {
    switch (type) {
    case DT_F32:
        memcpy(o, (const float *)w + (size_t)id * dim, (size_t)dim * sizeof(float));
        break;
    case DT_F16: {
        const uint16_t *row = (const uint16_t *)w + (size_t)id * dim;
        for (int i = 0; i < dim; i++) o[i] = host_f16f32(row[i]);
        break;
    }
    case DT_Q4_K: case DT_Q5_K: case DT_Q6_K: {
        int nb = dim / QK_K;
        size_t row_sz = row_bytes_quant(type, dim);
        size_t bs = block_size_quant(type);
        const uint8_t *row = (const uint8_t *)w + (size_t)id * row_sz;
        float blk[QK_K];
        for (int b = 0; b < nb; b++) {
            dequant_one_block_to(row + (size_t)b * bs, type, blk);
            memcpy(o + b * QK_K, blk, QK_K * sizeof(float));
        }
        break;
    }
    default:
        fprintf(stderr, "Unsupported emb type %d\n", type);
        exit(1);
    }
}

static void apply_rope_neox(float *vec, int n_heads, int head_dim, int n_rot, int pos,
                            float freq_base, float freq_scale, const float *freq_factors) {
    float theta_scale = powf(freq_base, -2.0f / (float)n_rot);
    for (int h = 0; h < n_heads; h++) {
        float theta = (float)pos;
        for (int i = 0; i < n_rot; i += 2) {
            float ff = freq_factors ? freq_factors[i / 2] : 1.0f;
            float angle = freq_scale * theta / ff;
            float cr = cosf(angle);
            float ci = sinf(angle);
            int idx = h * head_dim + i;
            float v0 = vec[idx], v1 = vec[idx + 1];
            vec[idx]     = v0 * cr - v1 * ci;
            vec[idx + 1] = v0 * ci + v1 * cr;
            theta *= theta_scale;
        }
    }
}

static void build_ple(Model *m, int token, float *ple_out) {
    Config *c = &m->cfg;
    Weights *w = &m->w;
    State *s = &m->s;
    int dim = c->dim;
    int L = c->n_layers;
    int pd = c->ple_dim;
    int ple_row_dim = pd * L;
    float *ple_row = (float *)malloc((size_t)ple_row_dim * sizeof(float));
    emb_lookup(ple_row, w->ple_tok, w->ple_tok_t, token, ple_row_dim);

    float scale_tok = sqrtf((float)pd);
    for (int i = 0; i < ple_row_dim; i++) ple_row[i] *= scale_tok;

    mm(s->ple_ctx, s->x, w->ple_proj, dim, ple_row_dim, w->ple_proj_t);
    float scale_ctx = 1.0f / sqrtf((float)dim);
    for (int i = 0; i < ple_row_dim; i++) s->ple_ctx[i] *= scale_ctx;

    for (int l = 0; l < L; l++)
        rmsnorm(s->ple_ctx + (size_t)l * pd, s->ple_ctx + (size_t)l * pd,
                w->ple_proj_norm, pd, c->norm_eps);

    float scale_mix = 1.0f / sqrtf(2.0f);
    for (int l = 0; l < L; l++) {
        float *dst = ple_out + (size_t)l * pd;
        const float *tok = ple_row + (size_t)l * pd;
        const float *ctx = s->ple_ctx + (size_t)l * pd;
        for (int i = 0; i < pd; i++)
            dst[i] = (tok[i] + ctx[i]) * scale_mix;
    }
    free(ple_row);
}

static void logits_tied(Model *m, float *logits) {
    Config *c = &m->cfg;
    Weights *w = &m->w;
    State *s = &m->s;
    mm(logits, s->x, w->embd, c->dim, c->vocab_size, w->embd_t);
    float cap = c->logit_softcapping;
    if (cap > 0.0f) {
        float inv = 1.0f / cap;
        for (int i = 0; i < c->vocab_size; i++)
            logits[i] = cap * tanhf(logits[i] * inv);
    }
}

static size_t kv_cache_offset(int layer, int pos, int kv_dim, int max_seq) {
    return ((size_t)layer * max_seq + (size_t)pos) * (size_t)MAX_KV_DIM + 0;
    (void)kv_dim;
}

static float *kc_at(State *s, int layer, int pos, int max_seq) {
    return s->kc + kv_cache_offset(layer, pos, 0, max_seq);
}

static float *vc_at(State *s, int layer, int pos, int max_seq) {
    return s->vc + kv_cache_offset(layer, pos, 0, max_seq);
}

static void forward(Model *m, int token, int pos) {
    Config *c = &m->cfg;
    Weights *w = &m->w;
    State *s = &m->s;
    int dim = c->dim;
    int hidden = c->hidden_dim;
    int n_heads = c->n_heads;
    int n_kv = c->n_kv_heads;
    int max_seq = c->max_seq;
    int kv_mul = n_heads / n_kv;

    emb_lookup(s->x, w->embd, w->embd_t, token, dim);
    {
        float scale = sqrtf((float)dim);
        for (int i = 0; i < dim; i++) s->x[i] *= scale;
    }

    build_ple(m, token, s->ple);

    for (int l = 0; l < c->n_layers; l++) {
        int hd = layer_head_dim(c, l);
        int q_dim = layer_q_dim(c, l);
        int kv_dim = layer_kv_dim(c, l);
        int n_rot = layer_n_rot(c, l);
        int swa = is_swa_layer(c, l);
        float rope_base = swa ? c->rope_theta_swa : c->rope_theta;
        const float *freq_factors = swa ? NULL : w->rope_freqs;
        float attn_scale = 1.0f; /* Gemma4 f_attention_scale = 1.0 in llama.cpp */

        float x_before[4096];
        memcpy(x_before, s->x, (size_t)dim * sizeof(float));

        rmsnorm(s->xb2, s->x, w->attn_norm[l], dim, c->norm_eps);

        mm(s->q, s->xb2, w->wq[l], dim, q_dim, w->wq_t[l]);
        rmsnorm_head_inplace(s->q, w->q_norm[l], n_heads, hd, c->norm_eps);
        apply_rope_neox(s->q, n_heads, hd, n_rot, pos, rope_base, 1.0f, freq_factors);

        int kv_src = kv_source_for_layer(c, l);
        if (layer_has_kv(l)) {
            mm(s->k, s->xb2, w->wk[l], dim, kv_dim, w->wk_t[l]);
            mm(s->v, s->xb2, w->wv[l], dim, kv_dim, w->wv_t[l]);
            rmsnorm_head_inplace(s->k, w->k_norm[l], n_kv, hd, c->norm_eps);
            rmsnorm_head_noweight_inplace(s->v, n_kv, hd, c->norm_eps);
            apply_rope_neox(s->k, n_kv, hd, n_rot, pos, rope_base, 1.0f, freq_factors);

            float *kc_pos = kc_at(s, l, pos, max_seq);
            float *vc_pos = vc_at(s, l, pos, max_seq);
            memcpy(kc_pos, s->k, (size_t)kv_dim * sizeof(float));
            memcpy(vc_pos, s->v, (size_t)kv_dim * sizeof(float));
        }

        int kv_src_hd = layer_head_dim(c, kv_src);
        int t_start = 0;
        if (swa) {
            t_start = pos + 1 - c->sliding_window;
            if (t_start < 0) t_start = 0;
        }

        for (int h = 0; h < n_heads; h++) {
            float *qh = s->q + h * hd;
            int kvh = h / kv_mul;
            float *att_h = s->att + (size_t)h * max_seq;

            for (int t = t_start; t <= pos; t++) {
                float *kt = kc_at(s, kv_src, t, max_seq) + kvh * kv_src_hd;
                float score = 0.0f;
                for (int i = 0; i < hd; i++) score += qh[i] * kt[i];
                att_h[t] = score * attn_scale;
            }

            softmax(att_h + t_start, pos + 1 - t_start);
            for (int t = 0; t < t_start; t++) att_h[t] = 0.0f;

            float *oh = s->q_out + h * hd;
            memset(oh, 0, (size_t)hd * sizeof(float));
            for (int t = t_start; t <= pos; t++) {
                float a = att_h[t];
                float *vt = vc_at(s, kv_src, t, max_seq) + kvh * kv_src_hd;
                for (int i = 0; i < hd; i++) oh[i] += a * vt[i];
            }
        }

        mm(s->xb2, s->q_out, w->wo[l], q_dim, dim, w->wo_t[l]);
        rmsnorm(s->xb2, s->xb2, w->post_attn_norm[l], dim, c->norm_eps);
        for (int i = 0; i < dim; i++) s->x[i] = x_before[i] + s->xb2[i];

        memcpy(x_before, s->x, (size_t)dim * sizeof(float));

        rmsnorm(s->xb2, s->x, w->ffn_norm[l], dim, c->norm_eps);
        mm(s->hb,  s->xb2, w->gate[l], dim, hidden, w->gate_t[l]);
        mm(s->hb2, s->xb2, w->up[l],   dim, hidden, w->up_t[l]);
        for (int i = 0; i < hidden; i++)
            s->hb[i] = gelu_f32(s->hb[i]) * s->hb2[i];
        mm(s->xb2, s->hb, w->down[l], hidden, dim, w->down_t[l]);
        rmsnorm(s->xb2, s->xb2, w->post_ffw_norm[l], dim, c->norm_eps);
        for (int i = 0; i < dim; i++) s->x[i] = x_before[i] + s->xb2[i];

        memcpy(x_before, s->x, (size_t)dim * sizeof(float));

        mm(s->ple_gate, s->x, w->inp_gate[l], dim, c->ple_dim, w->inp_gate_t[l]);
        for (int i = 0; i < c->ple_dim; i++) s->ple_gate[i] = gelu_f32(s->ple_gate[i]);
        const float *ple_l = s->ple + (size_t)l * c->ple_dim;
        for (int i = 0; i < c->ple_dim; i++) s->ple_gate[i] *= ple_l[i];
        mm(s->xb2, s->ple_gate, w->proj[l], c->ple_dim, dim, w->proj_t[l]);
        rmsnorm(s->xb2, s->xb2, w->post_norm[l], dim, c->norm_eps);
        for (int i = 0; i < dim; i++) s->x[i] = x_before[i] + s->xb2[i];

        if (w->out_scale[l]) {
            float sc = w->out_scale[l][0];
            for (int i = 0; i < dim; i++) s->x[i] *= sc;
        }
    }

    rmsnorm(s->x, s->x, w->norm_out, dim, c->norm_eps);
    logits_tied(m, s->logits);
}

/* ================================================================
 * Sampling and generation
 * ================================================================ */

static float rng_f32(uint64_t *state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (float)((*state * 0x2545F4914F6CDD1DULL) >> 40) / (float)(1 << 24);
}

typedef struct { float p; int idx; } ProbIdx;

static int cmp_prob_desc(const void *a, const void *b) {
    float pa = ((const ProbIdx *)a)->p;
    float pb = ((const ProbIdx *)b)->p;
    return (pa < pb) - (pa > pb);
}

static int sample_token(float *logits, int n, float temp, float topp, uint64_t *rng) {
    if (temp <= 0.0f) {
        int best = 0;
        for (int i = 1; i < n; i++)
            if (logits[i] > logits[best]) best = i;
        return best;
    }

    for (int i = 0; i < n; i++) logits[i] /= temp;
    softmax(logits, n);

    float coin = rng_f32(rng);
    if (topp <= 0.0f || topp >= 1.0f) {
        float cdf = 0.0f;
        for (int i = 0; i < n; i++) {
            cdf += logits[i];
            if (coin < cdf) return i;
        }
        return n - 1;
    }

    ProbIdx *pi = (ProbIdx *)malloc((size_t)n * sizeof(ProbIdx));
    int np = 0;
    float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (logits[i] >= cutoff) {
            pi[np].p = logits[i];
            pi[np].idx = i;
            np++;
        }
    }
    qsort(pi, np, sizeof(ProbIdx), cmp_prob_desc);

    float cum = 0.0f;
    int last = np - 1;
    for (int i = 0; i < np; i++) {
        cum += pi[i].p;
        if (cum > topp) { last = i; break; }
    }

    float r = rng_f32(rng) * cum;
    float cdf = 0.0f;
    int result = pi[last].idx;
    for (int i = 0; i <= last; i++) {
        cdf += pi[i].p;
        if (r < cdf) { result = pi[i].idx; break; }
    }
    free(pi);
    return result;
}

static int is_special(Tok *tk, int id) {
    return id == tk->bos || id == tk->eos || id == tk->eot ||
           id == tk->turn_start || id == tk->turn_end ||
           id == tk->turn_user || id == tk->turn_model || id == tk->turn_system ||
           id == tk->think || id == tk->channel || id == tk->channel_thought || id == tk->channel_end;
}

typedef enum { OUT_HIDDEN = 0, OUT_THINKING = 1, OUT_ANSWER = 2 } PrintMode;

static void tokenbuf_init(TokenBuf *b) {
    b->tokens = NULL;
    b->n = 0;
    b->cap = 0;
}

static void tokenbuf_free(TokenBuf *b) {
    free(b->tokens);
    b->tokens = NULL;
    b->n = b->cap = 0;
}

static void tokenbuf_push(TokenBuf *b, int id) {
    if (b->n >= b->cap) {
        b->cap = b->cap ? b->cap * 2 : 256;
        b->tokens = (int *)realloc(b->tokens, (size_t)b->cap * sizeof(int));
    }
    b->tokens[b->n++] = id;
}

static char *tokens_to_text(Tok *tk, const int *tokens, int start, int end) {
    size_t cap = 256;
    size_t len = 0;
    char *buf = (char *)malloc(cap);
    buf[0] = 0;

    for (int i = start; i < end; i++) {
        int id = tokens[i];
        if (id < 0 || id >= tk->size || is_special(tk, id)) continue;
        const char *s = tk->vocab[id];
        int slen = tk->vlen[id];
        while (len + (size_t)slen + 1 > cap) {
            cap *= 2;
            buf = (char *)realloc(buf, cap);
        }
        for (int j = 0; j < slen; ) {
            if (j + 3 <= slen && (unsigned char)s[j] == 0xE2 &&
                (unsigned char)s[j + 1] == 0x96 && (unsigned char)s[j + 2] == 0x81) {
                buf[len++] = ' ';
                j += 3;
            } else {
                buf[len++] = s[j++];
            }
        }
    }
    buf[len] = 0;
    return buf;
}

static int is_thought_opener(Tok *tk, int id) {
    return id == tk->channel_thought || id == tk->channel;
}

static void parse_response(Tok *tk, const int *tokens, int n,
                           char **out_thinking, char **out_answer) {
    int thought_start = -1, thought_end = -1;
    for (int i = 0; i < n; i++) {
        if (is_thought_opener(tk, tokens[i]) && thought_start < 0)
            thought_start = i + 1;
        else if (tokens[i] == tk->channel_end)
            thought_end = i;
    }

    if (thought_start >= 0 && thought_end > thought_start) {
        if (out_thinking)
            *out_thinking = tokens_to_text(tk, tokens, thought_start, thought_end);
        if (out_answer)
            *out_answer = tokens_to_text(tk, tokens, thought_end + 1, n);
        return;
    }

    if (out_thinking) *out_thinking = NULL;
    if (out_answer) *out_answer = tokens_to_text(tk, tokens, 0, n);
}

static void print_tok(Tok *tk, int id) {
    if (id < 0 || id >= tk->size || is_special(tk, id)) return;
    const char *s = tk->vocab[id];
    int len = tk->vlen[id];
    for (int i = 0; i < len; ) {
        if (i + 3 <= len && (unsigned char)s[i] == 0xE2 &&
            (unsigned char)s[i + 1] == 0x96 && (unsigned char)s[i + 2] == 0x81) {
            putchar(' ');
            i += 3;
        } else {
            putchar(s[i]);
            i++;
        }
    }
    fflush(stdout);
}

static void print_tok_mode(Tok *tk, int id, PrintMode mode) {
    if (mode == OUT_HIDDEN) return;
    print_tok(tk, id);
}

#define PREFILL_BAR_WIDTH 40

static void prefill_progress_update(int done, int total) {
    if (total <= 0) return;
    if (done > total) done = total;
    int filled = (done * PREFILL_BAR_WIDTH) / total;
    int pct = (done * 100) / total;
    fprintf(stderr, "\rPrefill [");
    for (int i = 0; i < PREFILL_BAR_WIDTH; i++)
        fputc(i < filled ? '=' : ' ', stderr);
    fprintf(stderr, "] %3d%% (%d/%d)", pct, done, total);
    fflush(stderr);
}

static void prefill_progress_done(int n_tokens, double elapsed_sec) {
    double tps = (elapsed_sec > 0.0) ? (double)n_tokens / elapsed_sec : 0.0;
    fprintf(stderr, "\rPrefill [");
    for (int i = 0; i < PREFILL_BAR_WIDTH; i++)
        fputc('=', stderr);
    fprintf(stderr, "] 100%% (%d/%d)\n", n_tokens, n_tokens);
    fprintf(stderr, "Prefill complete: %d tokens in %.2fs (%.2f tok/s)\n",
            n_tokens, elapsed_sec, tps);
}

static void decode_progress_done(int n_tokens, double elapsed_sec) {
    double tps = (elapsed_sec > 0.0) ? (double)n_tokens / elapsed_sec : 0.0;
    fflush(stdout);
    fputc('\n', stdout);
    fflush(stdout);
    fprintf(stderr, "\nDecode complete: %d tokens in %.2fs (%.2f tok/s)\n",
            n_tokens, elapsed_sec, tps);
}

static void throughput_summary(int n_prefill, double prefill_sec,
                               int n_decode, double decode_sec,
                               double total_sec) {
    double prefill_tps = (prefill_sec > 0.0) ? (double)n_prefill / prefill_sec : 0.0;
    double decode_tps  = (decode_sec > 0.0)  ? (double)n_decode / decode_sec  : 0.0;
    int n_total = n_prefill + n_decode;
    double total_tps = (total_sec > 0.0) ? (double)n_total / total_sec : 0.0;
    fprintf(stderr, "--- throughput ---\n");
    fprintf(stderr, "  prefill: %.2f tok/s\n", prefill_tps);
    fprintf(stderr, "  decode:  %.2f tok/s\n", decode_tps);
    fprintf(stderr, "  total:   %.2f tok/s\n", total_tps);
}

static int generate(Model *m, int *prompt, int n_prompt,
                     int max_new, float temp, float topp, uint64_t seed,
                     int enable_thinking, int show_thinking, TokenBuf *gen_out) {
    uint64_t rng = seed ? seed : 1;
    int token = prompt[0];
    int gen = 0;
    int prefill_reported = 0;
    int decode_timing = 0;
    double prefill_sec = 0.0;
    PrintMode pmode = enable_thinking ? OUT_HIDDEN : OUT_ANSWER;
    int saw_thought = 0;

    if (gen_out) gen_out->n = 0;

    struct timespec t0, t1, t_prefill, t_decode;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    clock_gettime(CLOCK_MONOTONIC, &t_prefill);
    if (n_prompt > 0)
        prefill_progress_update(0, n_prompt);

    for (int pos = 0; pos < n_prompt + max_new - 1; pos++) {
        if (pos >= m->cfg.max_seq) {
            fprintf(stderr, "\n[max sequence length %d reached]\n", m->cfg.max_seq);
            break;
        }

        forward(m, token, pos);

        if (n_prompt > 0 && pos < n_prompt) {
            prefill_progress_update(pos + 1, n_prompt);
            if (pos == n_prompt - 1 && !prefill_reported) {
                struct timespec t_now;
                clock_gettime(CLOCK_MONOTONIC, &t_now);
                prefill_sec = (t_now.tv_sec - t_prefill.tv_sec)
                    + (t_now.tv_nsec - t_prefill.tv_nsec) / 1e9;
                prefill_progress_done(n_prompt, prefill_sec);
                prefill_reported = 1;
                clock_gettime(CLOCK_MONOTONIC, &t_decode);
                decode_timing = 1;
            }
        }

        int next;
        if (pos < n_prompt - 1) {
            next = prompt[pos + 1];
        } else {
            if (!decode_timing) {
                clock_gettime(CLOCK_MONOTONIC, &t_decode);
                decode_timing = 1;
            }
            next = sample_token(m->s.logits, m->cfg.vocab_size, temp, topp, &rng);
            if (next == m->tok.eos || next == m->tok.eot) break;

            if (enable_thinking) {
                if (is_thought_opener(&m->tok, next)) {
                    saw_thought = 1;
                    pmode = show_thinking ? OUT_THINKING : OUT_HIDDEN;
                    if (show_thinking) fputs("\n--- Thinking ---\n", stderr);
                } else if (next == m->tok.channel_end) {
                    pmode = OUT_ANSWER;
                    if (show_thinking && saw_thought)
                        fputs("\n--- Answer ---\n", stderr);
                }
            }

            gen++;
            if (gen_out) tokenbuf_push(gen_out, next);
            print_tok_mode(&m->tok, next, pmode);
        }
        token = next;
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    double decode_sec = 0.0;
    if (decode_timing) {
        decode_sec = (t1.tv_sec - t_decode.tv_sec)
            + (t1.tv_nsec - t_decode.tv_nsec) / 1e9;
        decode_progress_done(gen, decode_sec);
    }

    printf("\n\n--- %d prompt tokens + %d generated tokens ---\n", n_prompt, gen);
    printf("--- %.1fs total ---\n", elapsed);
    if (prefill_reported || decode_timing)
        throughput_summary(n_prompt, prefill_sec, gen, decode_sec, elapsed);

    return gen;
}

static char *read_user_line(const char *label) {
    fprintf(stderr, "%s", label);
    fflush(stderr);
    char *line = NULL;
    size_t cap = 0;
    ssize_t n = getline(&line, &cap, stdin);
    if (n < 0) {
        free(line);
        return NULL;
    }
    while (n > 0 && (line[n - 1] == '\n' || line[n - 1] == '\r')) line[--n] = 0;
    if (n == 0 || !strcmp(line, "/quit") || !strcmp(line, "/exit")) {
        free(line);
        return NULL;
    }
    return line;
}

static int run_chat_turn(Model *m, ChatHistory *hist, int enable_thinking, int show_thinking,
                         int max_new, float temp, float topp, uint64_t seed) {
    int *prompt = (int *)malloc(MAX_PROMPT_TOKS * sizeof(int));
    int n_prompt = chat_encode_history(&m->tok, hist, enable_thinking, prompt, MAX_PROMPT_TOKS);
    if (n_prompt < 0) {
        fprintf(stderr, "Error: conversation too long (max %d tokens)\n", MAX_PROMPT_TOKS);
        free(prompt);
        return -1;
    }

    fputs("Model: ", stdout);
    fflush(stdout);

    TokenBuf gen;
    tokenbuf_init(&gen);
    generate(m, prompt, n_prompt, max_new, temp, topp, seed,
             enable_thinking, show_thinking, &gen);
    free(prompt);

    char *thinking = NULL, *answer = NULL;
    parse_response(&m->tok, gen.tokens, gen.n, &thinking, &answer);

    if (enable_thinking && show_thinking && thinking && thinking[0]) {
        fprintf(stderr, "\n[thinking stripped from history: %d chars]\n", (int)strlen(thinking));
    }

    if (!answer || !answer[0]) {
        free(thinking);
        free(answer);
        tokenbuf_free(&gen);
        fprintf(stderr, "Warning: empty model response\n");
        return 0;
    }

    chat_history_push(hist, CHAT_ASSISTANT, answer);
    free(thinking);
    free(answer);
    tokenbuf_free(&gen);
    return 0;
}

static void run_single_shot(Model *m, const char *prompt, int enable_thinking, int show_thinking,
                            int max_new, float temp, float topp, uint64_t seed) {
    int n_prompt_tokens;
    int *prompt_tokens = chat_encode(&m->tok, prompt, enable_thinking, &n_prompt_tokens);
    printf("Prompt: \"%s\" (%d tokens)%s\n\n", prompt, n_prompt_tokens,
           enable_thinking ? " [thinking]" : "");

    TokenBuf gen;
    tokenbuf_init(&gen);
    generate(m, prompt_tokens, n_prompt_tokens, max_new, temp, topp, seed,
             enable_thinking, show_thinking, &gen);
    free(prompt_tokens);
    tokenbuf_free(&gen);
}

static void run_interactive(Model *m, const char *initial_prompt, int enable_thinking,
                            int show_thinking, int max_new, float temp, float topp,
                            uint64_t seed) {
    ChatHistory hist;
    chat_history_init(&hist);

    fprintf(stderr, "Interactive mode ( /quit to exit )\n");
    if (enable_thinking) {
        fprintf(stderr, "Thinking mode: ON%s\n",
                show_thinking ? " (showing trace)" : " (answer only)");
    }

    char *line = NULL;
    if (initial_prompt && initial_prompt[0]) {
        line = strdup(initial_prompt);
    }

    for (;;) {
        if (!line) {
            line = read_user_line("You: ");
            if (!line) break;
        }

        chat_history_push(&hist, CHAT_USER, line);
        free(line);
        line = NULL;

        if (run_chat_turn(m, &hist, enable_thinking, show_thinking,
                          max_new, temp, topp, seed) < 0)
            break;
    }

    chat_history_free(&hist);
    fprintf(stderr, "\nBye.\n");
}

int main(int argc, char *argv[]) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf> [options]\n", argv[0]);
        fprintf(stderr, "Options:\n");
        fprintf(stderr, "  -p <prompt>       User prompt (default: Hello)\n");
        fprintf(stderr, "  -n <tokens>       Max tokens to generate per turn (default: 256)\n");
        fprintf(stderr, "  -t <temp>         Temperature (default: 0.6)\n");
        fprintf(stderr, "  -k <topp>         Top-p sampling (default: 0.9)\n");
        fprintf(stderr, "  -s <seed>         Random seed (default: time)\n");
        fprintf(stderr, "  -l <len>          Max sequence length (default: 8192)\n");
        fprintf(stderr, "  -i                Interactive multi-turn mode\n");
        fprintf(stderr, "  --think           Enable thinking mode\n");
        fprintf(stderr, "  --show-thinking   Show reasoning trace (requires --think)\n");
        return 1;
    }

    char *model_path = argv[1];
    char *prompt     = "Hello, how are you?";
    int   max_tokens = 256;
    float temp       = 0.6f;
    float topp       = 0.9f;
    uint64_t seed    = (uint64_t)time(NULL);
    int   max_seq    = 8192;
    int   interactive = 0;
    int   enable_thinking = 0;
    int   show_thinking = 0;

    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "-i") || !strcmp(argv[i], "--interactive")) {
            interactive = 1;
            continue;
        }
        if (!strcmp(argv[i], "--think")) {
            enable_thinking = 1;
            continue;
        }
        if (!strcmp(argv[i], "--show-thinking")) {
            enable_thinking = 1;
            show_thinking = 1;
            continue;
        }
        if (i + 1 >= argc) {
            fprintf(stderr, "Error: option %s requires a value\n", argv[i]);
            return 1;
        }
        if      (!strcmp(argv[i], "-p")) prompt     = argv[i + 1];
        else if (!strcmp(argv[i], "-n")) max_tokens = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "-t")) temp       = (float)atof(argv[i + 1]);
        else if (!strcmp(argv[i], "-k")) topp       = (float)atof(argv[i + 1]);
        else if (!strcmp(argv[i], "-s")) seed       = (uint64_t)strtoull(argv[i + 1], NULL, 10);
        else if (!strcmp(argv[i], "-l")) max_seq    = atoi(argv[i + 1]);
        else {
            fprintf(stderr, "Error: unknown option %s\n", argv[i]);
            return 1;
        }
        i++;
    }

    printf("Loading %s ...\n", model_path);

    Model model;
    memset(&model, 0, sizeof(model));

    model.fd = open(model_path, O_RDONLY);
    if (model.fd < 0) {
        fprintf(stderr, "Error: cannot open %s\n", model_path);
        return 1;
    }
    struct stat st;
    fstat(model.fd, &st);
    model.fsz = (size_t)st.st_size;
    model.fdata = (uint8_t *)mmap(NULL, model.fsz, PROT_READ, MAP_PRIVATE, model.fd, 0);
    if (model.fdata == MAP_FAILED) {
        fprintf(stderr, "Error: mmap failed\n");
        return 1;
    }

    char **merges = NULL;
    int n_merges = 0;
    parse_gguf(&model, &merges, &n_merges);

    model.cfg.max_seq = max_seq;

    Config *c = &model.cfg;
    printf("Model: dim=%d hidden=%d layers=%d heads=%d kv_heads=%d vocab=%d\n",
           c->dim, c->hidden_dim, c->n_layers, c->n_heads, c->n_kv_heads, c->vocab_size);
    printf("       swa_hd=%d full_hd=%d ple=%d swa_win=%d shared_kv=%d softcap=%.0f max_seq=%d\n",
           c->head_dim_swa, c->head_dim_full, c->ple_dim, c->sliding_window,
           c->shared_kv_layers, c->logit_softcapping, c->max_seq);

    load_weights(&model);
    init_tokenizer(&model.tok, merges, n_merges);
    alloc_state(&model.s, c);

    if (interactive) {
        run_interactive(&model, prompt, enable_thinking, show_thinking,
                        max_tokens, temp, topp, seed);
    } else {
        run_single_shot(&model, prompt, enable_thinking, show_thinking,
                        max_tokens, temp, topp, seed);
    }
    free_state(&model.s);
    free_weight_ptrs(&model.w, c->n_layers);
    free(c->swa_layers);
    for (int i = 0; i < model.nti; i++) free(model.ti[i].name);
    free(model.ti);
    for (int i = 0; i < model.tok.size; i++) free(model.tok.vocab[i]);
    free(model.tok.vocab);
    free(model.tok.vlen);
    free(model.tok.scores);
    free(model.tok.htab);
    munmap(model.fdata, model.fsz);
    close(model.fd);

    return 0;
}
