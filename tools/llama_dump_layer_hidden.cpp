#include "llama.h"
#include "ggml.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

struct DumpCtx {
    int  target_pos = -1;
    bool active     = false;
};

static void hidden_stats(const float *v, int n, float *out_sum, float *out_l2, float *out_max) {
    double sum = 0.0, l2 = 0.0;
    float amax = 0.0f;
    for (int i = 0; i < n; i++) {
        float x = v[i];
        sum += x;
        l2 += (double)x * x;
        float ax = x >= 0.0f ? x : -x;
        if (ax > amax) amax = ax;
    }
    *out_sum = (float)sum;
    *out_l2 = (float)sqrt(l2);
    *out_max = amax;
}

static void dump_tensor_row(const ggml_tensor *t, int row, int pos, const char *label) {
    if (!t || !t->data || t->type != GGML_TYPE_F32) return;

    const int n0 = (int)t->ne[0];
    const int n1 = (int)t->ne[1];
    if (row < 0 || row >= n1) return;

    const float *base = (const float *)t->data;
    const float *v = base + (size_t)row * (size_t)n0;

    float sum, l2, amax;
    hidden_stats(v, n0, &sum, &l2, &amax);

    if (label && strncmp(label, "l_out-", 6) == 0) {
        int layer = atoi(label + 6);
        fprintf(stderr, "[llama hidden pos %d] layer %2d sum=%.4f l2=%.4f max=%.4f\n",
                pos, layer, sum, l2, amax);
    } else if (label && strstr(label, "attn_out")) {
        fprintf(stderr, "[llama hidden pos %d] %s sum=%.4f l2=%.4f max=%.4f\n",
                pos, label, sum, l2, amax);
    } else if (label && !strcmp(label, "after_emb")) {
        fprintf(stderr, "[llama hidden pos %d] after_emb sum=%.4f l2=%.4f max=%.4f\n",
                pos, sum, l2, amax);
    } else if (label) {
        fprintf(stderr, "[llama hidden pos %d] %s sum=%.4f l2=%.4f max=%.4f\n",
                pos, label, sum, l2, amax);
    }
}

static void dump_kv_tensor3d(const ggml_tensor *t, int pos, int layer, const char *tag) {
    if (!t || !t->data || t->type != GGML_TYPE_F32) return;
    const int64_t ne0 = t->ne[0];
    const int64_t ne1 = t->ne[1];
    const int64_t ne2 = t->ne[2];
    const int nflat = (int)(ne0 * ne1);
    const int tok = (int)ne2 - 1;
    if (tok < 0) return;
    const float *v = (const float *)t->data + (size_t)tok * (size_t)nflat;
    float sum, l2, amax;
    hidden_stats(v, nflat, &sum, &l2, &amax);
    fprintf(stderr, "[llama hidden pos %d] layer %2d %s sum=%.4f l2=%.4f max=%.4f",
            pos, layer, tag, sum, l2, amax);
    for (int i = 0; i < 4 && i < nflat; i++) fprintf(stderr, " v%d=%.6f", i, v[i]);
    fputc('\n', stderr);
}

static bool layer_dump_cb(ggml_tensor *t, bool ask, void *user_data) {
    auto *ctx = (DumpCtx *)user_data;
    if (!ctx->active) return true;
    if (ask) return true;
    if (!t->name || !t->data) return true;

    if (!strcmp(t->name, "inp_scaled")) {
        dump_tensor_row(t, (int)t->ne[1] - 1, ctx->target_pos, "after_emb");
        return true;
    }
    if (strncmp(t->name, "l_out-", 6) == 0) {
        dump_tensor_row(t, (int)t->ne[1] - 1, ctx->target_pos, t->name);
        return true;
    }
    if (strncmp(t->name, "attn_out-", 9) == 0) {
        char label[32];
        int layer = atoi(t->name + 9);
        snprintf(label, sizeof(label), "layer %2d attn_out", layer);
        dump_tensor_row(t, (int)t->ne[1] - 1, ctx->target_pos, label);
        return true;
    }
    if (!strcmp(t->name, "attn_norm-0")) {
        dump_tensor_row(t, (int)t->ne[1] - 1, ctx->target_pos, "layer  0 attn_norm");
        return true;
    }
    if (strncmp(t->name, "Kcur_normed-", 12) == 0) {
        int layer = atoi(t->name + 12);
        dump_kv_tensor3d(t, ctx->target_pos, layer, "K_pre_rope");
        return true;
    }
    if (strncmp(t->name, "Kcur_pos-", 9) == 0) {
        int layer = atoi(t->name + 9);
        dump_kv_tensor3d(t, ctx->target_pos, layer, "K_store");
        return true;
    }
    if (strncmp(t->name, "Kcur-", 5) == 0) {
        int layer = atoi(t->name + 5);
        dump_kv_tensor3d(t, ctx->target_pos, layer, "K_pre_mm");
        return true;
    }
    if (strncmp(t->name, "kqv_out-", 8) == 0) {
        char label[32];
        int layer = atoi(t->name + 8);
        snprintf(label, sizeof(label), "layer %2d q_out", layer);
        dump_tensor_row(t, (int)t->ne[1] - 1, ctx->target_pos, label);
    }
    return true;
}

static void usage(const char *argv0) {
    fprintf(stderr,
        "Usage: %s -m model.gguf --tokens prompt_ids.txt [--force gen_ids.txt]\n"
        "       --dump-at-pos N   (0-based sequence position)\n", argv0);
}

static std::vector<int> read_int_file(const char *path) {
    std::vector<int> out;
    FILE *f = fopen(path, "r");
    if (!f) { perror(path); exit(1); }
    int v;
    while (fscanf(f, "%d", &v) == 1) out.push_back(v);
    fclose(f);
    return out;
}

int main(int argc, char **argv) {
    const char *model_path = nullptr;
    const char *tokens_path = nullptr;
    const char *force_path = nullptr;
    int dump_at_pos = -1;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) model_path = argv[++i];
        else if (!strcmp(argv[i], "--tokens") && i + 1 < argc) tokens_path = argv[++i];
        else if (!strcmp(argv[i], "--force") && i + 1 < argc) force_path = argv[++i];
        else if (!strcmp(argv[i], "--dump-at-pos") && i + 1 < argc) dump_at_pos = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-h")) { usage(argv[0]); return 0; }
        else { fprintf(stderr, "Unknown arg: %s\n", argv[i]); usage(argv[0]); return 1; }
    }
    if (!model_path || !tokens_path || dump_at_pos < 0) { usage(argv[0]); return 1; }

    std::vector<llama_token> prompt;
    for (int v : read_int_file(tokens_path))
        prompt.push_back((llama_token)v);

    std::vector<llama_token> force;
    if (force_path) {
        for (int v : read_int_file(force_path))
            force.push_back((llama_token)v);
    }

    const int n_prompt = (int)prompt.size();
    if (dump_at_pos >= n_prompt + (int)force.size()) {
        fprintf(stderr, "dump-at-pos %d out of range (prompt=%d force=%zu)\n",
                dump_at_pos, n_prompt, force.size());
        return 1;
    }

    ggml_backend_load_all();

    llama_model *model = llama_model_load_from_file(model_path, llama_model_default_params());
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }

    DumpCtx dump_ctx;
    dump_ctx.target_pos = dump_at_pos;

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = (uint32_t)(n_prompt + (int)force.size() + 8);
    cparams.n_batch = 1;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.type_k = GGML_TYPE_F32;
    cparams.type_v = GGML_TYPE_F32;
    cparams.cb_eval = layer_dump_cb;
    cparams.cb_eval_user_data = &dump_ctx;

    llama_context *ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "ctx failed\n"); return 1; }

    std::vector<llama_token> seq;
    seq.insert(seq.end(), prompt.begin(), prompt.end());
    seq.insert(seq.end(), force.begin(), force.end());
    if (dump_at_pos >= (int)seq.size()) {
        fprintf(stderr, "dump-at-pos %d out of range (seq=%zu)\n", dump_at_pos, seq.size());
        return 1;
    }

    for (int p = 0; p <= dump_at_pos; p++) {
        dump_ctx.active = (p == dump_at_pos);
        llama_token tok = seq[(size_t)p];
        if (llama_decode(ctx, llama_batch_get_one(&tok, 1))) {
            fprintf(stderr, "decode failed at pos %d\n", p);
            return 1;
        }
        dump_ctx.active = false;
    }

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
