#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cmath>

static void usage(const char *argv0) {
    fprintf(stderr,
        "Usage: %s -m model.gguf (-f prompt.txt | --tokens ids.txt)\n"
        "       [-n N] [-d step] [-k topk] [--force gen_ids.txt]\n", argv0);
}

static void dump_topk(const float *logits, int n_vocab, int k) {
    std::vector<int> idx((size_t)n_vocab);
    for (int i = 0; i < n_vocab; i++) idx[(size_t)i] = i;
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
        [&](int a, int b) { return logits[a] > logits[b]; });
    if (k > n_vocab) k = n_vocab;
    fprintf(stderr, "  top-%d logits:", k);
    for (int i = 0; i < k; i++)
        fprintf(stderr, " %d=%.4f", idx[(size_t)i], logits[idx[(size_t)i]]);
    fputc('\n', stderr);
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
    const char *prompt_path = nullptr;
    const char *force_path = nullptr;
    const char *tokens_path = nullptr;
    int n_predict = 50;
    int dump_at = -1;
    int topk = 8;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) model_path = argv[++i];
        else if (!strcmp(argv[i], "-f") && i + 1 < argc) prompt_path = argv[++i];
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) n_predict = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-d") && i + 1 < argc) dump_at = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-k") && i + 1 < argc) topk = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--force") && i + 1 < argc) force_path = argv[++i];
        else if (!strcmp(argv[i], "--tokens") && i + 1 < argc) tokens_path = argv[++i];
        else if (!strcmp(argv[i], "-h")) { usage(argv[0]); return 0; }
        else { fprintf(stderr, "Unknown arg: %s\n", argv[i]); usage(argv[0]); return 1; }
    }
    if (!model_path || (!prompt_path && !tokens_path)) { usage(argv[0]); return 1; }

    std::vector<llama_token> prompt_tokens;
    if (tokens_path) {
        for (int v : read_int_file(tokens_path))
            prompt_tokens.push_back((llama_token)v);
    } else {
        FILE *fp = fopen(prompt_path, "rb");
        if (!fp) { perror(prompt_path); return 1; }
        fseek(fp, 0, SEEK_END);
        long sz = ftell(fp);
        fseek(fp, 0, SEEK_SET);
        std::vector<char> prompt((size_t)sz + 1);
        if (fread(prompt.data(), 1, (size_t)sz, fp) != (size_t)sz) {
            fprintf(stderr, "read failed\n"); return 1;
        }
        prompt[(size_t)sz] = 0;
        fclose(fp);

        ggml_backend_load_all();
        llama_model *model_tmp = llama_model_load_from_file(model_path, llama_model_default_params());
        if (!model_tmp) { fprintf(stderr, "load failed\n"); return 1; }
        const llama_vocab *vocab = llama_model_get_vocab(model_tmp);
        const int n_prompt = -llama_tokenize(vocab, prompt.data(), (int32_t)sz, NULL, 0, true, true);
        prompt_tokens.resize((size_t)n_prompt);
        if (llama_tokenize(vocab, prompt.data(), (int32_t)sz, prompt_tokens.data(), n_prompt, true, true) < 0) {
            fprintf(stderr, "tokenize failed\n"); return 1;
        }
        llama_model_free(model_tmp);
    }

    std::vector<int> force;
    if (force_path) force = read_int_file(force_path);

    ggml_backend_load_all();

    llama_model *model = llama_model_load_from_file(model_path, llama_model_default_params());
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }

    const llama_vocab *vocab = llama_model_get_vocab(model);
    const int n_prompt = (int)prompt_tokens.size();

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = (uint32_t)(n_prompt + n_predict + 8);
    cparams.n_batch = (uint32_t)n_prompt;
    cparams.embeddings = true;
    llama_context *ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "ctx failed\n"); return 1; }

    const int n_vocab = llama_vocab_n_tokens(vocab);

    auto sparams = llama_sampler_chain_default_params();
    llama_sampler *smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    if (llama_decode(ctx, llama_batch_get_one(prompt_tokens.data(), n_prompt))) {
        fprintf(stderr, "prefill failed\n"); return 1;
    }

    llama_token tok = 0;
    for (int i = 0; i < n_predict; i++) {
        if (dump_at == i) {
            const float *logits = llama_get_logits_ith(ctx, -1);
            fprintf(stderr, "[llama logits at gen step %d, pos %d, n_prompt=%d]\n",
                    i, n_prompt + i, n_prompt);
            dump_topk(logits, n_vocab, topk);
            float *emb = llama_get_embeddings_ith(ctx, -1);
            if (emb) {
                const int n_embd = llama_model_n_embd(model);
                double sum = 0, l2 = 0;
                float amax = 0;
                for (int j = 0; j < n_embd; j++) {
                    float x = emb[j];
                    sum += x; l2 += (double)x * x;
                    float ax = x >= 0 ? x : -x;
                    if (ax > amax) amax = ax;
                }
                fprintf(stderr, "  hidden: sum=%.4f l2=%.4f max=%.4f\n",
                        (float)sum, (float)sqrt(l2), amax);
            }
        }
        if ((int)force.size() > i) tok = (llama_token)force[i];
        else tok = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, tok)) break;
        llama_batch batch = llama_batch_get_one(&tok, 1);
        batch.logits[0] = 1;
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "decode failed at %d\n", i);
            break;
        }
    }

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
