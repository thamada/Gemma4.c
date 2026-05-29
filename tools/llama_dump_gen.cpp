#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void usage(const char *argv0) {
    fprintf(stderr, "Usage: %s -m model.gguf [-n N] [-f prompt.txt]\n", argv0);
}

int main(int argc, char **argv) {
    const char *model_path = nullptr;
    const char *prompt_path = nullptr;
    int n_predict = 50;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) {
            model_path = argv[++i];
        } else if (!strcmp(argv[i], "-n") && i + 1 < argc) {
            n_predict = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-f") && i + 1 < argc) {
            prompt_path = argv[++i];
        } else if (!strcmp(argv[i], "-h")) {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown arg: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }
    if (!model_path || !prompt_path) {
        usage(argv[0]);
        return 1;
    }

    FILE *fp = fopen(prompt_path, "rb");
    if (!fp) {
        perror(prompt_path);
        return 1;
    }
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    std::vector<char> prompt((size_t)sz + 1);
    if (fread(prompt.data(), 1, (size_t)sz, fp) != (size_t)sz) {
        fprintf(stderr, "read failed\n");
        return 1;
    }
    prompt[(size_t)sz] = 0;
    fclose(fp);

    ggml_backend_load_all();

    llama_model_params mparams = llama_model_default_params();
    llama_model *model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "load failed\n");
        return 1;
    }

    const llama_vocab *vocab = llama_model_get_vocab(model);
    const int n_prompt = -llama_tokenize(vocab, prompt.data(), (int32_t)sz, NULL, 0, true, true);
    std::vector<llama_token> prompt_tokens((size_t)n_prompt);
    if (llama_tokenize(vocab, prompt.data(), (int32_t)sz, prompt_tokens.data(), n_prompt, true, true) < 0) {
        fprintf(stderr, "tokenize failed\n");
        return 1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = (uint32_t)(n_prompt + n_predict + 8);
    cparams.n_batch = (uint32_t)n_prompt;
    llama_context *ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "ctx failed\n");
        return 1;
    }

    auto sparams = llama_sampler_chain_default_params();
    llama_sampler *smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), n_prompt);
    if (llama_decode(ctx, batch)) {
        fprintf(stderr, "prefill failed\n");
        return 1;
    }

    printf("Prompt token IDs:");
    for (llama_token t : prompt_tokens) printf(" %d", (int)t);
    printf("\nGenerated token IDs:");

    llama_token tok = 0;
    for (int i = 0; i < n_predict; i++) {
        tok = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, tok)) break;
        printf(" %d", (int)tok);
        batch = llama_batch_get_one(&tok, 1);
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "\ndecode failed at %d\n", i);
            break;
        }
    }
    printf("\n");

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
