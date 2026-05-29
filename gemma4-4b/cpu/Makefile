CC      ?= cc
CFLAGS  ?= -O3 -std=c11 -Wall -Wextra -Wno-unused-parameter
LDFLAGS ?= -lm

MODEL   ?= ../gemma-4-E4B-it-Q4_K_M.gguf
PROMPT  ?= Hello, how are you?

.PHONY: build run clean

build:
	$(CC) $(CFLAGS) -o gemma4-cpu main.c $(LDFLAGS)

run: build
	./gemma4-cpu "$(MODEL)" -p "$(PROMPT)"

clean:
	rm -f gemma4-cpu
