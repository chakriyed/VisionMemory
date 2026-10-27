"""Times each model on the same question, with a warm-up call first (so model-load
time isn't counted against it) — closer to steady-state latency in a real voice loop."""
import time

import memory_llm

QUESTION = "Hey roommate, where is my phone?"
MODELS = ["qwen2.5:7b", "qwen2.5:1.5b"]


def main() -> None:
    for model in MODELS:
        memory_llm.MODEL = model
        memory_llm.ask(QUESTION, mode="hybrid")  # warm-up, discarded

        times = []
        answer = None
        for _ in range(3):
            start = time.perf_counter()
            answer = memory_llm.ask(QUESTION, mode="hybrid")
            times.append(time.perf_counter() - start)

        avg = sum(times) / len(times)
        print(f"{model:>14}: avg {avg:5.2f}s  (runs: {', '.join(f'{t:.2f}s' for t in times)}) -> {answer}")


if __name__ == "__main__":
    main()
