#!/usr/bin/env python3
# Needle-in-a-haystack long-context retrieval test.
# Adapted from vllm-project/vllm issue #47042 (GLM-5.2 sparse-MLA decode collapse),
# generalized to run against any OpenAI-compatible endpoint / model.
#
# Env:
#   NIAH_URL     endpoint (default http://127.0.0.1:30000/v1/chat/completions)
#   NIAH_MODEL   model name/tag the server serves (required — the served path)
#   NIAH_WORDS   comma list of context sizes in words (default 2000,8000,20000,35000)
#   NIAH_MAXTOK  max_tokens for the answer (default 2048)
#   NIAH_SEEDS   comma list of needle-layout seeds (default 0,1,2); summary reports
#                mean/min/max across seeds to separate real accuracy from variance
#   NIAH_TIMEOUT per-request timeout seconds (default 1800)
#   NIAH_WARMUP  1 (default) = for each length, one throwaway request THEN the scored
#                request (warmup N → score N). Do not warm 16k–35k before scoring 2k:
#                a later EngineCore 500 makes earlier lengths look like found=0/10.
#                Set 0 to disable warmup. NIAH_HALT_ON_FAIL=1 stops the ladder after a
#                warmup/score timeout so a dead engine does not keep printing 0/10.
#   NIAH_PAIR_SLEEP_S          seconds after finishing a length before the next pair
#                              (default 30). 0 disables.
#   NIAH_WARMUP_SCORE_SLEEP_S  seconds between warmup ok and the scored request
#                              (default 5). 0 disables.
import os, sys, json, random, time, urllib.request

URL = os.environ.get("NIAH_URL", "http://127.0.0.1:30000/v1/chat/completions")
MODEL = os.environ.get("NIAH_MODEL", "")
WORDS = [int(x) for x in os.environ.get("NIAH_WORDS", "2000,8000,20000,35000").split(",") if x.strip()]
MAXTOK = int(os.environ.get("NIAH_MAXTOK", "2048"))
TIMEOUT = float(os.environ.get("NIAH_TIMEOUT", "1800"))
# Needle layout is seeded, so a single run is deterministic (bit-exact repro on the
# same stack). Run multiple seeds to distinguish real accuracy from single-needle
# variance; the summary reports mean/min/max across seeds. Default 0,1,2.
SEEDS = [int(x) for x in os.environ.get("NIAH_SEEDS", "0,1,2").split(",") if x.strip()]
WARMUP = os.environ.get("NIAH_WARMUP", "1") == "1"
HALT_ON_FAIL = os.environ.get("NIAH_HALT_ON_FAIL", "1") == "1"
PAIR_SLEEP = float(os.environ.get("NIAH_PAIR_SLEEP_S", "30"))
WARMUP_SCORE_SLEEP = float(os.environ.get("NIAH_WARMUP_SCORE_SLEEP_S", "5"))
# Warmup uses a generous timeout (cold compile of a long-context shape can take minutes).
WARMUP_TIMEOUT = max(TIMEOUT, 1800.0)

FILLER = (
    "table chair window bottle pencil garden river mountain coffee planet "
    "engine guitar pillow ticket basket candle market silver button orange "
    "rocket napkin ladder pepper carpet helmet jacket mirror anchor pocket "
    "branch copper saddle tunnel violin wallet zipper meadow cactus pebble"
).split()
ANIMALS = ["elephant", "giraffe", "kangaroo", "penguin", "dolphin",
           "tiger", "rhinoceros", "octopus", "crocodile", "panda"]

SYSTEM = (
    "You read a word list and pick out the animals. Reply with a single "
    "comma-separated list of lowercase animal names. Output nothing else."
)


def make_haystack(n_words, seed=0):
    rng = random.Random(seed)
    words = [rng.choice(FILLER) for _ in range(n_words)]
    step = max(n_words // (len(ANIMALS) + 1), 1)
    for i, animal in enumerate(ANIMALS):
        words[min((i + 1) * step, len(words) - 1)] = animal
    return " ".join(words)


def _request(n_words, seed, max_tokens, timeout):
    """POST one NIAH request; return (message_dict, error_str). Exactly one is non-None."""
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Find the animals in this list:\n\n" + make_haystack(n_words, seed)},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        # Thinking models (e.g. GLM-5.1) emit chain-of-thought into a separate
        # reasoning field and leave `content` empty until the final answer; with a
        # small max_tokens the answer never appears in `content` and the score is a
        # false 0/10. Disable thinking so the answer lands in `content` directly.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(URL, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())["choices"][0]["message"], None
    except Exception as e:
        return None, str(e)


def warmup(n_words):
    """Throwaway request for this length only. Returns True on HTTP success."""
    _, err = _request(n_words, seed=0, max_tokens=8, timeout=WARMUP_TIMEOUT)
    status = "ok" if err is None else ("timeout/err: %s" % err)
    print("words=%6d  [warmup] %s" % (n_words, status), flush=True)
    return err is None


def run(n_words, seed=0):
    # Sentinel: None = timeout/transport error (NOT a wrong answer); int = score 0..10.
    msg, err = _request(n_words, seed, MAXTOK, TIMEOUT)
    if err is not None:
        print("words=%6d  seed=%d  TIMEOUT/ERROR  %s" % (n_words, seed, err), flush=True)
        return None
    # Score content plus any reasoning field (some servers surface CoT as
    # `reasoning` or `reasoning_content`) so a thinking model is never mis-scored.
    text = ((msg.get("content") or "") + " "
            + (msg.get("reasoning_content") or "") + " "
            + (msg.get("reasoning") or "")).lower()
    found = sorted(a for a in ANIMALS if a in text)
    print("words=%6d  seed=%d  found=%2d/10  %s" % (n_words, seed, len(found), found), flush=True)
    return len(found)


def main():
    if not MODEL:
        print("NIAH_MODEL must be set (the served model path/name)", file=sys.stderr)
        sys.exit(2)
    print("=== NIAH retrieval test ===", flush=True)
    print("url=%s  model=%s  sizes=%s  seeds=%s  warmup=%s halt_on_fail=%s "
          "pair_sleep=%.0fs warmup_score_sleep=%.0fs"
          % (URL, MODEL, WORDS, SEEDS, WARMUP, HALT_ON_FAIL, PAIR_SLEEP, WARMUP_SCORE_SLEEP),
          flush=True)
    # Pair each length: warmup N then score N. Warm-all-then-score-all lets a later
    # EngineCore 500 make earlier lengths look like found=0/10.
    results = {}  # n_words -> list of scores across seeds (None = timeout/error)
    halted = False
    for i, n in enumerate(WORDS):
        if halted:
            results[n] = [None] * len(SEEDS)
            print("words=%6d  SKIP (ladder halted)" % n, flush=True)
            continue
        if i > 0 and PAIR_SLEEP > 0:
            print(
                "[niah] sleep %.0fs between pairs (after words=%d, before words=%d)"
                % (PAIR_SLEEP, WORDS[i - 1], n),
                flush=True,
            )
            time.sleep(PAIR_SLEEP)
        if WARMUP:
            print("=== pair words=%d (warmup then score) ===" % n, flush=True)
            if not warmup(n):
                results[n] = [None] * len(SEEDS)
                print("words=%6d  SKIP score (warmup failed)" % n, flush=True)
                if HALT_ON_FAIL:
                    print("NIAH_HALT_ON_FAIL=1 — stopping remaining lengths", flush=True)
                    halted = True
                continue
            if WARMUP_SCORE_SLEEP > 0:
                print(
                    "[niah] sleep %.0fs between warmup and score words=%d"
                    % (WARMUP_SCORE_SLEEP, n),
                    flush=True,
                )
                time.sleep(WARMUP_SCORE_SLEEP)
        results[n] = [run(n, s) for s in SEEDS]
        if HALT_ON_FAIL and all(v is None for v in results[n]):
            print("NIAH_HALT_ON_FAIL=1 — stopping remaining lengths (score failed)", flush=True)
            halted = True
    print("=== NIAH summary (mean/min/max across %d seed(s)) ===" % len(SEEDS), flush=True)
    for n in WORDS:
        scored = results[n]
        vals = [v for v in scored if v is not None]
        n_to = sum(1 for v in scored if v is None)  # timeouts/errors, excluded from mean
        if not vals:
            print("  words=%6d  NO-RESULT (%d/%d timed out or errored — likely cold compile; "
                  "raise NIAH_TIMEOUT or keep NIAH_WARMUP=1)" % (n, n_to, len(scored)), flush=True)
            continue
        mean = sum(vals) / len(vals)
        extra = ("  [%d timeout/err excluded]" % n_to) if n_to else ""
        print("  words=%6d  mean=%.1f/10  min=%d  max=%d  (n=%d)%s"
              % (n, mean, min(vals), max(vals), len(vals), extra), flush=True)


if __name__ == "__main__":
    main()
