# Screencast storyboard — Continuum in 3 minutes

A frame-by-frame script for recording the demo. Total target: **3:00**.
Recording suggestion: OBS, Loom, or `asciinema rec` for the terminal panes.

## Pre-recording checklist

* Terminal font ≥ 14 pt; dark background; window 110 cols wide so the
  benchmark tables don't wrap.
* `cd /Users/mayanksahu/Continuum`
* Pre-warm the framework Python so the embedder doesn't load mid-take:
  `/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -c "import bench.synth"`
* Open `README.md` + `findings/longmemeval_2026-05.md` in adjacent
  browser tabs so you can flip when narrating (00:30 segment).

## Recording

### 00:00 – 00:15  ·  Cold open — what problem we solve

**Voice-over** (read straight, ~3 sentences):

> Most AI-agent memory is either a vector database that doesn't know
> which facts are current, or an append-only chat history that gets
> expensive fast. Continuum is the memory layer underneath your
> reasoner: tiered storage, with first-class supersession and
> bi-temporal queries. Here's the 60-second demo.

**Screen**: title card or just the GitHub README rendered. Cursor
hovers on the table at the top showing 100% supersession / 100%
bi-temporal.

### 00:15 – 01:15  ·  Run the demo

**Type at the prompt:**

```bash
make demo-chat
```

**Pause to let the script play (it's ~63 ms but the on-screen output
should be visible for ~60 seconds — slow your reading; the demo
output is the visual).**

**As output appears, narrate over these specific moments:**

* When `→ extracted user.location = NYC` and `→ extracted user.pets.dog = Rex` appear:
  > "Watch — Continuum doesn't store the raw turn and call it done.
  > It extracts atomic facts: location, pet's name. Those land in
  > LTM with their own IDs."

* When `→ extracted user.location = Boston (superseded prior)` appears:
  > "Now the user moves. The framework recognises that
  > `user.location` already has a value, and marks the NYC fact as
  > **superseded by** the Boston one. The old fact is **not deleted**
  > — that's the bi-temporal property."

* When `/show ltm` shows only Boston and Rex:
  > "Current-facts view: just Boston and Rex. This is what
  > `where do I live?` resolves to — by construction, not by
  > prompting."

* When `/show ltm --all` shows the supersession arrow:
  > "Full-history view: NYC is still there, with an arrow showing
  > which fact replaced it. Audit-trail by default."

* When the dog change happens (Rex → Luna):
  > "Same mechanic, second attribute. Watch — Rex now points at
  > Luna's ID. The system can answer 'what did the user say about
  > their dog *as of last month?*' and 'who's their dog *now?*' as
  > two distinct queries, both correctly."

### 01:15 – 02:00  ·  Show the benchmarks

**Type:**

```bash
make bench-supersession
```

(Wait for output — ~5 seconds.)

**Narrate over the table:**

> "Fifty scripted scenarios across location, employer, pets, marital
> status, vehicle, and hobbies. Continuum's supersession path: **100
> percent correct**. Append-only memory: **38 percent**. Naive
> systems return a now-stale fact in 31 of 50 cases. This isn't a
> tuning win — it's an architectural one. Vector stores cannot reach
> this number without re-implementing this schema."

**Type:**

```bash
make bench-bitemporal
```

**Narrate:**

> "And the bi-temporal benchmark: twenty scripted 'as of date Y'
> queries, fifteen point-in-time updates and five retroactive
> corrections. Continuum: **20 of 20**. The closest naive baseline,
> `naive_chronological`, gets the forward-only cases but collapses
> entirely on retroactive corrections — its filter on `recorded_at`
> can't see the corrected fact yet."

### 02:00 – 02:45  ·  Show the honest evaluation

**Switch to the browser tab with `findings/longmemeval_2026-05.md`.**

**Scroll to the headline table.**

**Narrate** (slower, this is the credibility moment):

> "We also ran Continuum through LongMemEval-S — the standard
> long-term memory benchmark. Seven full sweeps across four model
> families. Substring accuracy plateaus at 32 percent regardless of
> model or retriever — *even at 100 percent recall*. The Llama-70B
> long-context run, with the entire haystack in the prompt, scored
> the same 32 percent as top-k retrieval. The ceiling isn't memory;
> it's the model's ability to compose multi-hop answers at query
> time. We document this honestly rather than chase a leaderboard
> number with techniques that don't fit Continuum's scope."

**Scroll to the per-category chart** (`fig2_per_category.png`).

> "Continuum is not a reasoning engine. It's the memory layer that
> nobody else has built right. Plug it under LangGraph, AutoGen, or
> your own agent — that's where it belongs."

### 02:45 – 03:00  ·  Call to action

**Switch back to the terminal.**

**Type:**

```bash
ls examples/chat_agent/
```

**Narrate:**

> "Clone, `make demo-chat`, `make bench-all`. Sixty seconds to
> understand it, no infrastructure required. Repo link in the
> description."

**Fade.**

## Post-production

* If using Loom, set the **first frame** to the GitHub README's headline table
  — that's the click-through hook.
* Caption the supersession output line (`→ extracted user.location = Boston (superseded prior)`)
  with an arrow + "this is the killer feature."
* If you want a shorter cut, drop the bi-temporal benchmark
  (02:00–02:30) and ride supersession alone — that gives you a 2:30
  cut that's tighter for social.

## Asset checklist

* [ ] Pre-warmed terminal pane
* [ ] Browser tabs open: README, findings report
* [ ] OBS/Loom recording 1080p+ at 30 fps
* [ ] Mic input tested (avoid keyboard noise — use a script you can read)
* [ ] After: trim leading silence; add 1-line caption overlays at the
      key moments listed above
