# Final memory store (submission deliverable)

The files in this folder are byte-identical copies of `memory/runs/t4fix_hybrid_mem/`:
the store as it stands **after Session B of the final `hybrid_mem` run** (`t4fix_hybrid_mem_A` → `t4fix_hybrid_mem_B`,
two separate processes; see `session_manifest.json` for PIDs and SHA-256 hashes).

| File | Contents |
|---|---|
| `semantic_facts.json`, `semantic_vectors.npy`, `semantic_index.json` | Semantic memory: keyed user facts (active and superseded) and the vector index of active facts |
| `episodes.json`, `episodic_vectors.npy`, `episodic_index.json` | Episodic memory: time-stamped events with importance, including the end-of-session reflection |
| `forget_tombstones.json` | Forget record: key and SHA-256 fingerprint only (the forgotten value is not stored) |

`memory/runs/<experiment>/` keeps every run's own store (Task 3 `t3_hm_*`, Task 4 ablation `t4_*`, after-fix `t4fix_*`).
