# Reverb ASR — Decoding Parameters & Mode‑Specific Matrix

This document enumerates every runtime parameter exposed by `recognize_wav.py` in the Reverb fork (see source commit **HEAD**), grouped by **global settings** and **decode‑mode‑specific knobs**. Use it to drive dynamic UI rendering so the user only sees parameters relevant to the chosen decoding mode.

---

## 1  Global Parameters (Always Visible)

| Arg                        | Type    | Default | **Range / Allowed Values**     | Tooltip                                                                              |
| -------------------------- | ------- | ------- | ------------------------------ | ------------------------------------------------------------------------------------ |
| `audio_file`               | *path*  | —       | —                              | Input WAV/MP3/MP4 etc. Will be converted to 16 kHz mono WAV before inference.        |
| `result_dir`               | *path*  | —       | —                              | Folder to save CTM/text output. A sub‑folder is created per selected mode.           |
| `gpu`                      | *int*   | `-1`    | `-1` (CPU) or `0–7`            | CUDA device index; `‑1` forces CPU execution.                                        |
| `verbatimicity`            | *float* | `1.0`   | `0.0–1.0`                      | 1 = verbatim (ums, hesitations kept); 0 = clean. Values in‑between give mixed style. |
| `batch_size`               | *int*   | `1`     | `1–16`                         | Number of \~20‑s chunks decoded in parallel (higher → faster but more VRAM).         |
| `chunk_size`               | *int*   | `2051`  | `160–4096` (frames≈10 ms)      | Length of each chunk passed to the model; 2051 ≈ 20 s.                               |
| `decoding_chunk_size`      | *int*   | `‑1`    | `‑1` (full) or `160–2048`      | Streaming only: <0 = full context; >0 = fixed frames per step.                       |
| `num_decoding_left_chunks` | *int*   | `‑1`    | `‑1` (full) or `0–32`          | How many previous chunks of history to carry in streaming mode.                      |
| `simulate_streaming`       | *flag*  | `false` | true / false                   | Treat offline file as live stream (forces chunked decoding).                         |
| `timings_adjustment`       | *ms*    | `230`   | `0–500`                        | Subtracted from each timestamp to compensate for look‑ahead latency.                 |
| `overwrite_cmvn`           | *flag*  | `false` | true / false                   | Replace CMVN stats in checkpoint with values from YAML.                              |
| `tokenizer_symbols`        | *path*  | —       | —                              | Custom `tk.units.txt` (override YAML).                                               |
| `bpe_path`                 | *path*  | —       | —                              | Custom `tk.model` (override YAML).                                                   |
| `cmvn_path`                | *path*  | —       | —                              | Custom `cmvn` stats (override YAML).                                                 |
| `log_level`                | *enum*  | `INFO`  | DEBUG / INFO / WARNING / ERROR | Logging verbosity for console/debug pane.                                            |

---

## 2  Decode Modes

| Mode key                 | Short description                                          | Typical use                              |
| ------------------------ | ---------------------------------------------------------- | ---------------------------------------- |
| `ctc_greedy_search`      | Fastest; picks highest‑prob token at each frame (no beam). | Real‑time caption preview.               |
| `ctc_prefix_beam_search` | CTC beam search without attention.                         | Good accuracy / speed trade‑off.         |
| `attention`              | Pure attention decoder beam search.                        | Highest fluency, slower.                 |
| `attention_rescoring`    | CTC prefix beam → attention rescoring.                     | Default in Reverb demo; strong accuracy. |
| `joint_decoding`         | CTC & attention scoring combined per step.                 | Experimental, balanced.                  |

---

## 3  Mode‑Specific Parameter Matrix

| Parameter                  | Type  | Default | **Greedy** | **CTC‑Beam** | **Attention** | **Attn‑Rescore** | **Joint** | **UI Range**   | Tooltip                                                                           |
| -------------------------- | ----- | ------- | ---------- | ------------ | ------------- | ---------------- | --------- | -------------- | --------------------------------------------------------------------------------- |
| `beam_size`                | int   | 10      | —          | ✔            | ✔             | ✔                | ✔         | `1–30`         | Max hypotheses kept during beam search. Larger = better but slower.               |
| `length_penalty`           | float | 0.0     | —          | —            | ✔             | —                | ✔         | `‑2.0–2.0`     | Positive values favor longer sentences; negative shorter. Only attention & joint. |
| `blank_penalty`            | float | 0.0     | —          | ✔            | —             | ✔\*              | ✔\*       | `0.0–2.0`      | Penalize blank symbol to discourage stalls (*via underlying CTC step*).           |
| `ctc_weight`               | float | 0.1     | —          | —            | —             | ✔                | ✔         | `0.0–1.0`      | Weight of CTC score when rescoring / joint decoding.                              |
| `reverse_weight`           | float | 0.0     | —          | —            | —             | ✔                | —         | `0.0–0.5`      | Adds right‑to‑left attention probability; improves punctuation.                   |
| `decoding_chunk_size`      | int   | ‑1      | ✔          | ✔            | ✔             | ✔                | ✔         | Same as global | In streaming mode: frames per forward pass (set via global).                      |
| `num_decoding_left_chunks` | int   | ‑1      | ✔          | ✔            | ✔             | ✔                | ✔         | Same as global | History context for streaming.                                                    |

Legend: **✔** = parameter used; **—** = ignored. *Blank penalty indirectly applies where CTC beam is part of the algorithm.*

---

## 4 Sources

* Reverb argument parser (`recognize_wav.py`) ([raw.githubusercontent.com](https://raw.githubusercontent.com/revdotcom/reverb/main/asr/wenet/bin/recognize_wav.py))
* Hugging Face model card — list of decode modes & example command ([huggingface.co](https://huggingface.co/Revai/reverb-asr?utm_source=chatgpt.com))
