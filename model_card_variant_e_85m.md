---
language:
- en
tags:
- music
- midi
- gated-delta-network
- piano
- gdn
- grouped-query-attention
- autoregressive
license: cc-by-nc-sa-4.0
library_name: pytorch
datasets:
- projectlosangeles/Godzilla-MIDI-Dataset
parameters: 85000000
---

![Untitled (3)](https://cdn-uploads.huggingface.co/production/uploads/6497370627e41e26a3295604/YVvbud2ccQnmT3WMuaaeR.png)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Chickaboo/Pulse88-E-85M-Alpha-Preview/blob/main/colab_generation.ipynb)

**Pulse88-85M-Alpha Architectural Variant E** is a high-efficiency causal piano continuation model utilizing approximately 85 million parameters, trained on piano pieces from the Godzilla MIDI Dataset. It utilizes a novel hybrid architecture combining **Gated Delta Networks (GDN)** with **dense Grouped-Query Attention (GQA)** running in parallel to achieve optimal long-context musical coherence and high efficiency.

## Bullet Points
- **Architecture:** Parallel Hybrid Gated Delta Network + Dense GQA
- **Parameters:** ~85,000,000 (85M)
- **Vocabulary:** 374-token custom event vocabulary (delta, pitch, duration, velocity) plus MetaTokens for structural conditioning
- **Context Window:** 2048 tokens (512 seed / 1536 continuation)
- **Training Data:** Godzilla MIDI Dataset Piano Subset

### Variant E 85M Architecture Summary
Variant E is a decoder-only autoregressive piano MIDI model built on a custom event vocabulary with event quads (delta, pitch, duration, velocity). This ~85M profile transitions away from the older sparse attention anchors, instead utilizing **parallel hybrid-head blocks**. In each layer, the channel width is split into a Gated Delta Net (GDN) path and a dense Grouped-Query Attention (GQA) path. Both paths process the sequence in parallel and their outputs are fused via a linear projection before the residual connection. 

This approach provides the infinite-context decay and efficiency of GDN alongside the precise token-retrieval capabilities of standard dense GQA. The network utilizes tied token embeddings and output heads, dropout regularization, continuous time encoding, and dynamic output logit scaling (`1/sqrt(d_model)`). The training configuration requires real flash-linear-attention GDN kernels to efficiently scale and train, and runs on an AdamW optimizer with cosine learning rate decay, label smoothing, and weight decay.

### Tokenization & Vocabulary
The model processes MIDI data using a custom **delta tokenizer** with a total **vocabulary size of 374**. It represents musical notes as structured "event quads" (`event_size=4`), specifically decoding in the sequence: **Delta Onset (0-127) → Pitch (128-215) → Duration (216-343) → Velocity (344-359)**. 

To guide the model's generation process, the vocabulary includes **MetaTokens (360-373)**. These act as a structural prefix for musical attributes like **Density**, **Voices**, and **Register**, alongside a dedicated **START** token. By providing high-level structural context before the note sequence begins, the model's search space is effectively reduced. This bypasses the need for the model to infer the global style from scratch, freeing up its parameter capacity to focus entirely on musicality, phrasing, and harmonic development within those predefined constraints.

### Training
See the [training_logs.txt](https://huggingface.co/Chickaboo/Pulse88-E-85M-Alpha-Preview/blob/main/training_logs.txt) for exact loss numbers.
*(Training graphs placeholder)*

## Dataset
The model was trained on the **Godzilla MIDI Dataset** piano subset. This dataset, created by **Project Los Angeles (Aleksandr Lev)**, provides a massive and diverse corpus of MIDI data that allows the model to learn complex harmonic structures and temporal continuity.

## Demo

**Bluebird Continuation**
<audio controls>
  <source src="https://huggingface.co/Chickaboo/Pulse88-E-85M-Alpha-Preview/resolve/main/generations/bluebird_continuation.wav" type="audio/wav">
  Your browser does not support the audio element.
</audio>

<img src="https://huggingface.co/Chickaboo/Pulse88-E-85M-Alpha-Preview/resolve/main/generations/bluebird.png" width="800">

---
**Single Note Continuation**
In this generation the model was given only a single note (C4).  
*For optimal results, a longer seed is recommended.*

<audio controls>
  <source src="https://huggingface.co/Chickaboo/Pulse88-E-85M-Alpha-Preview/resolve/main/generations/c4_continuation.wav" type="audio/wav">
  Your browser does not support the audio element.
</audio>
<img src="https://huggingface.co/Chickaboo/Pulse88-E-85M-Alpha-Preview/resolve/main/generations/c4.png" width="800">
---

## Other Generations

**God Rest Ye Merry Gentlemen Continuation**

<audio controls>
  <source src="https://huggingface.co/Chickaboo/Pulse88-E-85M-Alpha-Preview/resolve/main/generations/godrest_continuation.wav" type="audio/wav">
  Your browser does not support the audio element.
</audio>

---
**Continuation of a simple motif**
<audio controls>
  <source src="https://huggingface.co/Chickaboo/Pulse88-E-85M-Alpha-Preview/resolve/main/generations/simple_motif_continuation.wav" type="audio/wav">
  Your browser does not support the audio element.
</audio>
---
**Sabrina by John Williams**
<audio controls>
  <source src="https://huggingface.co/Chickaboo/Pulse88-E-85M-Alpha-Preview/resolve/main/generations/john_williams_sabrina_continuation.wav" type="audio/wav">
  Your browser does not support the audio element.
</audio>
---
**Wii Channel Continuation**
<audio controls>
  <source src="https://huggingface.co/Chickaboo/Pulse88-E-85M-Alpha-Preview/resolve/main/generations/wii_channel_continuation.wav" type="audio/wav">
  Your browser does not support the audio element.
</audio>
---
*Audio rendered with [Advanced MIDI Renderer](https://huggingface.co/spaces/asigalov61/Advanced-MIDI-Renderer)*

## Intended Use
- Research and experimentation in symbolic piano continuation
- Evaluation of parallel GDN + Dense GQA hybrid architectural designs

## Limitations
- The model is limited to piano-only MIDI data and does not generalize to multi-instrument compositions.
- Performance degrades with very short or highly irregular input seeds.
- The model may produce repetitive or unstable outputs over long continuations.
- As an alpha preview, the model has not been extensively optimized for musical quality or stylistic control.

## Project Future and Purpose
The purpose of this project is the research of novel architectures in the symbolic music Machine learning space. This is a small scale preview of what is to come. At the moment, the model is somewhat undertrained, having completed only around 9,500 steps. I will be continuing its training to bring it to its full potential, and I plan to keep working on the architecture to stay on the bleeding edge of technology.

## Warranty
This model is intended for research purposes only. It is provided “as is,” without any warranties, express or implied. The authors make no guarantees regarding its performance, reliability, or fitness for a particular purpose. Use at your own risk.

## Special Thanks
Special thanks to **Aleksandr Lev**, creator of the Godzilla MIDI Dataset and Tegridy Code. He has generously taken the time to provide valuable feedback on every model released throughout this project. His consistent insights and expert guidance have been incredibly helpful in shaping these models.

## Citation & Credits
If you use this model, please credit the original data source:
```bibtex
@misc{GodzillaMIDIDataset2025,
  title        = {Godzilla MIDI Dataset: Enormous, comprehensive, normalized and searchable MIDI dataset for MIR and symbolic music AI purposes},
  author       = {Alex Lev},
  publisher    = {Project Los Angeles / Tegridy Code},
  year         = {2025},
  url          = {https://huggingface.co/datasets/projectlosangeles/Godzilla-MIDI-Dataset}
}
```
```bibtex
@inproceedings{lev2026tegridytools,
    title       = {tegridy-tools: Symbolic Music NLP Artificial Intelligence Toolkit},
    author      = {Aleksandr Lev},
    booktitle   = {GitHub},
    year        = {2026}
}
```