# Resource measurement for slide ABL_RES_SLIDE_2026

Measured on 2026-09-06, NVIDIA GeForce RTX 4060 Laptop GPU, PyTorch 2.6.0+cu124. This is a pilot inference benchmark, not a training or end-to-end pipeline benchmark.

The same 32 AI2D test questions are sampled with Python Random(42). Batch size is 1. Graph and CLIP checkpoints use training seed 42. Inputs are prepared on CPU before timing; timing includes transfer to GPU and answer scoring. OCR, SAM2, DINOv2, MiniLM, CLIP encoders, file reading and model loading are excluded. CLIP refers to its trained answer head over cached CLIP embeddings.

Qwen uses the existing compatible LoRA adapter derived from runs/vlm_qlora_mcq/adapter and the local Qwen2.5-VL-3B model, NF4 quantization, image max_pixels=262144 and at most 32 generated tokens. Its timed model includes the visual encoder and generation; PIL processing/tokenization and decoding are excluded. It is a different computation boundary from the modules over cached features, so these numbers cannot establish an end-to-end speedup. The adapter does not revalidate the archived 73.19% accuracy.

Graph/CLIP measurements: 10 warmup calls, 3 passes of 32 questions. Qwen: 2 warmup calls, 1 pass of 32 questions. Wall-clock timing uses CUDA synchronization. GPU memory is maximum torch.cuda.max_memory_allocated in GiB (2^30 bytes), including the model and current tensors; it excludes CUDA context, allocator reserve and other processes. This is not total board memory or minimum GPU capacity. An existing notebook process was present; GPU use was not exclusive.

Random uses CPU option selection, 320000 draws per pass, 3 passes, and no GPU memory. Its per-draw value is amortized.

Source logs for CLIP and hybrid Graph kNN contain a misleading latency_ms_per_question field: elapsed time includes training and validation. Those recorded values are not used for this slide. Dual/heterogeneous notebook metrics contain no resource measurements.

Reproduction: run `.venv/Scripts/python.exe resource-slide/benchmark.py MODEL` from the data workspace for MODEL in dual, attention, knn, clip, random, qwen. Run GPU modes sequentially. Per-model JSON stores raw timings, sample IDs and checkpoint paths. Existing checkpoints are read only. The compatibility exclusion of the unused epoch_view module is required for the old hybrid checkpoint; the remaining state dict is loaded strictly.
