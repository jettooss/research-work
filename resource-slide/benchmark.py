import sys, json, time, random, gc, hashlib
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / 'diagram_vqa'
sys.path[:0] = [str(PROJECT/'scripts'), str(PROJECT/'src'), str(PROJECT/'reports/presentation_metrics')]
OUT = Path(__file__).resolve().parent
torch.set_num_threads(4)
torch.manual_seed(42)
rows = [json.loads(x) for x in (ROOT/'ai2d/model_matrix_v1/manifest.jsonl').read_text(encoding='utf-8').splitlines()]
rows = random.Random(42).sample([x for x in rows if x['split']=='test'], 32)
for row in rows:
    row['image_path'] = str(ROOT/row['image_path'])
mode = sys.argv[1]
device = torch.device('cuda')

def measure(model, samples, forward, warmup=10, repeats=3):
    model.eval()
    latencies, peaks, predictions = [], [], []
    with torch.inference_mode():
        for i in range(warmup):
            sample = samples[i % len(samples)]
            forward(sample)
        torch.cuda.synchronize()
        for repeat in range(repeats):
            for sample in samples:
                gc.collect()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                start = time.perf_counter()
                output = forward(sample)
                torch.cuda.synchronize()
                latencies.append((time.perf_counter()-start)*1000)
                peaks.append(torch.cuda.max_memory_allocated()/1024**3)
                del output
            print('repeat', repeat+1, 'mean_ms', np.mean(latencies), flush=True)
    return {'mean_ms':float(np.mean(latencies)), 'median_ms':float(np.median(latencies)), 'sd_ms':float(np.std(latencies)), 'peak_vram_gib':max(peaks), 'latencies_ms':latencies, 'warmup':warmup,'repeats':repeats}

if mode in ('dual','attention','knn'):
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu', local_files_only=True)
    normalize = mode != 'knn'
    q = encoder.encode([r['question'] for r in rows], normalize_embeddings=normalize, convert_to_tensor=True).float()
    options = [encoder.encode(r['options'], normalize_embeddings=normalize, convert_to_tensor=True).float() for r in rows]
    del encoder
    if mode in ('dual','attention'):
        from evaluate_saved_retrieval import notebook_model, SCHEMA
        arch = 'sam2_dinov2_dual_branch_evidence_graph' if mode=='dual' else 'sam2_dinov2_heterogeneous_balanced_evidence_graph'
        checkpoint = PROJECT/'runs/ai2d'/arch/'seed42_full/checkpoint_best.pt'
        model = notebook_model('ai2d',arch)
        model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model'],strict=True)
        cache = PROJECT/'runs/ai2d/sam2_dinov2_regularized_sparse_learned_graph/seed42_full'/f'node_features_{SCHEMA}'
        samples=[]
        for i,r in enumerate(rows):
            x=torch.load(cache/f"{r['image_id']}.pt",map_location='cpu',weights_only=False)['x'].float().unsqueeze(0)
            samples.append((x,torch.ones(x.shape[:2],dtype=torch.bool),q[i:i+1],options[i].unsqueeze(0)))
        def forward(sample):
            values=[x.to(device) for x in sample]
            return model(*values)
    else:
        from train_model_matrix_local import GraphMatrixModel
        checkpoint = PROJECT/'runs/ai2d/hybrid_v4_multipos_training/seed42_full/checkpoint_best.pt'
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)['model']
        model=GraphMatrixModel('hybrid_gatv2_knn',hidden=state['input.weight'].shape[0],out_dim=state['node_out.weight'].shape[0]).to(device)
        if not any(k.startswith('epoch_view.') for k in state):
            del model.epoch_view  # Added later, unused by hybrid_gatv2_knn forward.
        model.load_state_dict(state,strict=True)
        samples=[(torch.load(ROOT/'model_matrix_cache/graphs/ai2d/hybrid_gatv2_knn'/f"{r['image_id']}.pt",map_location='cpu',weights_only=False),q[i],options[i]) for i,r in enumerate(rows)]
        def forward(sample):
            graph,question,answers=sample
            return model.vqa_logits(graph.clone(),question.to(device),answers.to(device))
    report=measure(model,samples,forward)
elif mode=='clip':
    from train_model_matrix_vision import FrozenVisionStore,VisionHeads
    store=FrozenVisionStore('ai2d','clip',ROOT/'model_matrix_cache/vision',torch.device('cpu'))
    samples=[(store.image(r),store.texts([r['question']])[0],store.texts(r['options'])) for r in rows]
    del store
    gc.collect()
    checkpoint=PROJECT/'runs/ai2d/clip_vqa_baseline/seed42_full/checkpoint_best.pt'
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)['heads']
    model=VisionHeads(512).to(device)
    model.load_state_dict(state,strict=True)
    report=measure(model,samples,lambda s:model.vqa(*[x.to(device) for x in s]))
elif mode=='random':
    rng=random.Random(42)
    values=[]
    for _ in range(3):
        start=time.perf_counter()
        for _ in range(10000):
            for row in rows: rng.randrange(len(row['options']))
        values.append((time.perf_counter()-start)*1000/(10000*len(rows)))
    checkpoint=None
    report={'mean_ms':float(np.mean(values)),'peak_vram_gib':0,'repeats':3,'draws_per_repeat':320000}
elif mode=='qwen':
    from vqa_retrieval.vlm_service import QwenVlmRunner
    from qwen_vl_utils import process_vision_info
    checkpoint=PROJECT/'runs/ai2d_top_models_100_seed_analysis/qwen25_vl_qlora/adapter_compat'
    runner=QwenVlmRunner(model_path=ROOT/'models/Qwen2.5-VL-3B-Instruct',adapter_path=checkpoint,use_4bit=True,device_mode='auto',max_pixels=512*512)
    model=runner.model
    samples=[]
    for r in rows:
        block='\n'.join(f'{i}: {v}' for i,v in enumerate(r['options']))
        prompt='Read the diagram and answer the multiple-choice question. Return JSON only: {"answer": "exact option text"}.\n'+f"Question: {r['question']}\nOptions:\n{block}"
        messages=[{'role':'user','content':[{'type':'image','image':r['image_path'],'max_pixels':512*512},{'type':'text','text':prompt}]}]
        images,videos=process_vision_info(messages)
        samples.append(runner.processor(text=[runner.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)],images=images,videos=videos,padding=True,return_tensors='pt'))
    def forward(sample):
        return model.generate(**{k:v.to(device) for k,v in sample.items()},max_new_tokens=32,do_sample=False)
    report=measure(model,samples,forward,warmup=2,repeats=1)
    report.update({'quantization':'4-bit NF4','max_pixels':512*512,'max_new_tokens':32,'device_map':str(getattr(model,'hf_device_map',None))})
else: raise ValueError(mode)
report.update({'model':mode,'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'samples':len(rows),'sample_ids':[r['sample_id'] for r in rows],'selection_seed':42,'checkpoint':str(checkpoint),'protocol':'batch=1; prepared inputs on CPU; transfer to GPU and complete answer scoring/generation; CUDA synchronize; no OCR/SAM/DINO/text encoders or disk IO; torch.cuda peak allocated includes model and active inputs; other processes excluded'})
(OUT/f'{mode}.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k not in ('latencies_ms','sample_ids')},ensure_ascii=False),flush=True)
