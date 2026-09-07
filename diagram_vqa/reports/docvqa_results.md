# DocVQA: воспроизводимые результаты

ANLS открытых ответов и retrieval-метрики публикуются раздельно.
Старый checkpoint `docvqa-auto` с Mean R@1=0.04156 считается legacy и не входит в таблицу.

## Открытые ответы на official val

| Модель | ANLS | Exact accuracy | Latency, ms/question | Статус |
|---|---:|---:|---:|---|
| Most frequent | 0.0079 | 0.0079 | 0.03 | available_full |
| Random train answer | 0.0018 | 0.0006 | 0.07 | available_full |
| Question lexical retrieval | 0.0659 | 0.0389 | 1.56 | available_full |
| Qwen2.5-VL-3B | 0.9059 | 0.8469 | 28501.69 | available_full |

## Поиск документа

| Модель | Val Mean R@1 | Test Mean R@1 | Val Mean R@5 | Статус |
|---|---:|---:|---:|---|
| Random | 0.0003 | 0.0009 | 0.0031 | available_full |
| OCR text | 0.3839 | 0.3786 | 0.5620 | available_full |
| CLIP ViT-B/32 | 0.0364 | 0.0427 | 0.0780 | available_full |

## OCR-графовые абляции

| Вариант | Seeds | Val Mean R@1, mean±std | Test Mean R@1, mean±std | Статус |
|---|---:|---:|---:|---|
| text_only | 3 | 0.0355 ± 0.0033 | 0.0367 ± 0.0042 | available_full |
| coords_no_edges | 3 | 0.0348 ± 0.0013 | 0.0380 ± 0.0053 | available_full |
| knn_k3 | 3 | 0.0355 ± 0.0027 | 0.0386 ± 0.0051 | available_full |
| knn_k5 | 3 | 0.0355 ± 0.0033 | 0.0419 ± 0.0020 | available_full |
| typed_spatial | 3 | 0.0358 ± 0.0030 | 0.0410 ± 0.0009 | available_full |
