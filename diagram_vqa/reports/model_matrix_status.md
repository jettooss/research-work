# Матрица моделей 11×3

Текущий статус: smoke-контур завершён, полные запуски ещё не завершены.

- 33 исходных и 33 executed-ноутбука созданы.
- Все 33 пары прошли реальное обучение/оценку на малой выборке.
- Для каждой модели используется один воспроизводимый seed 42. Ранее созданные результаты seed 43/44 сохраняются как дополнительные legacy-артефакты, но больше не планируются.
- Аудит: `available_smoke=33`, `smoke_ready=true`, пропущенных checkpoints/adapters/submissions нет.
- `defense_ready=false`, пока все 33 пары не получат `available_full`.

Метрики VQA и retrieval хранятся и публикуются раздельно. Composite используется только для выбора checkpoint. Результаты вне `runs/model_matrix` и `runs/model_matrix_smoke` считаются legacy и не смешиваются с новым протоколом.

Основные артефакты:

- `experiments/model_matrix_protocol.json`
- `reports/model_matrix_smoke_aggregate.json`
- `reports/model_matrix_smoke_audit.json`
- `experiment_registry/model_matrix_v1.json`
- `model_registry/model_matrix_v1.json`
