# Evaluación de confidence por ventanas con `trajectory-v5`

## Estado

Esta evaluación se generó el 2026-07-14 usando los replays persistentes de `test01`–`test09`. Los resultados son exclusivamente de development: los mismos videos se usaron durante el ajuste del scorer, sólo existen 6 vueltas primarias y no hay un holdout independiente. `lap_score` es un score heurístico en `[0,1]`, no una probabilidad calibrada.

Revisiones de referencia:

- Scorer `swimtrack-ai`: `10a0fe50f0b834597d3a3f55fb2e093262087f9b`, `score_version=trajectory-v5`.
- Evaluator `swimtrack-ansible`: `e4d92148685a924f9b83753254e2e315c478dd6f`.
- Reducer shadow `swimtrack-front`: `44aad58c28c982b8ffb2d62f917d54a38233ba30`.
- Ground truth: `e2e/lap-ground-truth.yml`, transcrito desde `input_vids/TIMESTAMPS.md` con SHA-256 `90cb9889296f9198110946d9d6b4d7a2da7cd9ec9aa9fac809efef2a2c7160bd`.

La procedencia completa, incluidos los hashes de los nueve streams y de cada artefacto de salida, está en `results/lap-windows/trajectory-v5/provenance.json`.

## Definición evaluada

- Ventanas half-open `[start_ms,end_ms)`, ancladas en `t=0`.
- `stride=X`, sin overlap.
- `X` evaluado en `1`, `2`, `3` y `4 s`.
- Una fila por `(video_id,lane_id,window_index)`.
- Episode key `(video_id,lane_id,candidate_episode_id)` y reducción al máximo `lap_score` conservando su `candidate_time_ms` asociado.
- Score de ventana igual al máximo de los episodios cuyo candidate time pertenece a la ventana; no se agregan las repeticiones por frame.
- Coverage igual a la fracción de frames activos de la ventana con `evaluable=true`.
- `evaluable_fraction < 0.5` produce `abstain`; esa fila no aporta TN.
- Threshold principal mostrado `0.05`, únicamente exploratorio.
- Métrica strict: ground truth y candidato deben caer en la misma ventana canónica.
- Métrica temporal: matching uno-a-uno por carril con tolerancia `±2000 ms`.
- `test01`–`test08` forman el aggregate primary. `test09` permanece secondary porque tiene dos nadadores en el mismo carril y viola el supuesto del modelo.

## Sensibilidad a `X` con threshold `0.05`

| X | Ventanas primary | Evaluables | Abstain | Coverage | Strict TP/TN/FP/FN | Strict F1 | Strict MCC | Tolerant TP/TN/FP/FN | Tolerant F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 s | 326 | 295 | 31 | 90.49% | 1/284/5/5 | 0.167 | 0.149 | 6/289/0/0 | 1.000 |
| 2 s | 167 | 151 | 16 | 90.42% | 2/141/4/4 | 0.333 | 0.306 | 6/145/0/0 | 1.000 |
| 3 s | 116 | 105 | 11 | 90.52% | 3/96/3/3 | 0.500 | 0.470 | 6/99/0/0 | 1.000 |
| 4 s | 89 | 78 | 11 | 87.64% | 4/70/2/2 | 0.667 | 0.639 | 6/72/0/0 | 1.000 |

El aumento de F1 strict al aumentar `X` es un efecto de la grilla y no evidencia una mejora del scorer. La métrica temporal reconoce los seis candidatos primarios dentro de la incertidumbre real de las anotaciones para los cuatro valores de `X`.

## Sensibilidad al threshold para `X=2 s`

| Threshold | TP | TN | FP | FN | Precision | Recall | F1 | Specificity | Balanced accuracy | MCC |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.03 | 6 | 143 | 2 | 0 | 0.750 | 1.000 | 0.857 | 0.986 | 0.993 | 0.860 |
| 0.05 | 6 | 145 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.07 | 6 | 145 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.08 | 5 | 145 | 0 | 1 | 1.000 | 0.833 | 0.909 | 1.000 | 0.917 | 0.910 |

Esta tabla usa la variante temporal-tolerant. El rango `0.05–0.07` separa perfectamente este pequeño conjunto de development, pero no se fija como threshold de producto.

## Secondary: `test09`

Con `X=2 s` y threshold `0.05`, `test09` tiene 26 ventanas evaluables y coverage `100%`. Su evaluación temporal produce `TP=1`, `TN=23`, `FP=2`, `FN=1` y `F1=0.4`. Este resultado aparece en `aggregate_secondary` y `aggregate_all`, pero no modifica `aggregate_primary`. Como hay dos ground-truth events y múltiples episodios en un mismo carril, la matriz temporal secundaria tiene semántica de matching por evento y no debe interpretarse como una matriz binaria pura por fila.

## Artefactos

Cada directorio `results/lap-windows/trajectory-v5/x{1000,2000,3000,4000}/` contiene:

- `windows.jsonl`: una fila por ventana para el threshold principal `0.05`, incluidos `lap_score`, coverage, label, ambiguity y decisión `lap|no_lap|abstain`.
- `evaluation-005.json`: filas, episodios reducidos, métricas por video, aggregates primary/secondary/all y el threshold sweep.
- `threshold-curve.json`: configuración y aggregates para `0.03`, `0.05`, `0.07` y `0.08` sin reprocesar los streams.

Conteos JSONL primary + secondary: `377`, `193`, `133` y `103` filas para `X=1`, `2`, `3` y `4 s`, respectivamente. Los conteos primary correspondientes son `326`, `167`, `116` y `89`.

## Front en shadow mode

El Front reduce observaciones por `(lane_id,candidate_episode_id)` dentro de cada request/video, conserva el máximo y emite como máximo un `lap_decisions` al primer cruce del threshold. `LAP_EPISODE_MODE=shadow` es el default y `LAP_CONFIDENCE_THRESHOLD` no tiene default. Sin threshold, sólo se registran resúmenes sanitizados; no se emiten decisiones positivas. El campo visible `count` conserva su semántica histórica y no se incrementa por vueltas.

No existe un modo `active`. Activar el conteo visible queda explícitamente pospuesto hasta obtener un dataset nuevo, separar development/holdout y bloquear un threshold.

## Validación local

- Evaluator: 8 unit tests aprobados.
- Front: 25 tests aprobados.
- Ruff y `ruff format --check`: aprobados en los archivos nuevos.
- Ansible syntax check: aprobado para `deploy-worktree.yml` y `deploy-front.yml`.
- Baseline reproducido para `X=2 s`: 167 ventanas primary, 151 evaluables, 16 abstenciones; strict `2/141/4/4`; temporal `6/145/0/0`.

## Reproducción

Ejecutar desde `swimtrack-ansible/`, siempre con `uv`:

```bash
uv run --script scripts/evaluate_lap_windows.py --window-size-ms 2000 --stride-ms 2000 --anchor-ms 0 --coverage-threshold 0.5 --threshold 0.05 --sweep-threshold 0.03 --sweep-threshold 0.05 --sweep-threshold 0.07 --sweep-threshold 0.08 --expected-score-version trajectory-v5 --stream test01=../results/timestamps-trajectory-v5-replay/20260513_201705_test01/stream.sse --stream test02=../results/timestamps-trajectory-v5-replay/20260513_201705_test02/stream.sse --stream test03=../results/timestamps-trajectory-v5-replay/20260513_201705_test03/stream.sse --stream test04=../results/timestamps-trajectory-v5-replay/20260513_201705_test04/stream.sse --stream test05=../results/timestamps-trajectory-v5-replay/20260513_201705_test05/stream.sse --stream test06=../results/timestamps-trajectory-v5-replay/20260513_201705_test06/stream.sse --stream test07=../results/timestamps-trajectory-v5-replay/20260513_201705_test07/stream.sse --stream test08=../results/timestamps-trajectory-v5-replay/20260513_201705_test08/stream.sse --stream test09=../results/timestamps-trajectory-v5-replay/20260513_201705_test09/stream.sse --output ../results/lap-windows/trajectory-v5/x2000/evaluation-005.json --rows-output ../results/lap-windows/trajectory-v5/x2000/windows.jsonl
```

## Trabajo que requiere datos nuevos

1. Recolectar videos nuevos con un nadador por carril, incluyendo cámaras, nadadores, estilos, velocidades y oclusiones distintas.
2. Separar development y holdout por video o nadador antes de ajustar el threshold.
3. Seleccionar el threshold usando sólo development y evaluar holdout una vez.
4. Evaluar calibración probabilística sólo cuando haya suficientes episodios positivos y negativos independientes.
5. Habilitar el conteo visible únicamente después de fijar el contrato de producto y sus regresiones.
