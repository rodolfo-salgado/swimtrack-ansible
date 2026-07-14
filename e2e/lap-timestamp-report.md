# Ajuste del lap score con TIMESTAMPS.md

## Alcance

`e2e/lap-ground-truth.yml` transcribe y valida los timestamps, checksums y metadatos de `input_vids/TIMESTAMPS.md`. `test01`–`test08` forman el conjunto primario: seis vueltas anotadas en cinco videos y tres videos sin vuelta. `test09` es secundario porque dos nadadores comparten el mismo carril, lo que viola el supuesto actual de un nadador por carril.

La anotación declara aproximadamente ±1 s de precisión y una duración de vuelta de aproximadamente 2 s. La evaluación acepta una predicción hasta ±2 s de su timestamp central, agrupa observaciones del mismo `candidate_episode_id` como una visita a la pared y hace matching uno a uno por carril. Informa TP, FP, FN, precision, recall y F1. Todavía no informa TN porque no se ha definido la duración `X` ni la alineación de los intervalos `lap`/`no_lap` del producto.

## Ajustes implementados

- `trajectory-v2` exige una observación previa en el interior del carril antes de habilitar una vuelta. Esto evita interpretar la partida desde una pared como vuelta y calcula la calidad sobre la ventana local del candidato.
- `trajectory-v3` representa cada visita a una pared como un episodio persistente y publica `candidate_episode_id`. Esto consolida las observaciones de entrada y salida que antes podían contar dos eventos para una sola vuelta.
- `trajectory-v4` conserva 10 s de trayectoria, permite hasta 6 s entre la evidencia inbound y outbound y usa las detecciones dentro del ROI como fallback cuando ByteTrack no tiene un track activo. Esto recupera la segunda vuelta de `test08`, donde el nadador desaparece aproximadamente 5.1 s bajo el agua.
- `trajectory-v5` aplica un gate de continuidad a las observaciones raw del fallback: tolerancia longitudinal inicial 0.12, crecimiento 0.04 por segundo y máximo 0.20. El reproceso live de v4 reveló reflejos que saltaban `0.351→0.948` en 83 ms y `0.589→0.948` en 33 ms en `test04`; v5 rechaza esos desplazamientos físicamente imposibles sin rechazar la reaparición coherente de `test08`.

El `lap_score` sigue siendo una métrica heurística continua y no una probabilidad calibrada. `no_lap_score` es su complemento sólo cuando la ventana contiene evidencia suficiente; un cuadro no evaluable no se etiqueta automáticamente como `no_lap`.

## Metodología reproducible

Los nueve videos se procesaron completos mediante Apache → Front → AI/TensorRT en GPU 0 con `trajectory-v4` y `tracking_diagnostics=boxes`. Los streams están bajo `../results/timestamps-trajectory-v4/`. `scripts/replay_lap_scores.py` reprodujo exactamente esas observaciones con `trajectory-v5`, sin volver a ejecutar RT-DETRv2, TensorRT ni ByteTrack, lo que aísla el efecto del gate de continuidad. `scripts/evaluate_lap_events.py` evaluó los eventos contra el manifiesto.

## Curva primaria de trajectory-v5

| Threshold exploratorio | TP | FP | FN | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.03 | 6 | 2 | 0 | 0.750 | 1.000 | 0.857 |
| 0.05 | 6 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| 0.06 | 6 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| 0.07 | 6 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| 0.08–0.20 | 5 | 0 | 1 | 1.000 | 0.833 | 0.909 |
| 0.25 | 1 | 0 | 5 | 1.000 | 0.167 | 0.286 |

La separación observada es `max(FP)=0.037999` y `min(TP)=0.072141`. El rango 0.05–0.07 separa perfectamente este conjunto, pero no se fija como threshold de producto: los mismos videos sirvieron para diagnosticar y ajustar el scorer, por lo que esta métrica es de desarrollo y no una estimación imparcial de generalización.

## Eventos observados

| Video | Tipo | Endpoint | Predicción | Score | Error al GT |
|---|---|---|---:|---:|---:|
| test01 | no lap | — | — | — | — |
| test02 | no lap | — | — | — | — |
| test03 | no lap | — | — | — | — |
| test04 | TP | near | 31.667 s | 0.239407 | 1.667 s |
| test05 | TP | near | 23.250 s | 0.277994 | 1.750 s |
| test05 | FP débil | far | 42.050 s | 0.030644 | — |
| test06 | TP | near | 17.733 s | 0.204143 | 1.267 s |
| test07 | TP | near | 27.933 s | 0.236331 | 0.933 s |
| test07 | FP débil | far | 51.350 s | 0.037999 | — |
| test08 | TP | near | 23.967 s | 0.208577 | 0.033 s |
| test08 | TP con gap | far | 45.200 s | 0.072141 | 1.800 s |

`test09` no entra en el agregado. Con dos nadadores en el mismo carril, v5 detecta sólo una de las dos vueltas cercanas y añade eventos espurios; entre thresholds 0.15 y 0.20 obtiene TP=1, FP=1 y FN=1. Resolverlo requiere asociación multiobjetivo por carril y no un ajuste del threshold del caso de un nadador.

## Validación del despliegue final

`trajectory-v5` se desplegó en la máquina GPU y el Front volvió al nivel de diagnostics `counts`. Se reprocesaron live los dos casos de regresión bajo `../results/timestamps-trajectory-v5/`: `test04` entregó 3601/3601 eventos SSE y un único episodio sobre 0.05 (`31.667 s`, score `0.239407`); `test08` entregó 4322/4322 y sus dos episodios (`23.967 s`, score `0.208577`; `45.200 s`, score `0.072141`). La evaluación conjunta a 0.05 quedó TP=3, FP=0 y FN=0 en `../results/timestamps-trajectory-v5/evaluation-005.json`.

Los smoke tests de AI/TensorRT y Front pasaron sobre las dos máquinas. El E2E publicado Apache → Front → AI procesó y validó los 20/20 cuadros de su fixture SSE.

## Interpretación y trabajo pendiente

El scorer ya entrega la señal necesaria para que el Front aplique posteriormente un threshold y emita una sola notificación por `candidate_episode_id`. Antes de elegir ese threshold se necesita un conjunto holdout con más nadadores, estilos, velocidades e iluminación, idealmente anotado de forma independiente. Después se puede seleccionar el punto de operación con una curva precision/recall o calibrar el score con un método supervisado.

Para construir la matriz de confusión clásica por intervalos falta fijar `X` y la regla de alineación. Una vez definidos, cada intervalo que solape el intervalo tolerado de una vuelta será `lap`; los demás intervalos dentro de Inicio–Fin serán `no_lap`. Los eventos actuales permiten calcular TP/FP/FN sin inventar TN mientras esa decisión de producto siga pendiente.
