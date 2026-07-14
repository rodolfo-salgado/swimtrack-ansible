# Tracker sweep `tracker-v1`

> Este reporte conserva el criterio provisional utilizado durante el run histórico. El manifiesto formal `lap-ground-truth.yml`, creado después, reemplaza el timestamp de `test06` por la anotación fuente de `19,0 s` y usa una tolerancia de ±2 s.

## Método

El sweep procesó secuencialmente los videos completos `test01` (`no_lap`) y `test06` (`lap`) con nueve configuraciones de detector, ROI y ByteTrack sobre el mismo despliegue Front → AI en GPU 0. La anotación de producto es provisional: `test01` no contiene giro y el giro de `test06` se fijó visualmente en `18,1 s`, con tolerancia de `1,5 s`. Todos los runs conservaron diagnostics en el 100 % de los frames. La ejecución comenzó el 14 de julio de 2026 a las 00:40 UTC y terminó a las 01:21 UTC.

`Margin` es `max(lap_score de test06) - max(lap_score de test01)`; un valor positivo separa correctamente estos dos ejemplos. `Coverage` es la fracción de frames con al menos un track activo, `IDs` es el número de IDs ByteTrack únicos y `Longest` es la mayor secuencia consecutiva de un mismo ID. `Error` es la diferencia entre el candidate time de `test06` y la anotación de `18.100 ms`.

| Variante | Margin | Test01 coverage / IDs / longest | Test06 coverage / IDs / longest | Error test06 |
|---|---:|---:|---:|---:|
| `legacy_default` | +0,0053 | 16,10 % / 9 / 0,50 s | 17,14 % / 22 / 0,83 s | 733 ms |
| `roi_legacy` | +0,0053 | 16,10 % / 9 / 0,50 s | 14,22 % / 19 / 0,58 s | 733 ms |
| `score_020` | +0,0373 | 18,39 % / 8 / 1,23 s | 18,07 % / 19 / 1,13 s | 750 ms |
| `score_015` | +0,0621 | 18,95 % / 8 / 1,23 s | 19,99 % / 20 / 1,17 s | 767 ms |
| `area_250` | **+0,0647** | **19,50 % / 8 / 1,23 s** | **19,99 % / 20 / 1,17 s** | 767 ms |
| `area_150` | +0,0647 | 19,50 % / 8 / 1,23 s | 19,99 % / 20 / 1,17 s | 767 ms |
| `track_035` | −0,0569 | 24,98 % / 13 / 1,23 s | 21,73 % / 35 / 1,20 s | 67 ms |
| `buffer_090` | −0,0569 | 24,98 % / 13 / 1,23 s | 21,73 % / 35 / 1,20 s | 67 ms |
| `match_090` | −0,0662 | 27,83 % / 9 / 1,23 s | 24,55 % / 21 / 1,18 s | 50 ms |

## Recomendación

Usar `area_250` como default de despliegue: ROI habilitado, `score_threshold=0.15`, `min_box_area=250`, `track_threshold=0.45`, `track_buffer=60`, `match_threshold=0.80` y `mot20=false`. Obtuvo el mejor margin positivo (`+0,0647`), aumentó coverage respecto de `legacy_default` en ambos videos y redujo los IDs de `test01` de 9 a 8. `area_150` produjo exactamente los mismos resultados, por lo que el área 250 es preferible como gate más conservador sin pérdida observable en este dataset.

Bajar `track_threshold` a 0,35 o aumentar `match_threshold` a 0,90 mejoró coverage y redujo el error temporal del candidate, pero invirtió la separación de clases: el score máximo de `no_lap` quedó por encima del de `lap`. Además, `track_035` elevó la fragmentación a 13 IDs en `test01` y 35 en `test06`. Para el objetivo actual de contar con una confidence metric separable, ese intercambio no es aceptable.

## Limitaciones

- Sólo se evaluaron dos videos y una única perspectiva; el margin observado no estima generalización, F1 ni un threshold de producción.
- Las etiquetas y el timestamp del giro son anotaciones visuales provisionales, no un ground truth formal revisado por más de un anotador.
- `area_250` y `area_150` empataron porque bajar el área no incorporó observaciones adicionales en estos dos videos; hacen falta ejemplos con nadadores pequeños en el extremo lejano para distinguirlos.
- Coverage mezcla fallos del detector, asociación del tracker y periodos sin nadador visible; no equivale directamente a recall.
- Los IDs ByteTrack no son identidad de producto. La identidad estable debe seguir siendo el carril y la lógica definitiva de vuelta aún está pendiente.
- El runner varió parámetros de forma incremental, por lo que no cubrió el producto cartesiano completo ni interacciones entre todos los valores.
- La configuración seleccionada se guardó como default tanto en `swimtrack-ai` como en Ansible, pero debe seguir considerándose una baseline experimental hasta ampliar el dataset.

Los artefactos fuente del run están en `results/tracker-sweeps/tracker-v1/` fuera del repositorio `swimtrack-ansible`; `comparison.json` y `comparison.csv` conservan los valores completos utilizados en este reporte.
