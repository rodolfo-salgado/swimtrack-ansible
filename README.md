# SwimTrack Ansible

Despliega `swimtrack-ai` en la máquina GPU accesible mediante el alias SSH `proyecto_ia_gpu`, instala las dependencias con `uv`, copia el ONNX y su external data, limita el proceso a GPU 0 y mantiene Uvicorn mediante un servicio `systemd --user` persistente. También despliega `swimtrack-front` en `proyecto_ia` como servicio Gunicorn de usuario. Para esta VM temporal, AI se publica en la red privada usando el rango de puertos ya abierto y el token compartido, sin cambios de firewall ni `sudo`.

## Requisitos del controller

- Ejecutar desde Debian/WSL con el alias `proyecto_ia_gpu` definido en `~/.ssh/config`.
- Tener `swimtrack-ansible` y `swimtrack-ai` como directorios hermanos.
- Generar `swimtrack-ai/artifacts/models/rtdetrv2_s.onnx` con `uv run --script swimtrack-ai/scripts/export_rtdetrv2_onnx.py` antes del despliegue. El archivo `.onnx.data` es opcional para artifacts externos heredados y no se requiere para el artifact actual.
- Usar `uv`; Ansible Core queda fijado en `uv.lock`.

La configuración de inventario fuerza `-F ~/.ssh/config` porque el OpenSSH de este controller rechaza un archivo global con permisos inseguros. El alias sigue siendo la única dirección remota almacenada por este proyecto. Usa siempre `./ansible-run`: además de ejecutar Ansible mediante `uv`, fija explícitamente `ANSIBLE_CONFIG` porque Ansible ignora por seguridad archivos de configuración encontrados automáticamente dentro del montaje world-writable `/mnt/c`.

## Preflight

```bash
uv sync --locked
./ansible-run ansible-playbook playbooks/preflight.yml
```

El preflight no modifica el host. Comprueba el usuario `grupo1`, Python 3.12, `uv`, GPU 0, driver, glibc, espacio libre, checkout limpio, modelos locales, user bus y `Linger=yes`.

## Despliegue

```bash
./ansible-run ansible-playbook playbooks/deploy.yml
```

La primera ejecución descarga aproximadamente 3.5 GiB de dependencias CUDA/TensorRT y construye el engine FP16, por lo que puede tardar varios minutos. El playbook espera hasta 20 minutos por `/readyz` y muestra el journal del servicio si falla.

Después del despliegue ejecuta una inferencia autenticada sobre un frame sintético. Este smoke test crea y elimina una tracking session, valida el contrato del batch y no revela el token:

```bash
./ansible-run ansible-playbook playbooks/smoke.yml
```

El token compartido se genera una sola vez en `~/.ansible/swimtrack-ai/auth_token`, queda con permisos `0600` sobre el filesystem Linux y nunca se imprime ni se incluye en Git. No se guarda dentro de `/mnt/c` porque ese montaje WSL no garantiza permisos POSIX restrictivos. El rol del Front lo lee desde el controller y lo instala en su environment file `0600`; no lo copies manualmente, no lo envíes por chat ni lo guardes en archivos versionados. Para un controller compartido, reemplaza este mecanismo por Ansible Vault o un secret manager.

La configuración del inventario publica AI en `10.0.218.101:7001`, dentro del rango ya abierto de la VM. El deploy no modifica UFW ni requiere privilegios de administrador; el servicio sigue ejecutándose como `grupo1`. El token sigue siendo obligatorio en todas las rutas `/v1/*`, pero la red no queda restringida por origen: esta configuración es exclusiva de la VM temporal del proyecto.

```bash
./ansible-run ansible-playbook playbooks/deploy.yml
```

No publiques AI en `0.0.0.0` ni reutilices esta configuración fuera de la red privada y la VM temporal del proyecto.

## Operación

```bash
ssh -F ~/.ssh/config proyecto_ia_gpu systemctl --user status swimtrack-ai.service
ssh -F ~/.ssh/config proyecto_ia_gpu journalctl --user --unit swimtrack-ai.service --follow
ssh -F ~/.ssh/config proyecto_ia 'curl --fail --show-error http://10.0.218.101:7001/readyz'
```

Reejecutar el playbook es seguro: los modelos usan checksums, las dependencias solo se sincronizan cuando cambia `uv.lock`, el token se conserva y el servicio solo se reinicia ante cambios. Para simular una actualización ya desplegada:

```bash
./ansible-run ansible-playbook playbooks/deploy.yml --check --diff
```

La revisión desplegada se fija en `inventory/group_vars/gpu_hosts.yml`. Modifica ese SHA únicamente después de publicar y validar el commit correspondiente en `swimtrack-ai`.

## Frontend

El playbook del front fija una revisión publicada de `swimtrack-front`, crea un entorno `.venv` con `uv`, instala `requirements.txt`, genera una `FLASK_SECRET_KEY` privada en el controller y mantiene Gunicorn en un puerto loopback configurable. El inventario de `proyecto_ia` usa `127.0.0.1:7101` porque 7001 ya está ocupado en ese host. Configura `VISION_BASE_URL=http://10.0.218.101:7001`, carga el token compartido desde el controller y comprueba una sesión autenticada contra AI.

```bash
./ansible-run ansible-playbook playbooks/preflight-front.yml
./ansible-run ansible-playbook playbooks/deploy-front.yml
./ansible-run ansible-playbook playbooks/smoke-front.yml
```

`swimtrack_front_revision` es el único pin inmutable de release del Front. Actualízalo a un commit publicado en `inventory/group_vars/front_hosts.yml` antes de desplegar; `deploy-front.yml` hace checkout de ese commit. `deploy-worktree.yml` ya no copia archivos fuente locales al checkout remoto: valida que el checkout local limpio de `main` sea el commit publicado configurado por ese pin y después invoca el mismo rol estándar.

El servicio se prepara con `URL_PREFIX=/swimtrack/`. Para publicarlo, ejecuta el playbook dedicado. Actualiza el único mapeo existente de `/swimtrack/` en el VirtualHost activo, valida Apache antes de recargarlo y no agrega otro sitio ni un `ProxyPass` duplicado. El host ya concede a `grupo1` `sudo` sin contraseña para esta operación específica.

```bash
./ansible-run ansible-playbook playbooks/publish-front.yml
```

El proxy conserva el prefijo al reenviar a Gunicorn y permite hasta diez minutos para un stream de video:

```apache
ProxyPass        /swimtrack/  http://127.0.0.1:7101/swimtrack/ connectiontimeout=5 timeout=600
ProxyPassReverse /swimtrack/ http://127.0.0.1:7101/swimtrack/
```

## Prueba E2E publicada

La prueba E2E genera temporalmente un clip de dos segundos y 20 frames desde `input_vids/20260513_201705_test01.mp4`, lo sube a `http://127.0.0.1/swimtrack/api/detect` en `proyecto_ia` y valida la cadena Apache → Gunicorn/Flask → AI GPU → SSE. No forma parte del despliegue normal porque realiza inferencia real y TensorRT serializa ese trabajo.

```bash
./ansible-run ansible-playbook playbooks/e2e-front.yml
```

El contrato de transporte y los resultados que se pueden exigir hoy están en `e2e/reference-video.yml`. La prueba exige SSE, un evento por frame, timestamps, dimensiones, cajas válidas y el conteo acumulado legacy de IDs. Para un video con `--expected-confirmed-identities`, además exige `identity_summary` válido en todos los frames y que su máximo y valor final sean el número de personas físicas esperado. `known_good_run` conserva una observación aprobada de la primera corrida real sin convertirla en una regla de producto prematuramente.

## Evaluación temporal de vueltas

`e2e/lap-ground-truth.yml` transcribe las anotaciones de `input_vids/TIMESTAMPS.md` para los nueve videos y fija sus checksums, metadatos, intervalos activos y eventos de vuelta. Los videos `test01`–`test08` forman el conjunto primario. `test09` se conserva como conjunto secundario porque tiene dos nadadores en el mismo carril y queda fuera del agregado primario.

La anotación tiene una incertidumbre de ±1 s y una vuelta dura aproximadamente 2 s. Por eso cada vuelta se representa con un intervalo nominal de ±1 s y se acepta una predicción hasta ±2 s de su timestamp central: 1 s de media duración más 1 s de incertidumbre. Las predicciones que incluyen `candidate_episode_id` se agrupan por visita física a la pared; los streams legacy sin ese campo se deduplican en una ventana de 2 s. Los eventos resultantes se asignan uno a uno por carril, maximizando TP y minimizando el error temporal absoluto. El evaluador informa TP, FP, FN, precision, recall y F1; no inventa TN para un problema de detección de eventos.

Para evaluar uno o más streams con un threshold explícito:

```bash
uv run --script scripts/evaluate_lap_events.py --threshold 0.20 --stream test01=../results/test01/stream.sse --stream test06=../results/test06/stream.sse --output ../results/lap-events-020.json
```

`aggregate_primary.complete` sólo será `true` cuando el comando incluya `test01`–`test08`; el resultado de `test09` aparece en `aggregate_all`, pero no modifica las métricas primarias.

El ajuste realizado con estas anotaciones, la curva exploratoria de `trajectory-v5`, sus limitaciones y la validación live final están documentados en `e2e/lap-timestamp-report.md`.

### Métricas por ventanas

`scripts/evaluate_lap_windows.py` reduce primero cada `(video_id,lane_id,candidate_episode_id)` a su score máximo y después genera una fila por ventana half-open. `X`, stride, anchor, coverage threshold y confidence threshold son parámetros explícitos. Una ventana con coverage insuficiente produce `abstain`, no TN. El reporte separa la matriz strict de la variante temporal-tolerant y publica TP, TN, FP, FN, precision, recall, F1, specificity, balanced accuracy, MCC, coverage y abstention rate.

```bash
uv run --script scripts/evaluate_lap_windows.py --window-size-ms 2000 --stride-ms 2000 --anchor-ms 0 --coverage-threshold 0.5 --threshold 0.05 --sweep-threshold 0.03 --sweep-threshold 0.07 --sweep-threshold 0.08 --expected-score-version trajectory-v5 --stream test01=../results/test01/stream.sse --output ../results/lap-windows-x2000.json --rows-output ../results/lap-windows-x2000.jsonl
```

El mismo dataset threshold-independent se reutiliza internamente para todo el sweep. El baseline completo, los artefactos persistentes y las limitaciones de development están documentados en `e2e/lap-window-report.md`. Ningún resultado de ese reporte fija un threshold de producto.

### Recalcular lap scores sin GPU

Un stream histórico ya contiene los timestamps, dimensiones y `boxes` inferidos por GPU. `replay_lap_scores.py` conserva esos datos y todas las demás claves de cada cuadro, reemplaza únicamente `lap_scores` mediante el `LapAnalyzer` de un checkout local de `swimtrack-ai` y genera otro stream SSE aceptado directamente por el evaluador. No ejecuta RT-DETRv2, TensorRT ni ByteTrack. Si el stream fue capturado con `tracking_diagnostics=boxes`, el replay también reconstruye el fallback de los scorers actuales a las detecciones dentro del ROI cuando ByteTrack no tiene un track activo; el nivel `counts` no contiene coordenadas suficientes para hacerlo.

```bash
uv run --script scripts/replay_lap_scores.py --stream ../results/test06/stream.sse --output ../results/test06/stream-trajectory-v5.sse --ai-source ../swimtrack-ai
uv run --script scripts/evaluate_lap_events.py --threshold 0.20 --stream test06=../results/test06/stream-trajectory-v5.sse --output ../results/test06/evaluation-trajectory-v5.json
```

El replay infiere el `fps` desde la mediana de los deltas de `time`. Usa `--fps 60` si el stream tiene un solo cuadro o si se quiere fijar el valor explícitamente, y `--calibration-id` para seleccionar otra calibración soportada por el código AI indicado.
