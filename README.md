# SwimTrack Ansible

Despliega `swimtrack-ai` en la máquina GPU accesible mediante el alias SSH `proyecto_ia_gpu`, sin Docker ni sudo. El playbook instala las dependencias con `uv`, copia el ONNX y su external data, limita el proceso a GPU 0 y mantiene Uvicorn mediante un servicio `systemd --user` persistente.

## Requisitos del controller

- Ejecutar desde Debian/WSL con el alias `proyecto_ia_gpu` definido en `~/.ssh/config`.
- Tener `swimtrack-ansible`, `swimtrack-ai` y `swimtrack-poc` como directorios hermanos.
- Conservar `swimtrack-poc/artifacts/models/rtdetrv2_s.onnx` y `rtdetrv2_s.onnx.data`.
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

El token compartido se genera una sola vez en `~/.ansible/swimtrack-ai/auth_token`, queda con permisos `0600` sobre el filesystem Linux y nunca se imprime ni se incluye en Git. No se guarda dentro de `/mnt/c` porque ese montaje WSL no garantiza permisos POSIX restrictivos. Para configurar `swimtrack-front`, copia manualmente su valor a `VISION_AUTH_TOKEN`; no envíes el token por chat ni lo guardes en archivos versionados. Para un controller compartido, reemplaza este mecanismo por Ansible Vault o un secret manager.

El servicio queda disponible solamente en `127.0.0.1:8001` de la máquina GPU. Desde la máquina del front abre el tunnel:

```bash
ssh -NT -F ~/.ssh/config -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -L 127.0.0.1:18001:127.0.0.1:8001 proyecto_ia_gpu
```

Configura `VISION_BASE_URL=http://127.0.0.1:18001` en el front.

## Operación

```bash
ssh -F ~/.ssh/config proyecto_ia_gpu systemctl --user status swimtrack-ai.service
ssh -F ~/.ssh/config proyecto_ia_gpu journalctl --user --unit swimtrack-ai.service --follow
curl --fail --show-error http://127.0.0.1:18001/readyz
```

Reejecutar el playbook es seguro: los modelos usan checksums, las dependencias solo se sincronizan cuando cambia `uv.lock`, el token se conserva y el servicio solo se reinicia ante cambios. Para simular una actualización ya desplegada:

```bash
./ansible-run ansible-playbook playbooks/deploy.yml --check --diff
```

La revisión desplegada se fija en `inventory/group_vars/gpu_hosts.yml`. Modifica ese SHA únicamente después de publicar y validar el commit correspondiente en `swimtrack-ai`.
