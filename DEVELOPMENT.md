# Development notes

## AI/dev container (bind mount)

This setup keeps your repo on the host (good for Emacs) and mounts it into a container that has Codex/Claude installed.

Build and start:
```bash
USER_UID=$(id -u) USER_GID=$(id -g) docker compose -f docker-compose.ai.yml up -d --build
```

Enter the container:
```bash
docker compose -f docker-compose.ai.yml exec ai bash
```

The first start installs `.[dev,qbf]` into a venv at `/home/developer/.venv`. You can override extras:
```bash
PYTHON_FBAS_EXTRAS=".[dev]" docker compose -f docker-compose.ai.yml up -d --build
```

Codex auth is shared from the host by bind-mounting `~/.codex` read-write. Log in once on the host, then restart the container:
```bash
codex login --device-auth
docker compose -f docker-compose.ai.yml up -d
```

To update dependencies, `rm ~/.python-fbas-container-setup` and re-run the container, or delete the venv at `~/.venv`.
