# Copyright 2024 MemAdaptor — optional Slurm + Apptainer launch for memory VDB.
"""Submit a batch job on another node to run ``python -m agent_system.memory.local_service``."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import textwrap
import time
from typing import Any

import requests
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _append_vdb_probe_log(memory_dir: str, line: str) -> None:
    """Append one line to a local file for health-check diagnostics (TrainRunner side)."""
    path = os.path.join(memory_dir, "vdb_health_probe.log")
    try:
        with open(path, "a", encoding="utf-8") as fp:
            fp.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {line}\n")
    except OSError:
        pass


def _as_plain_dict(memory_config: DictConfig) -> dict[str, Any]:
    if OmegaConf.is_config(memory_config):
        return OmegaConf.to_container(memory_config, resolve=True)
    return dict(memory_config)


def start_memory_vdb_on_slurm_node(
    memory_config: DictConfig,
    memory_dir: str,
    spawn_config: dict[str, Any],
    startup_timeout: int,
    log_path: str,
) -> "LocalMemoryServerHandle":
    """Write spawn JSON + sbatch script, ``sbatch`` it, wait for health, return handle (scancel on close)."""
    if not shutil.which("sbatch"):
        raise RuntimeError(
            "remote_slurm_launch.enable=true requires `sbatch` on PATH. "
            "This runs inside the Ray job worker: if that node has no Slurm client, either run training "
            "from a submit host that schedules workers with sbatch available, or set "
            "env.memory.auto_start_server=false and start the VDB job manually, then set env.memory.vdb_base_url."
        )

    plain = _as_plain_dict(memory_config)
    rl = plain.get("remote_slurm_launch") or {}
    if not isinstance(rl, dict):
        rl = {}

    partition = rl.get("partition") or ""
    if not str(partition).strip():
        raise ValueError("remote_slurm_launch.partition is required when remote_slurm_launch.enable=true")

    cpus = int(rl.get("cpus_per_task") or 4)
    mem = str(rl.get("mem") or "32G")
    account = rl.get("account")
    ex_raw = rl.get("exclude_nodes")
    if isinstance(ex_raw, (list, tuple)):
        exclude_nodes = ",".join(str(x).strip() for x in ex_raw if str(x).strip())
    else:
        exclude_nodes = str(ex_raw or "").strip()
    gres = rl.get("gres")
    apptainer_sif = str(rl.get("apptainer_sif") or "").strip()
    apptainer_binds = str(rl.get("apptainer_binds") or "/mnt:/mnt").strip()
    apptainer_nv = bool(rl.get("apptainer_nv", True))
    conda_sh = str(rl.get("conda_sh") or "").strip() or os.path.expanduser(
        "~/miniconda3/etc/profile.d/conda.sh"
    )
    conda_env = str(rl.get("conda_env") or "").strip()
    if not conda_env:
        conda_env = os.environ.get("CONDA_DEFAULT_ENV", "verl-agent")

    repo_root = str(rl.get("repo_root") or "").strip() or os.environ.get("MEMADAPTOR_REPO_ROOT") or REPO_ROOT

    sbatch_extra = rl.get("sbatch_extra_lines") or []
    if not isinstance(sbatch_extra, list):
        sbatch_extra = []

    scancel_on_close = bool(rl.get("scancel_on_close", True))
    os.makedirs(memory_dir, exist_ok=True)

    config_json_path = os.path.join(memory_dir, "memory_server_spawn_config.json")
    with open(config_json_path, "w", encoding="utf-8") as f:
        json.dump(spawn_config, f, indent=2)

    ip_file = os.path.join(memory_dir, "vdb_node_ip.txt")
    for path in (ip_file,):
        if os.path.isfile(path):
            try:
                os.remove(path)
            except OSError:
                pass

    wrapper_sh = os.path.join(memory_dir, "run_memory_vdb_inner.sh")
    wrapper_body = textwrap.dedent(
        f"""\
        #!/bin/bash
        set -eo pipefail
        # conda 的 activate/deactivate 钩子在 nounset 下会引用未定义变量而失败
        set +u
        source "{conda_sh}"
        conda activate "{conda_env}"
        set -u
        cd "{repo_root}"
        exec python -m agent_system.memory.local_service --config-json "{config_json_path}"
        """
    )
    with open(wrapper_sh, "w", encoding="utf-8") as f:
        f.write(wrapper_body)
    os.chmod(wrapper_sh, 0o755)

    bind_opts = ""
    if apptainer_binds:
        for part in apptainer_binds.split():
            part = part.strip()
            if part:
                bind_opts += f" -B {part}"

    nv_flag = "--nv" if apptainer_nv else ""

    if apptainer_sif:
        launcher_inner = f'apptainer exec {nv_flag}{bind_opts} "{apptainer_sif}" bash "{wrapper_sh}"'
    else:
        launcher_inner = f'bash "{wrapper_sh}"'

    sbatch_lines = [
        "#!/bin/bash",
        "#SBATCH -J e05-vdb",
        f"#SBATCH -p {partition}",
        "#SBATCH -N 1",
        "#SBATCH -n 1",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --mem={mem}",
    ]
    sbatch_lines.extend(
        [
            f"#SBATCH -o {memory_dir}/slurm_vdb-%j.out",
            f"#SBATCH -e {memory_dir}/slurm_vdb-%j.err",
        ]
    )
    if account:
        sbatch_lines.append(f"#SBATCH -A {account}")
    if exclude_nodes:
        sbatch_lines.append(f"#SBATCH --exclude={exclude_nodes}")
    if gres:
        sbatch_lines.append(f"#SBATCH --gres={gres}")
    for line in sbatch_extra:
        line = str(line).strip()
        if line:
            sbatch_lines.append(line)

    sbatch_body = "\n".join(sbatch_lines) + "\n\nset -eo pipefail\n"
    sbatch_body += textwrap.dedent(
        f"""\
        IP_FILE="{ip_file}"
        hostname -I | awk '{{print $1}}' > "$IP_FILE"
        {launcher_inner}
        """
    )

    sbatch_path = os.path.join(memory_dir, "launch_memory_vdb.sbatch")
    with open(sbatch_path, "w", encoding="utf-8") as f:
        f.write(sbatch_body)

    res = subprocess.run(
        ["sbatch", sbatch_path],
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0:
        raise RuntimeError(f"sbatch failed: {res.stderr or res.stdout}")

    job_id = None
    for token in (res.stdout or "").replace("\n", " ").split():
        if token.isdigit():
            job_id = token
            break
    if not job_id:
        raise RuntimeError(f"Could not parse Slurm job id from sbatch output: {res.stdout!r}")

    port = int(spawn_config["port"])
    deadline = time.time() + max(1, int(rl.get("startup_timeout") or startup_timeout))
    health_ok = False
    base_url = ""
    last_err: str | BaseException | None = None

    probe_log = os.path.join(memory_dir, "vdb_health_probe.log")
    try:
        with open(probe_log, "w", encoding="utf-8") as fp:
            fp.write(
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} sbatch_ok job_id={job_id} "
                f"port={port} timeout_s={max(1, int(rl.get('startup_timeout') or startup_timeout))}\n"
            )
            fp.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} probe_log={probe_log}\n")
    except OSError:
        pass

    last_no_ip_log = 0.0
    while time.time() < deadline:
        if not os.path.isfile(ip_file):
            now = time.time()
            if now - last_no_ip_log >= 10.0:
                _append_vdb_probe_log(memory_dir, f"wait vdb_node_ip.txt (not present yet) job_id={job_id}")
                last_no_ip_log = now
            time.sleep(1)
            continue
        try:
            with open(ip_file, "r", encoding="utf-8") as f:
                node_ip = f.read().strip().split()[0]
        except (OSError, IndexError):
            time.sleep(1)
            continue
        if not node_ip:
            time.sleep(1)
            continue
        base_url = f"http://{node_ip}:{port}"
        health_url = f"{base_url}/health"
        try:
            # Bypass HTTP(S)_PROXY: cluster compute IPs must not go through Squid (often HTTP 403).
            with requests.Session() as sess:
                sess.trust_env = False
                r = sess.get(health_url, timeout=3)
            if r.status_code == 200:
                health_ok = True
                _append_vdb_probe_log(memory_dir, f"OK GET {health_url} -> 200")
                break
            snippet = (r.text or "")[:200].replace("\n", " ")
            last_err = f"HTTP {r.status_code} from {health_url} body_prefix={snippet!r}"
            _append_vdb_probe_log(memory_dir, last_err)
        except requests.RequestException as exc:
            last_err = exc
            _append_vdb_probe_log(memory_dir, f"GET {health_url} failed: {type(exc).__name__}: {exc}")
        try:
            sq = subprocess.run(
                ["squeue", "-j", job_id, "-h"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if sq.returncode == 0 and not (sq.stdout or "").strip():
                raise RuntimeError(
                    f"Slurm job {job_id} no longer in queue before VDB became healthy. "
                    f"Check {memory_dir}/slurm_vdb-*.out / .err. Last health error: {last_err}"
                )
        except FileNotFoundError:
            pass
        time.sleep(2)

    if not health_ok:
        if scancel_on_close:
            subprocess.run(["scancel", job_id], capture_output=True)
        hint = (
            f"Timed out waiting for remote memory VDB at expected URL (port {port}). "
            f"Last error: {last_err!s}. Probe log: {probe_log}. "
            f"Slurm logs: {memory_dir}/slurm_vdb-*.out .err. Job id: {job_id}"
        )
        raise RuntimeError(hint)

    from .local_service import LocalMemoryServerHandle

    return LocalMemoryServerHandle(
        process=None,
        base_url=base_url,
        log_path=log_path,
        slurm_job_id=job_id if scancel_on_close else None,
    )
