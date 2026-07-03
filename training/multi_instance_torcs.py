# Parallel TORCS process manager for Windows (legacy multi-instance era).
# Per-instance config dirs, port patching, health checks, GUI automation.
# Modes: --mode setup | test | kill.

from __future__ import annotations

import os
import sys
import time
import shutil
import socket
import subprocess
import threading
import queue
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

from config import Config, MultiInstanceConfig, TORCS_EXE, TORCS_CONFIG_DIR


# ─────────────────────────────────────────────────────────────────────
#  SCR Server port patching
# ─────────────────────────────────────────────────────────────────────

# In TORCS with SCR patch, the server port is configured in the robot XML.
# On Windows, the relevant file is typically in:
#   <torcs_dir>\config\raceman\practice.xml  (or via command-line arg)
#   or in the scr_server robot config.
#
# The cleanest approach: TORCS SCR server accepts `wtorcs.exe -p <port>`
# (if built with that flag) OR reads from a config file.
# We support BOTH approaches and fall back gracefully.

SCR_SERVER_CONFIG_PATHS = [
    # Relative to the TORCS install dir — try these locations in order
    r"config\scr_server\default.xml",
    r"config\raceman\practice.xml",
    r"drivers\scr_server\scr_server.xml",
]

def _find_scr_config(torcs_dir: str) -> Optional[str]:
    # Find the SCR server XML config file in a TORCS install dir.
    for rel_path in SCR_SERVER_CONFIG_PATHS:
        full = os.path.join(torcs_dir, rel_path)
        if os.path.exists(full):
            return full
    return None


def _patch_xml_port(xml_path: str, new_port: int) -> bool:
    # Patch the UDP port value in an SCR server XML config.
    # Returns True if patch was applied, False if port attr not found.
    #
    # TORCS SCR XML structure (common patterns):
    # <attnum name="port" val="3001"/>
    # OR as a child of <section name="scr_server">
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        patched = False
        for elem in root.iter("attnum"):
            if elem.get("name") in ("port", "serverport", "scr_port"):
                elem.set("val", str(new_port))
                patched = True
        if patched:
            tree.write(xml_path, encoding="unicode", xml_declaration=True)
        return patched
    except Exception as e:
        print(f"[MultiInstance] XML patch failed for {xml_path}: {e}")
        return False


def _is_port_available(port: int) -> bool:
    # Quick check if a UDP port is free.
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind(("127.0.0.1", port))
        s.close()
        return True
    except OSError:
        return False


# ─────────────────────────────────────────────────────────────────────
#  Instance slot
# ─────────────────────────────────────────────────────────────────────

@dataclass
class InstanceSlot:
    idx:       int
    port:      int
    inst_dir:  str
    process:   Optional[subprocess.Popen] = field(default=None, repr=False)
    autostart: Optional[subprocess.Popen] = field(default=None, repr=False)
    in_use:    bool = False
    last_use:  float = 0.0
    crash_count: int = 0

    @property
    def is_alive(self) -> bool:
        if self.process is None:
            return False
        return self.process.poll() is None

    def kill(self):
        # Terminate the TORCS process.
        for proc in (self.autostart, self.process):
            if proc is not None:
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
        self.process   = None
        self.autostart = None


# ─────────────────────────────────────────────────────────────────────
#  One-time setup: clone TORCS install directories
# ─────────────────────────────────────────────────────────────────────

def setup_instances(
    n_instances:   int    = None,
    base_port:     int    = None,
    torcs_base_dir: str   = None,
    force_rebuild: bool   = False,
    verbose:       bool   = True,
) -> List[str]:
    # Create N TORCS instance directories (one-time setup).
    #
    # For each instance i:
    # - Copies D:\torcs\torcs → D:\torcs\torcs_inst{i}  (if not exists)
    # - Patches the SCR server port to base_port + i in each clone's XML
    #
    # Returns: list of instance directory paths.
    #
    # NOTE: Cloning is skipped if the directory already exists unless force_rebuild=True.
    # This function is SAFE to call multiple times — it's idempotent.
    cfg    = Config.multi
    n      = n_instances   or cfg.n_instances
    port0  = base_port     or cfg.base_port
    src    = torcs_base_dir or cfg.torcs_base_dir

    if not os.path.isdir(src):
        raise FileNotFoundError(
            f"[MultiInstance] TORCS base directory not found: {src}\n"
            f"Expected the TORCS install at {TORCS_EXE}"
        )

    inst_dirs = []

    for i in range(n):
        port     = port0 + i
        inst_dir = cfg.instance_dir_pattern.format(idx=i)

        if verbose:
            print(f"[MultiInstance] Instance {i}: dir={inst_dir!r}  port={port}")

        if os.path.isdir(inst_dir) and not force_rebuild:
            if verbose:
                print(f"  → Already exists, skipping copy.")
        else:
            if os.path.isdir(inst_dir):
                shutil.rmtree(inst_dir)
            if verbose:
                print(f"  → Copying from {src!r} …")
            shutil.copytree(src, inst_dir, dirs_exist_ok=False)
            if verbose:
                print(f"  → Copy complete.")

        # Patch port in XML config files
        patched_any = False
        for rel_path in SCR_SERVER_CONFIG_PATHS:
            xml_path = os.path.join(inst_dir, rel_path)
            if os.path.exists(xml_path):
                ok = _patch_xml_port(xml_path, port)
                if ok:
                    patched_any = True
                    if verbose:
                        print(f"  → Patched port {port} in {rel_path}")

        # Inject custom headless practice.xml
        custom_practice_xml = os.path.join(os.path.dirname(os.path.dirname(__file__)), "practice.xml")
        target_practice_xml = os.path.join(inst_dir, "config", "raceman", "practice.xml")
        if os.path.exists(custom_practice_xml):
            os.makedirs(os.path.dirname(target_practice_xml), exist_ok=True)
            shutil.copy2(custom_practice_xml, target_practice_xml)
            # Make sure to patch the port in the newly copied file as well!
            _patch_xml_port(target_practice_xml, port)
            if verbose:
                print(f"  → Injected headless practice.xml")

        if not patched_any and verbose:
            print(f"  [WARN] Could not find/patch SCR server XML for instance {i}.")
            print(f"         Will rely on command-line port argument to wtorcs.exe.")

        inst_dirs.append(inst_dir)

    if verbose:
        print(f"\n[MultiInstance] Setup complete: {n} instances ready.")
        print(f"  Ports: {port0} … {port0 + n - 1}")
    return inst_dirs


# ─────────────────────────────────────────────────────────────────────
#  Instance pool
# ─────────────────────────────────────────────────────────────────────

class InstancePool:
    # Thread-safe pool of TORCS instances.
    #
    # Usage:
    # pool = InstancePool()
    # pool.start_all()
    #
    # slot = pool.acquire()          # blocks until a slot is free
    # try:
    # result = evaluate(slot.port)
    # finally:
    # pool.release(slot)
    #
    # pool.stop_all()

    def __init__(self, n_instances: int = None, base_port: int = None,
                 verbose: bool = True):
        cfg   = Config.multi
        self.n        = n_instances or cfg.n_instances
        self.port0    = base_port   or cfg.base_port
        self.cfg      = cfg
        self.verbose  = verbose
        self._slots:  List[InstanceSlot] = []
        self._sem     = threading.Semaphore(0)  # released as slots become available
        self._lock    = threading.Lock()
        self._avail   = queue.Queue()           # available slot indices

    # ── Lifecycle ──────────────────────────────────────────────────

    def start_all(self):
        # Launch all N TORCS instances.
        # Ensure instance directories exist
        setup_instances(n_instances=self.n, base_port=self.port0,
                        verbose=self.verbose)

        for i in range(self.n):
            port     = self.port0 + i
            inst_dir = Config.multi.instance_dir_pattern.format(idx=i)
            slot     = InstanceSlot(idx=i, port=port, inst_dir=inst_dir)
            self._slots.append(slot)
            self._launch_slot(slot)

        # Wait for instances to fully start up
        wait = self.cfg.startup_wait_s
        if self.verbose:
            print(f"[Pool] Waiting {wait:.0f}s for TORCS instances to start...")
        time.sleep(wait)

        # Mark all as available
        for slot in self._slots:
            self._avail.put(slot.idx)
            self._sem.release()

        if self.verbose:
            print(f"[Pool] {self.n} TORCS instances ready on ports "
                  f"{self.port0}…{self.port0 + self.n - 1}")

    def stop_all(self):
        # Kill all TORCS instances.
        for slot in self._slots:
            slot.kill()
        if self.verbose:
            print("[Pool] All instances stopped.")

    # ── Slot management ─────────────────────────────────────────────

    def acquire(self, timeout: float = 600.0) -> InstanceSlot:
        # Block until a slot is free, then return it (marked as in_use=True).
        # Raises TimeoutError if no slot becomes available within timeout seconds.
        if not self._sem.acquire(timeout=timeout):
            raise TimeoutError("[Pool] Timed out waiting for a free TORCS slot.")
        idx  = self._avail.get()
        slot = self._slots[idx]
        with self._lock:
            slot.in_use   = True
            slot.last_use = time.time()
        # If the instance crashed, restart it
        if not slot.is_alive:
            self._restart_slot(slot)
        return slot

    def release(self, slot: InstanceSlot):
        # Return a slot to the pool. Restart the instance if it crashed.
        with self._lock:
            slot.in_use = False
        if not slot.is_alive:
            self._restart_slot(slot)
        self._avail.put(slot.idx)
        self._sem.release()

    # ── TORCS process management ─────────────────────────────────────

    def _launch_slot(self, slot: InstanceSlot):
        # Start a TORCS process for this slot.
        if slot.is_alive:
            return

        torcs_exe  = self.cfg.torcs_exe
        inst_dir   = slot.inst_dir
        port       = slot.port

        if self.verbose:
            print(f"[Pool] Starting instance {slot.idx} (port {port})...")

        # Build command line
        # wtorcs.exe on Windows often ignores -r, so we MUST use autostart_win.py
        cmd = [torcs_exe, f"-p{port}"]
        
        # Set the working dir to the instance dir so it reads its own config
        env = dict(os.environ)
        env["TORCS_DATA"] = inst_dir  # some TORCS builds use this env var

        try:
            proc = subprocess.Popen(
                cmd,
                cwd=inst_dir,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            slot.process = proc
        except FileNotFoundError:
            # Fallback if port argument fails
            cmd = [torcs_exe]
            proc = subprocess.Popen(
                cmd,
                cwd=inst_dir,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            slot.process = proc

        # Launch autostart (sends menu keystrokes to start practice race)
        time.sleep(1.0) # Wait for window to appear
        self._run_autostart(slot)

    def _run_autostart(self, slot: InstanceSlot):
        # Run the autostart script to navigate TORCS menus to start the race.
        python_exe = sys.executable
        autostart_script = os.path.join(os.path.dirname(os.path.dirname(__file__)), "autostart_win.py")
        if not os.path.exists(autostart_script):
            if self.verbose:
                print(f"  [WARN] autostart_win.py not found, TORCS may need manual start.")
            return

        try:
            proc = subprocess.Popen(
                [python_exe, autostart_script],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            slot.autostart = proc
            # We MUST wait for the autostart keystrokes to finish before we return!
            # Otherwise the next iteration of start_all() will launch another TORCS
            # window which will steal focus and break the keystrokes.
            # autostart_win.py needs ~14-16s before it even sends keys (internal
            # sleep(10) + focus + settle(2.5) + ~2.4s of keystrokes), so a 10s
            # timeout killed it BEFORE any key was sent → race never started.
            proc.wait(timeout=40.0)
        except subprocess.TimeoutExpired:
            if self.verbose:
                print(f"  [WARN] Autostart timeout for slot {slot.idx}")
            proc.kill()
        except Exception as e:
            if self.verbose:
                print(f"  [WARN] Autostart failed for slot {slot.idx}: {e}")

    def _restart_slot(self, slot: InstanceSlot):
        # Kill and restart a crashed/hung TORCS instance.
        slot.kill()
        slot.crash_count += 1
        if self.verbose:
            print(f"[Pool] Restarting instance {slot.idx} "
                  f"(crash #{slot.crash_count})...")
        time.sleep(2.0)
        self._launch_slot(slot)
        # Wait for it to come up
        time.sleep(self.cfg.startup_wait_s)


# ─────────────────────────────────────────────────────────────────────
#  Parallel evaluator (for Optuna)
# ─────────────────────────────────────────────────────────────────────

class ParallelEvaluator:
    # Evaluates multiple teacher parameter sets simultaneously using the pool.
    #
    # eval_fn(params, port) → result_dict
    # Called by a worker thread for each slot.
    #
    # Usage in Optuna:
    # pool = InstancePool()
    # pool.start_all()
    # evaluator = ParallelEvaluator(pool, eval_fn=evaluate_teacher)
    #
    # # Submit N evaluations (non-blocking)
    # futures = [evaluator.submit(params_i) for params_i in batch]
    # results = [f.result() for f in futures]    # wait for all
    #
    # pool.stop_all()

    def __init__(self, pool: InstancePool,
                 eval_fn: Callable,
                 max_workers: int = None):
        self._pool     = pool
        self._eval_fn  = eval_fn
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers or pool.n
        )

    def submit(self, params, **kwargs) -> Future:
        # Submit one evaluation asynchronously. Returns a Future.
        return self._executor.submit(self._run_one, params, kwargs)

    def _run_one(self, params, kwargs: dict):
        slot = self._pool.acquire(timeout=600.0)
        try:
            return self._eval_fn(params, port=slot.port, **kwargs)
        except Exception as e:
            print(f"[ParallelEval] Error on port {slot.port}: {e}")
            traceback.print_exc()
            return {"best_lap": float("inf"), "avg_lap": float("inf"),
                    "laps": 0, "max_dist": 0.0}
        finally:
            self._pool.release(slot)

    def shutdown(self):
        self._executor.shutdown(wait=False)


# ─────────────────────────────────────────────────────────────────────
#  CLI: setup and smoke-test
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Multi-Instance TORCS Manager — setup and smoke test"
    )
    ap.add_argument("--mode", choices=["setup", "test", "stop"],
                    default="setup",
                    help="setup: create instance dirs | test: launch+verify | stop: kill all")
    ap.add_argument("--n", type=int, default=None,
                    help="Number of instances (default: Config.multi.n_instances)")
    ap.add_argument("--force", action="store_true",
                    help="Force re-clone instance directories even if they exist")
    args = ap.parse_args()

    n = args.n or Config.multi.n_instances

    if args.mode == "setup":
        print(f"\n[Setup] Creating {n} TORCS instance directories...")
        dirs = setup_instances(n_instances=n, force_rebuild=args.force, verbose=True)
        print(f"\n[Setup] Done. Directories:")
        for d in dirs:
            print(f"  {d}")
        print("\nNext step: run with --mode test to verify instances start correctly.")

    elif args.mode == "test":
        print(f"\n[Test] Launching {n} TORCS instances and verifying connectivity...")
        pool = InstancePool(n_instances=n, verbose=True)
        pool.start_all()

        print(f"\n[Test] Acquiring all {n} slots...")
        slots = []
        for i in range(n):
            slot = pool.acquire(timeout=30.0)
            slots.append(slot)
            print(f"  Acquired slot {slot.idx} (port {slot.port}) — alive={slot.is_alive}")

        print("\n[Test] Releasing all slots...")
        for slot in slots:
            pool.release(slot)

        print("\n[Test] Stopping all instances...")
        pool.stop_all()
        print("[Test] Done. If all slots showed alive=True, multi-instance is working.")

    elif args.mode == "stop":
        print("[Stop] Killing all TORCS instances...")
        import psutil
        torcs_name = os.path.basename(Config.multi.torcs_exe).lower()
        killed = 0
        for proc in psutil.process_iter(["name", "pid"]):
            try:
                if torcs_name in proc.info["name"].lower():
                    proc.terminate()
                    killed += 1
            except Exception:
                pass
        print(f"[Stop] Killed {killed} TORCS processes.")
