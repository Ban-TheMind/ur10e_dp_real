"""Run a visible training stage; the thread heartbeat releases evaluation gates."""
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "outputs/real_lift_20261107"
RUNS = {"finetune": ROOT / "outputs/lift_finetune_20261107_run01",
        "scratch": ROOT / "outputs/lift_scratch_20261107_run01"}


def state(phase, status, **values):
    dest = CONTROL / (phase + "_state.json")
    temp = dest.with_suffix(".tmp")
    temp.write_text(json.dumps(dict(phase=phase, status=status, timestamp=time.time(), **values), indent=2)+"\n")
    temp.replace(dest)


def wait_for(path, message):
    print(message, flush=True)
    last = 0
    while not path.exists():
        if time.monotonic()-last >= 60:
            print(time.strftime("%H:%M:%S"), message, flush=True)
            last = time.monotonic()
        time.sleep(5)


def execute(command, logfile):
    print("\n$ " + shlex.join(command) + "\n", flush=True)
    subprocess.run(["bash", "-o", "pipefail", "-c",
                    shlex.join(command)+" 2>&1 | tee "+shlex.quote(str(logfile))],
                   cwd=ROOT, check=True)


def main():
    phase = sys.argv[1]
    CONTROL.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    os.environ.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONUNBUFFERED="1")
    python = str(ROOT / ".venv_train/bin/python")
    state(phase, "waiting" if phase == "scratch" else "starting")
    if phase == "scratch":
        wait_for(CONTROL / "finetune_evaluation_passed.json",
                 "从零训练终端已就绪：等待微调完成并通过离线测试、图表导出。")
        summary = json.loads((RUNS["finetune"] / "analysis/summary.json").read_text())
        if summary["status"] != "passed":
            raise RuntimeError("Fine-tuning evaluation did not pass")
    with (CONTROL / "gpu.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        command = [python, "-u", "-m", "ur10e_dp_real.cli"]
        if phase == "finetune":
            command += ["finetune", "--checkpoint", str(ROOT / "weights/local_lift_3050.ckpt")]
        else:
            command += ["train", "--policy-config", str(ROOT / "configs/policy_rgb_transformer.json")]
        command += ["--data", str(ROOT / "data/lift_success"), "--output", str(RUNS[phase]),
                    "--settings", str(ROOT / ("configs/lift_finetune.json" if phase == "finetune" else "configs/lift_scratch.json"))]
        state(phase, "training", command=command)
        execute(command, CONTROL / (phase+"_train.log"))
        state(phase, "awaiting_evaluation")
        wait_for(CONTROL / ("evaluate_"+phase+".request"),
                 "训练完成，等待定时检查触发完整验证集测试和图表导出。")
        state(phase, "evaluating")
        execute([python, "-u", str(ROOT / "tools/analyze_real_training.py"), str(RUNS[phase])],
                CONTROL / (phase+"_evaluation.log"))
        summary = json.loads((RUNS[phase] / "analysis/summary.json").read_text())
        if summary["status"] != "passed":
            raise RuntimeError("Evaluation failed")
        gate = CONTROL / (phase+"_evaluation_passed.json")
        temp = gate.with_suffix(".tmp")
        temp.write_text(json.dumps(dict(status="passed", run=str(RUNS[phase]), timestamp=time.time()),indent=2)+"\n")
        temp.replace(gate)
        state(phase, "complete", run=str(RUNS[phase]))
    print("\n任务与图表已完成：", RUNS[phase], flush=True)


if __name__ == "__main__":
    try:
        main()
    except (Exception, KeyboardInterrupt) as exc:
        state(sys.argv[1], "failed", error=repr(exc))
        print("任务停止：", repr(exc), flush=True)
    input("\n按 Enter 关闭本终端。")
