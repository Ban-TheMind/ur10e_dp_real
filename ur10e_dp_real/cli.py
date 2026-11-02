"""Small command surface; hardware imports are lazy so training needs no ROS."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="UR10e Lift: collect, check, fine-tune and run DP"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("cameras")
    for name in ["collect", "observe", "run", "gripper-activate"]:
        p = commands.add_parser(name)
        p.add_argument("--site", required=True)
        if name in ("collect", "run"):
            p.add_argument("--output", required=True)
        if name == "run":
            p.add_argument("--checkpoint", required=True)
            p.add_argument("--device", default="cuda")
            p.add_argument("--inference-steps", type=int)
            p.add_argument("--duration", type=float, default=30)
    p = commands.add_parser("check")
    p.add_argument("root")
    p = commands.add_parser("summarize")
    p.add_argument("root")
    p = commands.add_parser("finetune")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--settings", default="configs/finetune.json")
    p.add_argument("--resume")
    p = commands.add_parser("model-check")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--inference-steps", type=int, default=2)
    args = parser.parse_args()
    if args.command == "cameras":
        import pyrealsense2 as rs

        context = rs.context()
        for camera in context.query_devices():
            print(
                json.dumps(
                    {
                        "name": camera.get_info(rs.camera_info.name),
                        "serial": camera.get_info(rs.camera_info.serial_number),
                    }
                )
            )
    elif args.command in ("collect", "observe", "run", "gripper-activate"):
        cfg = json.loads(Path(args.site).read_text())
        from .geometry import PolicyFrame

        PolicyFrame(cfg["policy_frame"])
        if cfg["sample_hz"] != 20:
            raise ValueError(
                "Source Lift contract is 20Hz; explicitly adapt temporal horizons before changing it"
            )
        if args.command == "gripper-activate":
            from .devices import Gripper

            gripper = object.__new__(Gripper)
            gripper.cfg = cfg["gripper"]
            print(gripper.transact("activate"))
        else:
            from . import runtime

            if args.command == "observe":
                runtime.inspect(cfg)
            elif args.command == "collect":
                runtime.collect(cfg, args.output)
            else:
                runtime.rollout(
                    cfg,
                    args.checkpoint,
                    args.output,
                    args.device,
                    args.inference_steps,
                    args.duration,
                )
    elif args.command == "check":
        from .recording import check_episode

        paths = sorted(Path(args.root).glob("episode_*"))
        if not paths:
            parser.error("No episodes found")
        for path in paths:
            print(json.dumps(check_episode(path)))
    elif args.command == "summarize":
        episodes = [
            json.loads(p.read_text())
            for p in Path(args.root).glob("episode_*/episode.json")
        ]
        evaluations = [
            m for m in episodes if m["kind"] == "evaluation" and m["complete"]
        ]
        labeled = [m for m in evaluations if m["success"] is not None]
        print(
            json.dumps(
                {
                    "episodes": len(evaluations),
                    "labeled": len(labeled),
                    "successes": sum(m["success"] for m in labeled),
                    "success_rate": (
                        sum(m["success"] for m in labeled) / len(labeled)
                        if labeled
                        else None
                    ),
                },
                indent=2,
            )
        )
    elif args.command == "finetune":
        from .training import train

        train(
            args.checkpoint,
            args.data,
            args.output,
            json.loads(Path(args.settings).read_text()),
            args.resume,
        )
    elif args.command == "model-check":
        import torch
        from .model import load_policy

        model, config, _ = load_policy(
            args.checkpoint, args.device, inference_steps=args.inference_steps
        )
        obs = {
            k: torch.zeros(1, model.n_obs_steps, *v["shape"], device=args.device)
            for k, v in config["shape_meta"]["obs"].items()
        }
        with torch.inference_mode():
            result = model.predict_action(obs)["action"]
        print(
            json.dumps(
                {
                    "action_shape": list(result.shape),
                    "finite": bool(torch.isfinite(result).all()),
                    "inference_steps": model.num_inference_steps,
                }
            )
        )


if __name__ == "__main__":
    main()
