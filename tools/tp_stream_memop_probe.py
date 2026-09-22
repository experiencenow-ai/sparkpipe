import argparse
import pathlib
import subprocess


def main():
    parser = argparse.ArgumentParser(
        description="Compile or explicitly run the isolated CUDA stream-memory-wait qualification probe. No production support is implied.")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--compile", action="store_true", help="Compile only; creates no CUDA context")
    action.add_argument("--run", action="store_true", help="Create a CUDA context and measure graph parameter update cost")
    parser.add_argument("--gpu-waits", action="store_true", help="With --run, launch three delayed-readiness/cancel/recovery checks")
    parser.add_argument("--cuda-root", type=pathlib.Path, help="CUDA toolkit root, required for --compile")
    parser.add_argument("--arch", help="Explicit GPU compile target, for example sm_121a")
    root = pathlib.Path(__file__).resolve().parents[1]
    parser.add_argument("--output", type=pathlib.Path, default=root / "build/qualification/tp_stream_memop_probe")
    args = parser.parse_args()
    if not args.compile and not args.run:
        if args.gpu_waits:
            parser.error("--gpu-waits requires --run")
        parser.print_help()
        return 0
    if args.gpu_waits and not args.run:
        parser.error("--gpu-waits requires --run")
    output = args.output.resolve()
    if args.compile:
        if args.cuda_root is None or not args.arch:
            parser.error("--compile requires --cuda-root and --arch")
        compiler = args.cuda_root.resolve() / "bin/nvcc"
        if not compiler.is_file():
            parser.error(f"CUDA compiler does not exist: {compiler}")
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [str(compiler), "-std=c++17", f"-arch={args.arch}", "-Xcompiler", "-pthread",
                   str(root / "tools/tp_stream_memop_probe.cu"), "-lcuda", "-o", str(output)]
    else:
        if args.cuda_root is not None or args.arch:
            parser.error("--cuda-root and --arch apply only to --compile")
        if not output.is_file():
            parser.error(f"compile the probe first; missing binary: {output}")
        command = [str(output), "--run"]
        if args.gpu_waits:
            command.append("--gpu-waits")
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
