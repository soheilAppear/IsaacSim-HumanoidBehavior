#!/usr/bin/env python3
"""Run Python inside a LIVE Isaac Sim session over TCP.

Isaac Sim ships ``isaacsim.code_editor.python_server``: a TCP server that executes
Python in the running Kit process and returns JSON. It is off by default. Start Isaac
Sim with ``tools/launch_isaac_vr.bat`` (which enables it), or turn it on from
Window > Extensions > "Python Server" in a session that is already open.

That turns questions which otherwise need a headset on someone's face --- "is the eye
device even there?", "what is the head pose right now?", "is the camera attached?" ---
into one command, and lets values be retuned without restarting a 7 GB app.

Usage:
    python tools/kit_exec.py "print(1 + 1)"
    python tools/kit_exec.py --file probe.py
    echo "print('hi')" | python tools/kit_exec.py -

Handy one-liners once the humanoid example is loaded and playing:

    # Which XR devices does the runtime actually expose?
    python tools/kit_exec.py "from omni.kit.xr.core import XRCore; \
print([d.get_name() for d in XRCore.get_singleton().get_all_input_devices()])"

    # Full first-person rig status (head pose, calibration, anchor offsets).
    python tools/kit_exec.py "print(EX.describe_xr())"

    # Gaze status: source, ray origin/direction, what it is hitting.
    python tools/kit_exec.py "print(EX._eye_gaze_tracker.describe())"

``EX`` is bound automatically to the running HumanoidExample instance when one is found.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8226

#: Bind ``EX`` to the live HumanoidExample so probes stay short. The example registers
#: itself with the browser, so it is reachable without the user having to find it.
PRELUDE = """
if "EX" not in globals() or EX is None:
    EX = None
    try:
        import gc
        from isaacsim.robot.policy.examples.interactive.humanoid.humanoid_example import HumanoidExample
        # isinstance() can dereference dead weak proxies in Kit's object graph.
        # Prefer a loaded example if the browser owns multiple sample instances.
        _examples = [obj for obj in gc.get_objects() if type(obj) is HumanoidExample]
        EX = next((obj for obj in _examples if getattr(obj, "g1", None) is not None), None)
        if EX is None and _examples:
            EX = _examples[0]
    except Exception:
        pass
"""


def execute(code: str, host: str, port: int, timeout: float) -> int:
    """Send *code* to the live Kit session and print the structured response.

    Returns:
        A process exit code: 0 when Kit reported success, 1 otherwise.
    """
    payload = json.dumps({"code": PRELUDE + "\n" + code}).encode("utf-8")
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            sock.settimeout(timeout)
            sock.sendall(payload)
            sock.shutdown(socket.SHUT_WR)  # the server executes on EOF, not on newline
            chunks = []
            while True:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                chunks.append(chunk)
    except ConnectionRefusedError:
        print(
            f"No Python server on {host}:{port}. Start Isaac Sim with\n"
            "    tools\\launch_isaac_vr.bat\n"
            "or enable 'Python Server' (isaacsim.code_editor.python_server) in\n"
            "Window > Extensions of the session that is already open.",
            file=sys.stderr,
        )
        return 2
    except OSError as error:
        print(f"Could not talk to Isaac Sim on {host}:{port}: {error}", file=sys.stderr)
        return 2

    raw = b"".join(chunks).decode("utf-8", errors="replace")
    try:
        response = json.loads(raw)
    except json.JSONDecodeError:
        print(raw)
        return 1

    output = response.get("output", "")
    if output:
        print(output, end="" if output.endswith("\n") else "\n")
    if "result" in response and response["result"] is not None:
        print(response["result"])
    if response.get("status") != "ok":
        print(f"{response.get('ename', 'Error')}: {response.get('evalue', '')}", file=sys.stderr)
        for line in response.get("traceback", []):
            print(line, file=sys.stderr)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("code", nargs="?", help="Python source to run in Isaac Sim; '-' reads stdin")
    parser.add_argument("--file", help="Read the source from this file instead")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--timeout", type=float, default=30.0, help="Socket timeout in seconds")
    args = parser.parse_args()

    if args.file:
        with open(args.file, encoding="utf-8") as handle:
            code = handle.read()
    elif args.code == "-":
        code = sys.stdin.read()
    elif args.code:
        code = args.code
    else:
        parser.error("give code as an argument, via --file, or as '-' for stdin")
    return execute(code, args.host, args.port, args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())
