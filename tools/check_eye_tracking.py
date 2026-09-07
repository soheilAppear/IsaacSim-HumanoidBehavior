#!/usr/bin/env python3
"""Check the Quest Pro eye-gaze chain end to end, without launching anything.

Every link in the chain leaves a trace on disk, so "is eye tracking reaching Isaac Sim?"
can be answered in a second instead of by putting the headset on and guessing:

    python tools/check_eye_tracking.py

The one link that actually breaks in practice is the headset telling Steam Link it has eye
tracking. Steam Link logs that explicitly, and Kit logs whether the resulting OpenXR
device ever showed up, so the two can be lined up against each other.
"""

from __future__ import annotations

import re
import sys
import time
from datetime import datetime
from pathlib import Path

STEAM_LOGS = Path(r"C:\Program Files (x86)\Steam\logs")
STEAMVR_SETTINGS = Path(r"C:\Program Files (x86)\Steam\config\steamvr.vrsettings")
KIT_LOGS = Path.home() / ".nvidia-omniverse" / "logs" / "Kit" / "Isaac-Sim XR VR" / "6.0"

OK, BAD, MEH = "  [ok]   ", "  [FAIL] ", "  [??]   "


def latest(directory: Path, pattern: str) -> Path | None:
    try:
        files = sorted(directory.glob(pattern), key=lambda f: f.stat().st_mtime)
    except OSError:
        return None
    return files[-1] if files else None


def read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def check_runtime() -> None:
    print("1. Active OpenXR runtime")
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Khronos\OpenXR\1") as key:
            runtime = winreg.QueryValueEx(key, "ActiveRuntime")[0]
    except Exception as error:
        print(MEH + f"could not read the registry: {error}")
        return
    if "steamxr" in runtime.lower():
        print(OK + f"SteamVR ({runtime})")
    else:
        print(BAD + f"not SteamVR: {runtime}")
        print("         SteamVR > Settings > OpenXR > Set SteamVR as OpenXR Runtime")


def check_share_setting() -> None:
    print("\n2. SteamVR 'share eye tracking data'")
    text = read(STEAMVR_SETTINGS)
    if not text:
        print(MEH + f"could not read {STEAMVR_SETTINGS}")
        return
    if re.search(r'"shareEyeTrackingData"\s*:\s*true', text):
        print(OK + "enabled")
    elif "shareEyeTrackingData" in text:
        print(BAD + "present but disabled -- turn it on in the Steam Link PC settings")
    else:
        print(MEH + "setting not present yet (it appears once Steam Link has run)")


def check_steam_link() -> None:
    print("\n3. Steam Link: did the headset offer eye tracking?")
    current = STEAM_LOGS / "driver_vrlink.txt"
    previous = STEAM_LOGS / "driver_vrlink.previous.txt"
    found_any = False
    for path in (current, previous):
        text = read(path)
        if not text:
            continue
        hits = re.findall(r"^(.*supports eye tracking, creating input component.*)$", text, re.M)
        for line in hits:
            found_any = True
            print(OK + line.strip())
    if not found_any:
        print(BAD + "no 'supports eye tracking, creating input component' line in either log")
        print("         The headset is not telling Steam Link it has eye tracking. On the")
        print("         Quest Pro: Settings > Movement Tracking > Eye Tracking ON, and")
        print("         re-grant Steam Link the eye tracking permission.")

    text = read(current)
    sessions = re.findall(r"^(\w{3} \w{3} \d+ \d{4} [\d:.]+).*Starting session with ID", text, re.M)
    shutdowns = re.findall(r"^(\w{3} \w{3} \d+ \d{4} [\d:.]+).*Performing shutdown", text, re.M)
    if sessions:
        print(f"         last Steam Link connect:  {sessions[-1]}")
    if shutdowns:
        print(f"         last Steam Link shutdown: {shutdowns[-1]}")


def check_kit() -> None:
    print("\n4. Isaac Sim: did the eye device reach Kit?")
    log = latest(KIT_LOGS, "kit_*.log")
    if log is None:
        print(MEH + f"no Kit logs under {KIT_LOGS}")
        return
    stamp = datetime.fromtimestamp(log.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    print(f"         newest session: {log.name}  ({stamp})")
    text = read(log)

    device_lines = re.findall(r"\[XR\] input devices (/user.*)$", text, re.M)
    if not device_lines:
        print(MEH + "no XR session in this log (VR was never started)")
        return
    devices = device_lines[-1].strip()
    print(f"         devices: {devices}")
    if "/user/eye/unified" in " ".join(device_lines):
        print(OK + "/user/eye/unified was created -- real eye gaze reached Isaac Sim")
    else:
        print(BAD + "/user/eye/unified never appeared -- the ray falls back to head direction")
        if "XR_EXT_eye_gaze_interaction" in text:
            print("         (the OpenXR extension IS enabled, so this is the headset/Steam")
            print("          Link side, not Isaac Sim's configuration)")

    for line in re.findall(r"\[EyeGaze\] ray source: (.*)$", text, re.M):
        print(f"         reported source: {line.strip()[:110]}")


def watch() -> int:
    """Tail the Steam Link log and report the moment the headset advertises eye tracking.

    The whole chain hinges on one line appearing in Steam Link's driver log. Watching for
    it live means the headset toggles can be fixed with immediate feedback, instead of
    launching a 7 GB application to find out.
    """
    path = STEAM_LOGS / "driver_vrlink.txt"
    print("Watching Steam Link for the eye-tracking handshake.")
    print(f"  {path}")
    print("\nNow, on the headset:")
    print("  1. Settings > Movement Tracking > Eye Tracking: ON (calibrate if offered)")
    print("  2. Settings > Privacy > App permissions > Eye tracking > allow Steam Link")
    print("  3. Quit Steam Link fully and reconnect it")
    print("\nWaiting for 'supports eye tracking, creating input component' ... Ctrl-C to stop.\n")

    try:
        position = path.stat().st_size
    except OSError:
        position = 0
    seen_connect = False
    try:
        while True:
            try:
                with path.open("r", encoding="utf-8", errors="replace") as handle:
                    handle.seek(position)
                    chunk = handle.read()
                    position = handle.tell()
            except OSError:
                time.sleep(1.0)
                continue
            for line in chunk.splitlines():
                if "Starting session with ID" in line:
                    seen_connect = True
                    print(f"  Steam Link connected   {line.split('[Info]')[0].strip()}")
                elif "Performing shutdown" in line:
                    print(f"  Steam Link disconnected {line.split('[Info]')[0].strip()}")
                elif "supports eye tracking" in line:
                    print("\n" + "=" * 60)
                    print("  EYE TRACKING IS NOW BEING SHARED. This is the line that was missing.")
                    print("  Leave Steam Link connected and start Isaac Sim VR now.")
                    print("=" * 60)
                    return 0
                elif seen_connect and "eye" in line.lower():
                    print(f"  (eye-related) {line.strip()[:100]}")
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopped. The handshake line never appeared, so the headset is still not")
        print("offering eye tracking to Steam Link. Virtual Desktop with 'Forward tracking")
        print("data' enabled is the other route that is known to work on this rig.")
        return 1


def main() -> int:
    if "--watch" in sys.argv:
        return watch()
    print("Quest Pro eye-gaze chain\n" + "=" * 60)
    check_runtime()
    check_share_setting()
    check_steam_link()
    check_kit()
    print("\n" + "=" * 60)
    print("Order matters: connect Steam Link FIRST and leave it connected, then start")
    print("Isaac Sim VR. /user/eye/unified appears about half a second into the session.")
    print("\nRun with --watch to sit on the Steam Link log and be told the instant the")
    print("headset starts sharing eye tracking, without launching Isaac Sim.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
