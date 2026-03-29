#!/usr/bin/env python3
"""
Docker Compose Port Diagnostic Tool
Run with: python debug_compose.py
Place this file in the same directory as your docker-compose.yml
"""

import re
import sys

def analyze_file(filepath="docker-compose.yml"):
    print(f"\n{'='*50}")
    print(f"  Analyzing: {filepath}")
    print(f"{'='*50}\n")

    try:
        # Read as raw bytes first
        with open(filepath, "rb") as f:
            raw = f.read()

        # Check for BOM
        if raw.startswith(b'\xef\xbb\xbf'):
            print("❌ BOM DETECTED at start of file (UTF-8 BOM) — this can break YAML parsers")
        else:
            print("✅ No BOM detected")

        # Decode to string
        content = raw.decode("utf-8", errors="replace")
        lines = content.splitlines()

        print(f"\n📄 File has {len(lines)} lines\n")

        # Scan every line for suspicious characters
        print("--- Line-by-line scan ---")
        found_issues = False
        for i, line in enumerate(lines, start=1):
            issues = []

            # Check for non-ASCII characters
            for j, ch in enumerate(line):
                code = ord(ch)
                if code > 127:
                    issues.append(f"  col {j+1}: non-ASCII char U+{code:04X} ({repr(ch)})")

            # Check for invisible/zero-width characters
            for j, ch in enumerate(line):
                if ch in '\u200b\u200c\u200d\ufeff\u00a0\u00ad':
                    issues.append(f"  col {j+1}: invisible char U+{ord(ch):04X} ({repr(ch)})")

            # Check for tabs vs spaces issues in ports section
            if "ports" in line or "-" in line:
                if "\t" in line:
                    issues.append(f"  TAB character found (YAML requires spaces!)")

            if issues:
                found_issues = True
                print(f"\n❌ Line {i}: {repr(line)}")
                for iss in issues:
                    print(iss)
            else:
                # Only print ports-related lines cleanly
                if any(k in line for k in ["ports", "8000", "8501", "-"]):
                    print(f"  Line {i}: {repr(line)}")

        if not found_issues:
            print("\n✅ No suspicious characters found in any line")

        # Specifically dump the ports sections byte by byte
        print("\n--- Byte dump of lines containing port numbers ---")
        for i, line in enumerate(lines, start=1):
            if "8000" in line or "8501" in line:
                print(f"\nLine {i}: {line}")
                byte_line = line.encode("utf-8")
                hex_vals = " ".join(f"{b:02x}" for b in byte_line)
                print(f"  Hex: {hex_vals}")
                ascii_repr = " ".join(
                    chr(b) if 32 <= b < 127 else f"[{b:02x}]"
                    for b in byte_line
                )
                print(f"  Chars: {ascii_repr}")

        # Try parsing as YAML
        print("\n--- YAML parse test ---")
        try:
            import yaml
            with open(filepath, "r", encoding="utf-8") as f:
                parsed = yaml.safe_load(f)

            services = parsed.get("services", {})
            for svc_name, svc in services.items():
                ports = svc.get("ports", [])
                print(f"\nService '{svc_name}' ports:")
                for p in ports:
                    print(f"  {repr(p)}")
                    # Check if it looks like a valid port mapping
                    if not re.match(r'^\d+:\d+$', str(p).strip()):
                        print(f"  ⚠️  This doesn't look like a valid port mapping!")

            print("\n✅ YAML parsed successfully")

        except ImportError:
            print("⚠️  PyYAML not installed — skipping YAML parse (run: pip install pyyaml)")
        except Exception as e:
            print(f"❌ YAML parse error: {e}")

    except FileNotFoundError:
        print(f"❌ File not found: {filepath}")
        print("   Make sure you run this script from the same directory as docker-compose.yml")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

    print(f"\n{'='*50}\n")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "docker-compose.yml"
    analyze_file(path)