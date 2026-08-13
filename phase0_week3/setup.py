"""
Week 3 Setup — installs the one new library needed
(sqlite3 is Python's standard library — nothing to install for that part)
Usage: python3 setup.py
"""
import subprocess
import sys

packages = ["requests", "apscheduler"]

print("Installing packages for Week 3...\n")
for pkg in packages:
    print(f"  Installing {pkg}...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", pkg, "-q",
         "--break-system-packages"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print(f"  ✓ {pkg}")
    else:
        print(f"  ✗ {pkg} failed — try manually:")
        print(f"    pip3 install {pkg} --break-system-packages")

print("\nAll done.")
print("\nIMPORTANT: Before running week3_pipeline.py,")
print("open config.py and paste your NewsAPI key in")
print("(same key you used for Week 2).")
print("\nTest it first:  python3 week3_pipeline.py --once")
print("Then run for real: python3 week3_pipeline.py")
