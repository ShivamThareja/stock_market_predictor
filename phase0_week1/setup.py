"""
Week 1 Setup — run this ONCE before anything else
Usage: python3 setup.py
"""
import subprocess
import sys

packages = [
    "yfinance",
    "pandas",
    "matplotlib",
    "seaborn",
    "jupyter",
    "openpyxl",
]

print("Installing packages for Week 1...\n")
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

print("\nAll done. Now run:")
print("  python3 week1_stocks.py")
