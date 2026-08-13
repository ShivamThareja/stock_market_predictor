"""
Week 4 Setup — installs FinBERT's dependencies
Usage: python3 setup.py

Note: torch + transformers are large downloads (~1GB combined).
First run of the main script will also download FinBERT's model
weights (~400MB) from Hugging Face — that only happens once, then
it's cached locally.
"""
import subprocess
import sys

packages = ["torch", "transformers", "pandas"]

print("Installing packages for Week 4 (this may take a few minutes)...\n")
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
print("\nRun: python3 week4_finbert.py")
print("(First run will also download FinBERT's model weights — one-time, ~400MB)")
