"""
Week 2 Setup — installs the one new library needed
Usage: python3 setup.py
"""
import subprocess
import sys

packages = ["requests", "pandas"]

print("Installing packages for Week 2...\n")
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
print("\nIMPORTANT: Before running week2_news.py,")
print("open config.py and paste your NewsAPI key in.")
print("\nThen run: python3 week2_news.py")
