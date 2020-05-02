import os
from pathlib import Path
import subprocess

path = Path("..").cwd() / "src"
os.chdir(path)

for file in path.glob("*.ipynb"):
    subprocess.run(
        f"jupytext --to md {Path(file).name}",
        shell=True,
        check=True,
        stdout=subprocess.PIPE,
    )
    res = subprocess.run(
        f"jupytext --to notebook {Path(file).stem}.md",
        shell=True,
        check=True,
        universal_newlines=True,
        stdout=subprocess.PIPE,
    )
    print(res.stdout.split("\n")[1])
    subprocess.run(
        f"rm {Path(file).stem}.md", shell=True, check=True, stdout=subprocess.PIPE
    )
