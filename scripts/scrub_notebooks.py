import argparse
import json
import os
from pathlib import Path
import subprocess

strip_metadata = {
    "jupytext": {"notebook_metadata_filter": "-all", "cell_metadata_filter": "-all"}
}

strip_metadata_string = json.dumps(strip_metadata).replace('"', '"')


def scrub(notebooks_path, scrubbed_path):
    for notebook_file in notebooks_path.glob("*.ipynb"):
        scrubbed_file = Path(scrubbed_path) / notebook_file.name
        subprocess.run(
            f"jupytext --output '{scrubbed_file}.md' --to md '{notebook_file}' --update-metadata '{strip_metadata_string}' ",
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
        )
        res = subprocess.run(
            f"jupytext --output '{scrubbed_file}' --to notebook '{scrubbed_file}.md'",
            shell=True,
            check=True,
            universal_newlines=True,
            stdout=subprocess.PIPE,
        )
        print(res.stdout.split("\n")[1])
        subprocess.run(
            f"rm '{scrubbed_file}.md'", shell=True, check=True, stdout=subprocess.PIPE
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("notebooks_path", type=Path)
    parser.add_argument("scrubbed_path", type=Path)
    p = parser.parse_args()
    assert os.path.exists(p.notebooks_path)
    assert os.path.exists(p.scrubbed_path)
    scrub(p.notebooks_path, p.scrubbed_path)
