"""One-shot cleanup: prune fpocket viz files we no longer need."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
removed_files = 0
freed_bytes = 0
for split in ["train", "test"]:
    base = ROOT / "structures" / split
    for sub in base.glob("*_out"):
        for child in sub.iterdir():
            if child.is_file():
                if child.name.endswith("_info.txt"):
                    continue
                try:
                    freed_bytes += child.stat().st_size
                    child.unlink()
                    removed_files += 1
                except OSError:
                    pass
            elif child.is_dir() and child.name == "pockets":
                for f in child.iterdir():
                    if f.name == "pocket0_atm.pdb":
                        continue
                    try:
                        freed_bytes += f.stat().st_size
                        f.unlink()
                        removed_files += 1
                    except OSError:
                        pass
print(f"removed {removed_files} files, freed {freed_bytes/(1024*1024):.0f} MB")
