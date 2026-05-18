"""
Fetch the coding DNA sequence (CDS) for every UniProt ID in the dataset.

Nucleotide foundation models (Nucleotide Transformer, Evo) need DNA, not
protein. The convergent_enzymes HF dataset ships only the AA sequence, so
this script does the two-step lookup:

  1) UniProt REST API for each Entry -> list of EMBL cross-references with
     a per-CDS GenPept protein ID (e.g. "ABR54795.1") and molecule type.
  2) NCBI EFetch (`db=protein, rettype=fasta_cds_na`) for each protein ID
     -> the actual coding DNA sequence in FASTA form.

We pick one EMBL CDS per UniProt entry (the first Genomic_DNA cross-ref).
Results are cached to disk so re-runs are cheap.

Outputs:
  data/dna_sequences.csv     Entry, embl_xref, embl_protein_id, cds_dna
  data/dna_fetch_status.csv  per-Entry status (ok / no_embl / efetch_fail / ...)
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_FASTA = DATA_DIR / "dna_sequences.csv"
STATUS = DATA_DIR / "dna_fetch_status.csv"

UNIPROT_API = "https://rest.uniprot.org/uniprotkb/{uid}.json"
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
REQ_TIMEOUT = 30

session = requests.Session()


def uniprot_embl_cds(uid: str) -> tuple[str | None, str | None]:
    """Return (embl_xref_id, embl_protein_id). Prefer Genomic_DNA cross-refs."""
    try:
        r = session.get(UNIPROT_API.format(uid=uid), timeout=REQ_TIMEOUT)
    except requests.RequestException:
        return None, None
    if r.status_code != 200:
        return None, None
    try:
        d = r.json()
    except Exception:
        return None, None
    candidates = []
    for x in d.get("uniProtKBCrossReferences", []):
        if x.get("database") != "EMBL":
            continue
        embl_xref = x.get("id")
        protein_id = None
        mol = None
        for p in x.get("properties", []):
            if p.get("key") == "ProteinId":
                protein_id = p.get("value")
            elif p.get("key") == "MoleculeType":
                mol = p.get("value")
        if protein_id and protein_id != "-":
            candidates.append((embl_xref, protein_id, mol))
    if not candidates:
        return None, None
    # Prefer Genomic_DNA, fall back to anything else
    genomic = [c for c in candidates if c[2] == "Genomic_DNA"]
    pick = genomic[0] if genomic else candidates[0]
    return pick[0], pick[1]


def efetch_cds_dna(protein_id: str, retries: int = 3) -> str | None:
    """EFetch CDS in nucleotide FASTA. Returns the concatenated sequence string.
    Retries on transient failures with exponential backoff."""
    for attempt in range(retries):
        try:
            r = session.get(EFETCH, params={
                "db": "protein", "id": protein_id,
                "rettype": "fasta_cds_na", "retmode": "text",
            }, timeout=REQ_TIMEOUT)
        except requests.RequestException:
            time.sleep(0.5 * (2 ** attempt))
            continue
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(0.5 * (2 ** attempt))
            continue
        if r.status_code != 200 or not r.text or r.text.startswith("Error"):
            return None
        lines = r.text.splitlines()
        seq = "".join(line.strip() for line in lines if line and not line.startswith(">"))
        if seq:
            return seq.upper()
        return None
    return None


def fetch_one(uid: str) -> dict:
    embl_xref, protein_id = uniprot_embl_cds(uid)
    if not protein_id:
        return {"Entry": uid, "embl_xref": "", "embl_protein_id": "",
                "cds_dna": "", "status": "no_embl_cds_xref"}
    dna = efetch_cds_dna(protein_id)
    if not dna:
        return {"Entry": uid, "embl_xref": embl_xref, "embl_protein_id": protein_id,
                "cds_dna": "", "status": "efetch_failed"}
    return {"Entry": uid, "embl_xref": embl_xref, "embl_protein_id": protein_id,
            "cds_dna": dna, "status": "ok"}


def fetch_all(ids: list[str], workers: int = 3) -> pd.DataFrame:
    """Run with 3 workers to stay under NCBI's 3 req/s limit (no API key)."""
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(fetch_one, uid): uid for uid in ids}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="DNA fetch"):
            rows.append(fut.result())
    return pd.DataFrame(rows)


def main(limit: int | None = None) -> None:
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    ids = pd.concat([train, test])["Entry"].drop_duplicates().tolist()
    if limit:
        ids = ids[:limit]
        print(f"[LIMIT] only {limit} ids")

    # Resume support: skip ids we've already fetched successfully
    done: set[str] = set()
    if OUT_FASTA.exists():
        prev = pd.read_csv(OUT_FASTA)
        done = set(prev[prev["status"] == "ok"]["Entry"].astype(str))
        print(f"resume: {len(done)} already fetched")
        ids = [u for u in ids if u not in done]

    df = fetch_all(ids)
    if OUT_FASTA.exists():
        df = pd.concat([prev, df], ignore_index=True).drop_duplicates(
            subset="Entry", keep="last")
    df.to_csv(OUT_FASTA, index=False)
    df[["Entry", "status", "embl_xref", "embl_protein_id"]].to_csv(
        STATUS, index=False)
    print()
    print(df["status"].value_counts().to_string())
    print(f"\nsaved {OUT_FASTA}")


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(limit=limit)
