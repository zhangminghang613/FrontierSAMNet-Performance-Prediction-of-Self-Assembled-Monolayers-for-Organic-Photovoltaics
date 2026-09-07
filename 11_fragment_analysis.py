from pathlib import Path
import re
import warnings
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import BRICS
from rdkit.Chem import Descriptors, Lipinski, Crippen, rdMolDescriptors

warnings.filterwarnings("ignore")

DATA_PATH = Path("data/processed/sam_clean.csv")

OUT_DIR = Path("results")
TAB_DIR = OUT_DIR / "tables"
TAB_DIR.mkdir(parents=True, exist_ok=True)

def mol_from_smiles(smiles):
    if pd.isna(smiles):
        return None

    smiles = str(smiles).strip()

    if not smiles:
        return None

    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol
    except Exception:
        return None


def canonical_smiles(smiles):
    mol = mol_from_smiles(smiles)

    if mol is None:
        return None

    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def clean_text_label(x, max_len=38):
    x = str(x)
    if len(x) <= max_len:
        return x
    return x[: max_len - 3] + "..."



SMARTS_LIBRARY = {
    
    "phosphonic acid / phosphonate": [
        "P(=O)(O)O",
        "P(=O)([O-])O",
        "P(=O)(O)[O-]",
    ],
    "carboxylic acid / carboxylate": [
        "C(=O)[O;H1]",
        "C(=O)[O-]",
        "C(=O)O",
    ],

    
    "carbazole core": [
        "n1c2ccccc2c2ccccc21",
        "[nX3]1c2ccccc2c2ccccc21",
    ],
    "benzene ring": [
        "c1ccccc1",
    ],
    "thiophene ring": [
        "c1ccsc1",
        "c1cccs1",
    ],

    # Exact C2/C3/C4 spacer flags are assigned by
    # classify_exact_alkyl_spacer() below.  They remain in the feature table
    # with their historical column names, but are intentionally not defined
    # by nested SMARTS such as CCP/CCCP/CCCCP.
    "C2 alkyl spacer": [
    ],
    "C3 alkyl spacer": [
    ],
    "C4 alkyl spacer": [
    ],
    "C6 alkyl chain": [
        "CCCCCC",
    ],

    
    "carbonyl group": [
        "C=O",
    ],
    "fluorinated carbon": [
        "[CX4](F)",
        "cF",
    ],
    "chlorinated aryl": [
        "cCl",
    ],
    "brominated aryl": [
        "cBr",
    ],
}


def compile_smarts_library():
    compiled = {}

    for name, smarts_list in SMARTS_LIBRARY.items():
        patterns = []

        for smarts in smarts_list:
            patt = Chem.MolFromSmarts(smarts)
            if patt is not None:
                patterns.append(patt)

        compiled[name] = patterns

    return compiled


SMARTS_PATTERNS = compile_smarts_library()


EXACT_SPACER_FEATURES = {
    2: "C2 alkyl spacer",
    3: "C3 alkyl spacer",
    4: "C4 alkyl spacer",
}


def _phosphonic_anchor_p_indices(mol):
    """Return P atoms belonging to the phosphonic acid/phosphonate motif."""
    if mol is None:
        return []

    p_indices = set()
    for patt in SMARTS_PATTERNS.get("phosphonic acid / phosphonate", []):
        try:
            for match in mol.GetSubstructMatches(patt):
                if match:
                    p_indices.add(int(match[0]))
        except Exception:
            continue

    return sorted(p_indices)


def _shortest_alkyl_path_from_p_to_ring(mol, p_idx):
    """Count non-ring carbon atoms between one anchor P and the nearest ring.

    The terminal ring atom is treated as the core-connection atom and is not
    included in the spacer length.  A valid alkyl spacer path must therefore
    have the form P-(aliphatic C)n-(ring atom).  Paths containing O, N, S or a
    ring atom inside the linker are not labelled as an alkyl spacer.

    Returns
    -------
    int or None
        0 for direct P-ring attachment, a positive integer for an exact alkyl
        spacer length, or None when no valid P-to-ring alkyl path exists.
    """
    ring_indices = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetIdx() != p_idx and atom.IsInRing()
    ]
    if not ring_indices:
        return None

    candidates = []
    for ring_idx in ring_indices:
        try:
            path = tuple(Chem.GetShortestPath(mol, int(p_idx), int(ring_idx)))
        except Exception:
            continue

        if len(path) < 2:
            continue

        linker_indices = path[1:-1]
        linker_atoms = [mol.GetAtomWithIdx(int(idx)) for idx in linker_indices]
        if not all(
            atom.GetAtomicNum() == 6
            and not atom.GetIsAromatic()
            and not atom.IsInRing()
            for atom in linker_atoms
        ):
            continue

        # Sort primarily by graph distance.  The second key makes selection
        # deterministic if symmetry produces several equally short paths.
        candidates.append((len(path), len(linker_indices), path))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    return int(candidates[0][1])


def classify_exact_alkyl_spacer(mol):
    """Classify the exact anchor-to-core alkyl spacer length.

    Every phosphonic-acid/phosphonate P atom is evaluated independently.  A
    molecule is assigned an exact Cn class only when all resolvable anchors
    give the same P-(aliphatic C)n-(ring atom) length.  Molecules with
    different lengths at different anchors are marked ambiguous and are not
    counted in any exact C2/C3/C4 column.
    """
    p_indices = _phosphonic_anchor_p_indices(mol)
    if not p_indices:
        return {
            "exact_spacer_length": np.nan,
            "exact_spacer_class": "no phosphonic anchor",
            "anchor_spacer_lengths": "",
        }

    anchor_lengths = [
        _shortest_alkyl_path_from_p_to_ring(mol, p_idx)
        for p_idx in p_indices
    ]
    resolved = [length for length in anchor_lengths if length is not None]
    serialized = "|".join(
        "NA" if length is None else str(int(length))
        for length in anchor_lengths
    )

    if not resolved:
        return {
            "exact_spacer_length": np.nan,
            "exact_spacer_class": "no ring-core path",
            "anchor_spacer_lengths": serialized,
        }

    unique_lengths = sorted(set(int(length) for length in resolved))
    if len(unique_lengths) != 1 or len(resolved) != len(anchor_lengths):
        return {
            "exact_spacer_length": np.nan,
            "exact_spacer_class": "ambiguous",
            "anchor_spacer_lengths": serialized,
        }

    length = unique_lengths[0]
    if length == 0:
        spacer_class = "direct/aryl linker"
    elif length in EXACT_SPACER_FEATURES:
        spacer_class = f"C{length}"
    elif length >= 5:
        spacer_class = "C5+"
    else:
        spacer_class = f"C{length} other"

    return {
        "exact_spacer_length": int(length),
        "exact_spacer_class": spacer_class,
        "anchor_spacer_lengths": serialized,
    }


def has_smarts(mol, motif_name):
    patterns = SMARTS_PATTERNS.get(motif_name, [])

    for patt in patterns:
        try:
            if mol.HasSubstructMatch(patt):
                return True
        except Exception:
            continue

    return False


def detect_smarts_features(mol, spacer_info=None):
    out = {}

    for motif in SMARTS_LIBRARY.keys():
        out[motif] = int(has_smarts(mol, motif))

    if spacer_info is None:
        spacer_info = classify_exact_alkyl_spacer(mol)

    exact_length = spacer_info.get("exact_spacer_length", np.nan)
    for length, motif in EXACT_SPACER_FEATURES.items():
        out[motif] = int(pd.notna(exact_length) and int(exact_length) == length)

    return out

def calc_rdkit_design_descriptors(mol):
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]

    return {
        "Ring count": rdMolDescriptors.CalcNumRings(mol),
        "Aromatic rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "P atoms": atoms.count("P"),
        "N atoms": atoms.count("N"),
        "O atoms": atoms.count("O"),
        "F atoms": atoms.count("F"),
        "Heavy atoms": mol.GetNumHeavyAtoms(),
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Crippen.MolLogP(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "HBD": Lipinski.NumHDonors(mol),
        "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
    }


def clean_brics_fragment(fragment):
    s = str(fragment)

    
    s = re.sub(r"\[\d+\*\]", "[*]", s)

    mol = mol_from_smiles(s)

    if mol is None:
        return None

    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def get_brics_fragments(mol):
    try:
        frags = BRICS.BRICSDecompose(
            mol,
            keepNonLeafNodes=False,
            returnMols=False,
        )
    except Exception:
        return set()

    out = set()

    for f in frags:
        cf = clean_brics_fragment(f)
        if cf is not None:
            out.add(cf)

    return out


def build_unique_sam_table(df):
    required_cols = ["SMILES", "pce"]

    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    work = df.copy()

    if "name" not in work.columns:
        work["name"] = "SAM"

    work["canonical_smiles"] = work["SMILES"].apply(canonical_smiles)

    invalid = work["canonical_smiles"].isna().sum()

    work = work.dropna(subset=["canonical_smiles", "pce"]).copy()

    unique = (
        work.groupby("canonical_smiles", as_index=False)
        .agg(
            name=("name", "first"),
            original_smiles=("SMILES", "first"),
            pce_mean=("pce", "mean"),
            pce_median=("pce", "median"),
            pce_max=("pce", "max"),
            record_count=("pce", "size"),
        )
    )

    mols = []

    for smi in unique["canonical_smiles"]:
        mols.append(mol_from_smiles(smi))

    unique["mol"] = mols

    return unique, invalid


def add_fragment_and_descriptor_features(unique):
    smarts_rows = []
    spacer_rows = []
    desc_rows = []
    brics_rows = []

    for row in unique.itertuples(index=False):
        mol = row.mol

        spacer_info = classify_exact_alkyl_spacer(mol)
        smarts = detect_smarts_features(mol, spacer_info=spacer_info)
        desc = calc_rdkit_design_descriptors(mol)
        brics = get_brics_fragments(mol)

        smarts_rows.append(smarts)
        spacer_rows.append(spacer_info)
        desc_rows.append(desc)
        brics_rows.append(brics)

    smarts_df = pd.DataFrame(smarts_rows)
    spacer_df = pd.DataFrame(spacer_rows)
    desc_df = pd.DataFrame(desc_rows)

    out = pd.concat(
        [
            unique.drop(columns=["mol"]).reset_index(drop=True),
            smarts_df.reset_index(drop=True),
            spacer_df.reset_index(drop=True),
            desc_df.reset_index(drop=True),
        ],
        axis=1,
    )

    out["brics_fragments"] = ["|".join(sorted(x)) for x in brics_rows]

    return out, brics_rows


def assign_high_low_groups(unique_features):
    q75 = unique_features["pce_mean"].quantile(0.75)
    q25 = unique_features["pce_mean"].quantile(0.25)

    out = unique_features.copy()
    out["pce_group"] = "middle"
    out.loc[out["pce_mean"] >= q75, "pce_group"] = "high"
    out.loc[out["pce_mean"] <= q25, "pce_group"] = "low"

    return out, q75, q25


def audit_exact_spacer_features(unique_features):
    """Validate exclusivity and write molecule-level and grouped audits."""
    spacer_cols = list(EXACT_SPACER_FEATURES.values())
    missing = [col for col in spacer_cols if col not in unique_features.columns]
    if missing:
        raise KeyError(f"Missing exact spacer feature columns: {missing}")

    overlap = unique_features[spacer_cols].sum(axis=1) > 1
    if overlap.any():
        bad = unique_features.loc[
            overlap,
            ["name", "canonical_smiles"] + spacer_cols,
        ]
        raise ValueError(
            "Exact C2/C3/C4 spacer features must be mutually exclusive. "
            f"Overlapping rows:\n{bad.to_string(index=False)}"
        )

    audit_cols = [
        "canonical_smiles",
        "name",
        "original_smiles",
        "pce_mean",
        "pce_median",
        "record_count",
        "pce_group",
        "exact_spacer_length",
        "exact_spacer_class",
        "anchor_spacer_lengths",
    ] + spacer_cols
    audit = unique_features[audit_cols].copy()
    audit.to_csv(
        TAB_DIR / "sam_exact_spacer_audit_unique_smiles.csv",
        index=False,
    )

    counts = (
        audit.groupby(["pce_group", "exact_spacer_class"], dropna=False)
        .size()
        .rename("sam_count")
        .reset_index()
        .sort_values(["pce_group", "exact_spacer_class"])
    )
    counts.to_csv(
        TAB_DIR / "sam_exact_spacer_counts_by_pce_group.csv",
        index=False,
    )

    return audit, counts

def compute_brics_enrichment(unique_features, brics_rows):

    high_mask = unique_features["pce_group"].eq("high").values
    low_mask = unique_features["pce_group"].eq("low").values

    high_total = int(high_mask.sum())
    low_total = int(low_mask.sum())

    all_frags = sorted(set().union(*brics_rows))

    rows = []

    for frag in all_frags:
        present = np.array([frag in s for s in brics_rows])

        high_count = int((present & high_mask).sum())
        low_count = int((present & low_mask).sum())

        total_count = high_count + low_count

        
        if total_count < 2:
            continue

        high_frequency = high_count / max(high_total, 1)
        low_frequency = low_count / max(low_total, 1)

        
        high_rate_pc = (high_count + 0.5) / (high_total + 1.0)
        low_rate_pc = (low_count + 0.5) / (low_total + 1.0)

        log2_enrichment = float(np.log2(high_rate_pc / low_rate_pc))

        rows.append(
            {
                "brics_fragment": frag,
                "high_count": high_count,
                "low_count": low_count,
                "high_total": high_total,
                "low_total": low_total,
                "high_frequency": high_frequency,
                "low_frequency": low_frequency,
                "frequency_difference": high_frequency - low_frequency,
                "log2_enrichment": log2_enrichment,
                "total_count_high_low": total_count,
            }
        )

    out = pd.DataFrame(rows)

    if out.empty:
        out.to_csv(
            TAB_DIR / "sam_brics_fragment_enrichment_unique_smiles.csv",
            index=False,
        )
        return out

    out = out.sort_values(
        ["log2_enrichment", "high_count"],
        ascending=[False, False],
    ).reset_index(drop=True)

    out.to_csv(
        TAB_DIR / "sam_brics_fragment_enrichment_unique_smiles.csv",
        index=False,
    )

    return out


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Cannot find {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    unique, _ = build_unique_sam_table(df)
    unique_features, brics_rows = add_fragment_and_descriptor_features(unique)
    unique_features, _, _ = assign_high_low_groups(unique_features)

    audit_exact_spacer_features(unique_features)

    unique_features.to_csv(
        TAB_DIR / "sam_unique_smiles_fragment_features.csv",
        index=False,
    )

    compute_brics_enrichment(unique_features, brics_rows)


if __name__ == "__main__":
    main()
