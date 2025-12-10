
from pathlib import Path
import pandas as pd
import yaml
from typing import Dict, Any, List

def xlsx_to_yaml(xlsx_path: Path, yaml_path: Path) -> None:
    """
    Convert a multi-sheet Excel config to a long YAML file.
    Expected sheets: Sections, Folders, Files, Levels, Variations, Tables.
    """
    xls = pd.ExcelFile(xlsx_path)
    data = {}
    for sheet in xls.sheet_names:
        df = pd.read_excel(xlsx_path, sheet_name=sheet)
        # Ensure NaNs become None in YAML
        data[sheet] = df.where(pd.notnull(df), None).to_dict(orient="records")
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

def yaml_to_xlsx(yaml_path: Path, xlsx_path: Path) -> None:
    """
    Convert a long YAML config (with top-level keys per sheet) to multi-sheet Excel.
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        data: Dict[str, List[Dict[str, Any]]] = yaml.safe_load(f) or {}
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        for sheet, rows in data.items():
            df = pd.DataFrame(rows)
            # preserve column order if possible
            if isinstance(rows, list) and rows:
                cols = list(rows[0].keys())
                df = df.reindex(columns=cols)
            df.to_excel(writer, index=False, sheet_name=sheet)

def roundtrip_validate(xlsx_in: Path) -> Dict[str, int]:
    """
    Optional helper: xlsx -> yaml -> xlsx; return row counts per sheet after roundtrip.
    """
    tmp_yaml = xlsx_in.with_suffix(".roundtrip.yaml")
    tmp_xlsx = xlsx_in.with_name(xlsx_in.stem + ".roundtrip.xlsx")
    xlsx_to_yaml(xlsx_in, tmp_yaml)
    yaml_to_xlsx(tmp_yaml, tmp_xlsx)
    out = {}
    a = pd.ExcelFile(xlsx_in)
    b = pd.ExcelFile(tmp_xlsx)
    for s in a.sheet_names:
        out[s] = pd.read_excel(tmp_xlsx, sheet_name=s).shape[0]
    return out
