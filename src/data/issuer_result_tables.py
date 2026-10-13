"""Quarterly issuer summary tables: preserve GAAP/non-GAAP and printed precision."""
import io
import re

from bs4 import BeautifulSoup
import pandas as pd


PERIOD = re.compile(r"^Q[1-4][\s'’-]*(?:FY)?\s*\d{2,4}$", re.I)


def _money(values):
    tokens = list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))
    text = "".join(tokens).replace(",", "").replace(" ", "").replace("$", "").rstrip("^")
    suffix = text[-1].upper() if text[-1:] in {"M", "B", "m", "b"} else None
    if suffix:
        text = text[:-1]
    if not re.fullmatch(r"\(?-?\d+(?:\.\d+)?\)?", text):
        raise ValueError("current-period monetary cell is missing or ambiguous")
    negative = text.startswith("(") or text.startswith("-")
    number = text.replace("$", "").replace("(", "").replace(")", "").lstrip("-")
    quantum = 10 ** (-len(number.split(".")[1]) if "." in number else 0)
    return (-1 if negative else 1) * float(number), quantum, suffix


def parse_quarterly_results(raw):
    soup = BeautifulSoup(raw, "html.parser")
    for marker in soup.find_all("sup"):
        marker.decompose()
    found, rejected = {}, []
    tables = []
    for table_number, table in enumerate(soup.find_all("table")):
        previous = table.find_previous(["p", "h2", "h3", "h4"])
        caption = previous.get_text(" ", strip=True) if previous else ""
        label = table.find_previous(string=re.compile(r"^(?:Non-GAAP(?:\s*\([^)]*\))?|GAAP)\s+(?:Quarterly\s+)?Financial Results\s*$", re.I))
        basis_caption = caption if re.search(r"\bgaap\b", caption, re.I) else (str(label).strip() if label else caption)
        rows = table.find_all("tr")
        starts = [i for i, row in enumerate(rows) if row.get_text(" ", strip=True).lower() in {"gaap", "non-gaap"}]
        if len(starts) > 1:
            for index, start in enumerate(starts):
                end = starts[index + 1] if index + 1 < len(starts) else len(rows)
                tables.append((table_number, caption, basis_caption, "<table>" + "".join(str(row) for row in rows[start:end]) + "</table>"))
        else:
            tables.append((table_number, caption, basis_caption, str(table)))
    for table_number, caption, basis_caption, table in tables:
        try:
            frame = pd.read_html(io.StringIO(table), header=None, flavor="lxml")[0].fillna("").astype(str)
        except ValueError:
            continue
        if any(not isinstance(value, int) for value in frame.columns):
            columns = frame.columns.to_frame(index=False).T if isinstance(frame.columns, pd.MultiIndex) else pd.DataFrame([list(frame.columns)])
            columns = columns.map(lambda value: "" if str(value).startswith("Unnamed:") else re.sub(r"\.\d+$", "", str(value)))
            frame.columns = range(len(frame.columns)); columns.columns = frame.columns
            frame = pd.concat([columns, frame], ignore_index=True)
        labels = frame.iloc[:, 0].str.replace(r"\s+", " ", regex=True).str.strip()
        rev_rows = [i for i, value in enumerate(labels) if re.match(r"^Revenue(?:\s|$)", value, re.I)]
        eps_all = [i for i, value in enumerate(labels) if re.match(r"^(?:Diluted\s+)?(?:Earnings(?:\s*\(loss\))?\s+per\s+(?:diluted\s+)?share|EPS)\b", value, re.I)]
        eps_rows = [i for i in eps_all if not re.search(r"\bexcluding\b|\bwithout\b", labels.iloc[i], re.I)]
        if len(rev_rows) != 1 or len(eps_rows) != 1:
            continue
        header = frame.iloc[:min(rev_rows[0], eps_rows[0])]
        period_rows = [i for i in range(len(header)) if any(PERIOD.fullmatch(value.strip()) for value in header.iloc[i])]
        if not period_rows:
            continue  # Annual and forecast tables are not quarterly actuals.
        period_row = header.iloc[period_rows[0]]
        header_scope = " ".join(header.to_numpy().ravel()).lower()
        scope = (caption + " " + header_scope).lower()
        basis_scope = header_scope if re.search(r"\bgaap\b", header_scope) else basis_caption.lower()
        if re.search(r"\boutlook\b|\bforecast\b|\bapproximately\b", scope):
            continue
        mixed = None
        for i in range(len(header)):
            values = [value.strip().lower() for value in header.iloc[i]]
            if "gaap" in values and "non-gaap" in values:
                mixed = {"gaap": values.index("gaap"), "non_gaap": values.index("non-gaap")}
                break
        if mixed:
            starts = sorted(mixed.items(), key=lambda item: item[1])
            groups = [(basis, max(1, start), starts[index + 1][1] if index + 1 < len(starts) else len(frame.columns)) for index, (basis, start) in enumerate(starts)]
        elif "non-gaap" in basis_scope:
            groups = [("non_gaap", 1, len(frame.columns))]
        elif re.search(r"\bgaap\b", basis_scope):
            groups = [("gaap", 1, len(frame.columns))]
        else:
            continue
        for basis, start, end in groups:
            if basis in found:
                continue
            columns = [j for j in range(start, end) if PERIOD.fullmatch(period_row.iloc[j].strip())]
            if not columns:
                continue
            original_period = period_row.iloc[columns[0]].strip()
            columns = [j for j in columns if period_row.iloc[j].strip() == original_period]
            current = re.sub(r"[’'-]", " ", original_period)
            try:
                missing = {}
                try:
                    revenue, revenue_quantum, suffix = _money(frame.iloc[rev_rows[0], columns])
                except ValueError as error:
                    revenue, revenue_quantum, suffix = None, None, None
                    missing["revenue"] = str(error)
                try:
                    eps, eps_quantum, eps_suffix = _money(frame.iloc[eps_rows[0], columns])
                except ValueError as error:
                    eps, eps_quantum, eps_suffix = None, None, None
                    missing["eps"] = str(error)
                if revenue is None and eps is None:
                    raise ValueError("both current actual fields unavailable")
                if eps_suffix:
                    raise ValueError("EPS must be dollars per share")
                units = (labels.iloc[rev_rows[0]] + " " + " ".join(header.to_numpy().ravel())).lower()
                if suffix == "B" or re.search(r"\$\s*b\b|\bbillions?\b", units):
                    multiplier = 1000
                elif suffix == "M" or re.search(r"\$\s*m\b|\bmillions?\b", units):
                    multiplier = 1
                else:
                    raise ValueError("quarterly revenue units unknown")
                if revenue is not None and revenue <= 0:
                    raise ValueError("quarterly revenue must be positive")
                alternatives = []
                for row_number in eps_all:
                    if row_number == eps_rows[0]:
                        continue
                    value, quantum, unit = _money(frame.iloc[row_number, columns])
                    if unit:
                        raise ValueError("alternative EPS unit differs")
                    alternatives.append({"label": labels.iloc[row_number], "eps_usd": value, "rounding_half": quantum / 2})
                found[basis] = {"quarter_label": current, "revenue_million_usd": revenue * multiplier if revenue is not None else None,
                    "eps_usd": eps, "revenue_rounding_half_million": revenue_quantum * multiplier / 2 if revenue_quantum is not None else None,
                    "eps_rounding_half": eps_quantum / 2 if eps_quantum is not None else None, "table_number": table_number,
                    "unavailable_fields": missing, "eps_alternatives": alternatives,
                    "column_indices": columns, "caption": caption, "forecast_basis_match_certified": False}
            except ValueError as error:
                rejected.append({"table_number": table_number, "basis": basis, "error": str(error)})
        if len(found) == 2:
            break
    if not found:
        raise ValueError("no unambiguous quarterly actual summary; " + str(rejected[:3]))
    return {"basis": found, "rejected_tables": rejected, "selection": "first explicitly labelled quarterly actual summary per basis; no closeness-to-consensus selection"}
