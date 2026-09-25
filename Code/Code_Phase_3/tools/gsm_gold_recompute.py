#!/usr/bin/env python3
"""
gsm_gold_recompute.py — independent arithmetic check of the 20-item GSM gold audit.

Each gold label in results/processed/gsm_symbolic_gold_audit.csv is recomputed
from the problem text with an explicit formula written out below, and the
sheet is filled in. The formulas were written by an AI assistant (Claude,
25 Sept 2026) reading each problem; they are shown so a human can check every
step. This is an AI-assisted recomputation, not a human audit, and the sheet
says so.

    python tools/gsm_gold_recompute.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SHEET = ROOT / "results/processed/gsm_symbolic_gold_audit.csv"

FORMULAS = {
    "gsmorig_0737": ("32 - (3*4 + 7*2 + 8*0.5)", "water left after 3 giant, 7 medium, 8 small cubes"),
    "gsmsym_0084": ("10*56 * 1.30 * 0.75", "value after +30% then -25%"),
    "gsmorig_0440": ("2*(3*3 + 360/(3*3))", "length 3 yd = 9 ft; width = 360/9; perimeter"),
    "gsmsym_0098": ("16 * (1+3)**3", "infected total multiplies by 4 each day for 3 days"),
    "gsmorig_0858": ("10 * (1+6)**3", "infected total multiplies by 7 each day for 3 days"),
    "gsmorig_0164": ("3*35 - 90", "income lost doing it herself minus accountant fee"),
    "gsmsym_0030": ("102.5 - 25*0.98", "money left after 25 washers"),
    "gsmorig_0401": ("4*6*50*12", "4 weeks x 6 days x $50 x 12 months"),
    "gsmsym_0049": ("(450 + 450 + 1.2*(450+450)) * 2 * 0.45", "apartments x 2 collections x 0.45"),
    "gsmsym_0032": ("340000*(1 + 0.03 + 0.10) - 360000", "price + 3% + 10% minus budget"),
    "gsmsym_0008": ("640/8/10", "an eighth are tennis balls, a tenth of those white"),
    "gsmorig_1021": ("0.75*(0.4*60) + 0.5*(0.6*60)", "75% of easy + half of the rest"),
    "gsmsym_0039": ("100 * (210*20/60/10) / 10", "hours cleaning per day as % of a 10 h day"),
    "gsmorig_0265": ("350000*(1 + 0.05 + 0.12) - 400000", "price + 5% + 12% minus budget"),
    "gsmsym_0079": ("315/3 + 481", "a third of one puzzle plus a whole second puzzle"),
    "gsmsym_0026": ("(6 + 3 + 6) * 7", "family tacos per week"),
    "gsmsym_0078": ("((6+1)*60*6 + 2*((6+1)*60*6)/7) / 60", "weekly class hours plus 1/7 of weekly minutes on each weekend day"),
    "gsmsym_0081": ("180 - 2*43 - 17", "coins found by friends"),
    "gsmorig_0718": ("100*0.30*0.20/3", "interviews, offers, a third accept"),
    "gsmorig_0491": ("64 - 2*24", "eggs that do not fit on two trays of 24"),
}


def main() -> int:
    sheet = pd.read_csv(SHEET)
    recomputed, agrees, notes = [], [], []
    for qid, gold in zip(sheet["question_identifier"], sheet["correct_answer"]):
        formula, meaning = FORMULAS[qid]
        value = eval(formula, {"__builtins__": {}})  # arithmetic literals only
        value = round(value, 6)
        value = int(value) if float(value).is_integer() else value
        recomputed.append(value)
        agrees.append(abs(float(value) - float(gold)) < 1e-6)
        notes.append(f"{meaning}: {formula}. AI-assisted recomputation (Claude, 25 Sept 2026); "
                     f"requires human confirmation.")
    sheet["auditor_recomputed_answer"] = recomputed
    sheet["auditor_agrees"] = agrees
    sheet["auditor_notes"] = notes
    sheet.to_csv(SHEET, index=False)
    print(f"{sum(agrees)} of {len(agrees)} gold labels agree with the recomputation")
    return 0 if all(agrees) else 1


if __name__ == "__main__":
    raise SystemExit(main())
