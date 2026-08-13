"""
Synthetic HR data generator for the People Intelligence platform.

Produces a DuckDB warehouse (data/warehouse.duckdb) containing six raw tables and
a set of governed views. All data is 100% synthetic - no real person, no real
company, no scraped or anonymised source. Names are drawn from a fixed token
list and are not intended to resemble anybody.

The generator deliberately plants two kinds of structure:

1. Business signals, so that the demo questions return genuine findings:
     - Operations: highest voluntary attrition (~18%), concentrated at Manager level
     - Engineering: longest time-to-fill (~67 days) and the most open critical reqs
     - Customer Success: engagement declining, negative sentiment about workload
     - Marketing: time-to-fill deteriorating sharply in the most recent quarter
     - Commerce: fastest-growing function

2. Data-quality defects, so that the quality engine has something real to catch:
     duplicate employee ids, missing departments, terminated employees with no
     termination date, termination before hire, duplicate candidate records,
     absurd time-to-fill outliers, and missing survey scores.

Run:  python data/generate_data.py
"""
from __future__ import annotations

import random
import sys
from datetime import date, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pi import config  # noqa: E402

RNG = np.random.default_rng(config.RANDOM_SEED)
random.seed(config.RANDOM_SEED)

AS_OF = date.fromisoformat(config.AS_OF_DATE)
WINDOW_START = date(AS_OF.year - 2, AS_OF.month, 1)


# --------------------------------------------------------------------------
# Reference data
# --------------------------------------------------------------------------
LEVELS = ["IC1", "IC2", "IC3", "IC4", "Manager", "Senior Manager", "Director", "VP"]
LEVEL_WEIGHTS = [0.14, 0.20, 0.21, 0.15, 0.14, 0.08, 0.06, 0.02]

STRUCTURE = {
    "Commerce": {
        "departments": ["Merchandising", "Ecommerce Trading", "Category Management"],
        "job_families": ["Merchandising", "Commercial Analytics", "Product Management"],
        "start_headcount": 300,
        "annual_growth": 0.14,
        "base_vol_attrition": 0.092,
    },
    "Operations": {
        "departments": ["Fulfilment", "Supply Chain", "Manufacturing", "Logistics"],
        "job_families": ["Operations Management", "Supply Chain", "Quality"],
        "start_headcount": 780,
        "annual_growth": 0.02,
        "base_vol_attrition": 0.150,
    },
    "Engineering": {
        "departments": ["Platform", "Data & AI", "Storefront", "Security"],
        "job_families": ["Software Engineering", "Data Engineering", "Security Engineering"],
        "start_headcount": 460,
        "annual_growth": 0.09,
        "base_vol_attrition": 0.118,
    },
    "Customer Success": {
        "departments": ["Member Support", "Trust & Safety", "Service Operations"],
        "job_families": ["Customer Support", "Service Operations"],
        "start_headcount": 420,
        "annual_growth": 0.06,
        "base_vol_attrition": 0.142,
    },
    "Marketing": {
        "departments": ["Brand", "Performance Marketing", "Partnerships"],
        "job_families": ["Marketing", "Creative", "Commercial Analytics"],
        "start_headcount": 210,
        "annual_growth": 0.005,
        "base_vol_attrition": 0.110,
    },
    "Corporate": {
        "departments": ["Finance", "People", "Legal", "Strategy"],
        "job_families": ["Finance", "People Operations", "Legal"],
        "start_headcount": 190,
        "annual_growth": 0.01,
        "base_vol_attrition": 0.072,
    },
}

LOCATIONS = [
    "Jacksonville FL", "Tampa FL", "New York NY", "Austin TX",
    "Las Vegas NV", "Remote - US", "London UK", "Manchester UK",
]
LOCATION_W = [0.20, 0.11, 0.14, 0.12, 0.09, 0.20, 0.09, 0.05]

SOURCES = ["Employee Referral", "Internal Mobility", "LinkedIn", "Job Board",
           "Agency", "University", "Direct Application", "Sourced - Recruiter"]
SOURCE_W = [0.16, 0.07, 0.24, 0.17, 0.08, 0.06, 0.13, 0.09]

VOL_REASONS = ["Better opportunity", "Compensation", "Career growth", "Manager",
               "Workload / burnout", "Relocation", "Return to study", "Retirement"]
INVOL_REASONS = ["Performance", "Restructure", "Policy violation", "End of contract"]

FIRST = ["Avery", "Rowan", "Emerson", "Quinn", "Harper", "Sawyer", "Reese", "Marlow",
         "Devon", "Kai", "Noor", "Ines", "Priya", "Malik", "Theo", "Juno", "Wren",
         "Ari", "Sana", "Ravi", "Nadia", "Elias", "Mira", "Otis", "Lena", "Zane"]
LAST = ["Calder", "Bhatt", "Okafor", "Lindqvist", "Moreau", "Vance", "Ibarra", "Nakamura",
        "Ellison", "Duarte", "Farrow", "Haddad", "Kowalski", "Mbeki", "Prasad", "Reyes",
        "Sorensen", "Tanaka", "Ustinov", "Whitlock", "Yilmaz", "Zamora", "Ashton", "Blythe"]


def month_ends(start: date, end: date) -> list[date]:
    out, cur = [], date(start.year, start.month, 1)
    while cur <= end:
        nxt = date(cur.year + (cur.month == 12), (cur.month % 12) + 1, 1)
        out.append(nxt - timedelta(days=1))
        cur = nxt
    return out


MONTHS = month_ends(WINDOW_START, AS_OF)


# --------------------------------------------------------------------------
# Signal shaping
# --------------------------------------------------------------------------
def attrition_multiplier(bu: str, month_index: int, n_months: int) -> float:
    """Time trend on the voluntary attrition hazard, by business unit."""
    progress = month_index / max(n_months - 1, 1)
    if bu == "Operations":
        # steady deterioration: ends ~4.2pts above where it started
        return 0.74 + 0.62 * progress
    if bu == "Engineering":
        # flat, then a sharp step up in the final five months
        return 1.0 + (0.55 if month_index >= n_months - 5 else 0.0)
    if bu == "Customer Success":
        return 0.90 + 0.35 * progress
    if bu == "Commerce":
        return 1.05 - 0.20 * progress
    return 1.0


def level_multiplier(bu: str, level: str) -> float:
    if bu == "Operations" and level in ("Manager", "Senior Manager"):
        return 1.85
    if level in ("Director", "VP"):
        return 0.45
    if level in ("IC1", "IC2"):
        return 1.25
    return 1.0


def base_time_to_fill(bu: str, job_family: str, level: str) -> float:
    base = {
        "Engineering": 62, "Commerce": 44, "Operations": 33,
        "Customer Success": 31, "Marketing": 46, "Corporate": 48,
    }[bu]
    if job_family in ("Data Engineering", "Security Engineering"):
        base += 11
    if level in ("Director", "VP"):
        base += 26
    elif level in ("Manager", "Senior Manager"):
        base += 9
    return base


def ttf_trend(bu: str, opened: date) -> float:
    """Marketing time-to-fill deteriorates sharply in the last quarter."""
    months_ago = (AS_OF.year - opened.year) * 12 + (AS_OF.month - opened.month)
    if bu == "Marketing" and months_ago <= 4:
        return 12.0
    if bu == "Engineering" and months_ago <= 6:
        return 5.0
    return 0.0


# --------------------------------------------------------------------------
# 1. Employees, hires, exits  (monthly simulation)
# --------------------------------------------------------------------------
def _new_employee(eid: int, bu: str, hire_date: date, level: str | None = None) -> dict:
    cfg = STRUCTURE[bu]
    level = level or str(RNG.choice(LEVELS, p=LEVEL_WEIGHTS))
    dept = str(RNG.choice(cfg["departments"]))
    fam = str(RNG.choice(cfg["job_families"]))
    band = LEVELS.index(level)
    salary = int(np.clip(RNG.normal(62000 + band * 21500, 9000 + band * 2600), 38000, 385000))
    return {
        "employee_id": f"E{eid:06d}",
        "full_name": f"{random.choice(FIRST)} {random.choice(LAST)}",
        "email": f"user{eid:06d}@example-synthetic.test",
        "business_unit": bu,
        "department": dept,
        "job_family": fam,
        "job_level": level,
        "location": str(RNG.choice(LOCATIONS, p=LOCATION_W)),
        "work_model": str(RNG.choice(["Onsite", "Hybrid", "Remote"], p=[0.42, 0.36, 0.22])),
        "hire_date": hire_date,
        "termination_date": None,
        "termination_type": None,
        "termination_reason": None,
        "base_salary": salary,
        "bonus_target_pct": round(float(np.clip(RNG.normal(6 + band * 2.2, 2), 0, 45)), 1),
        "performance_rating": str(RNG.choice(
            ["Exceeds", "Strong", "Meets", "Developing"], p=[0.12, 0.30, 0.46, 0.12])),
        "recruiting_source": str(RNG.choice(SOURCES, p=SOURCE_W)),
    }


def simulate_workforce() -> tuple[pd.DataFrame, list[dict]]:
    employees: dict[str, dict] = {}
    hire_events: list[dict] = []
    next_id = 100001

    # ---- opening population (hired before the analysis window)
    for bu, cfg in STRUCTURE.items():
        for _ in range(cfg["start_headcount"]):
            tenure_days = int(np.clip(RNG.gamma(2.0, 620), 40, 5200))
            hd = WINDOW_START - timedelta(days=tenure_days)
            e = _new_employee(next_id, bu, hd)
            employees[e["employee_id"]] = e
            next_id += 1

    n_months = len(MONTHS)
    for mi, me in enumerate(MONTHS):
        m_start = me.replace(day=1)
        for bu, cfg in STRUCTURE.items():
            active = [e for e in employees.values()
                      if e["hire_date"] <= me and e["termination_date"] is None]
            active = [e for e in active if e["business_unit"] == bu]
            if not active:
                continue

            vol_hazard = cfg["base_vol_attrition"] / 12 * attrition_multiplier(bu, mi, n_months)
            invol_hazard = 0.030 / 12

            leavers_v, leavers_i = [], []
            for e in active:
                # brand-new joiners rarely leave in month one
                if (me - e["hire_date"]).days < 45:
                    continue
                p = vol_hazard * level_multiplier(bu, e["job_level"])
                if RNG.random() < p:
                    leavers_v.append(e)
                elif RNG.random() < invol_hazard:
                    leavers_i.append(e)

            for e in leavers_v:
                e["termination_date"] = m_start + timedelta(days=int(RNG.integers(0, 28)))
                e["termination_type"] = "Voluntary"
                weights = [0.24, 0.17, 0.16, 0.11, 0.13, 0.08, 0.05, 0.06]
                if bu in ("Operations", "Customer Success"):
                    weights = [0.19, 0.16, 0.12, 0.12, 0.27, 0.06, 0.04, 0.04]
                e["termination_reason"] = str(RNG.choice(VOL_REASONS, p=weights))
            for e in leavers_i:
                e["termination_date"] = m_start + timedelta(days=int(RNG.integers(0, 28)))
                e["termination_type"] = "Involuntary"
                e["termination_reason"] = str(RNG.choice(INVOL_REASONS, p=[0.44, 0.36, 0.06, 0.14]))

            # ---- hiring: backfill exits plus planned growth
            growth_add = len(active) * (cfg["annual_growth"] / 12)
            planned = len(leavers_v) + len(leavers_i) + growth_add
            # Marketing is under a hiring freeze in the recent period
            if bu == "Marketing" and mi >= n_months - 6:
                planned *= 0.45
            n_hires = int(RNG.poisson(max(planned, 0.1)))
            for _ in range(n_hires):
                hd = m_start + timedelta(days=int(RNG.integers(0, 28)))
                if hd > AS_OF:
                    continue
                e = _new_employee(next_id, bu, hd)
                employees[e["employee_id"]] = e
                hire_events.append({"employee_id": e["employee_id"], "business_unit": bu,
                                    "department": e["department"], "job_family": e["job_family"],
                                    "job_level": e["job_level"], "hire_date": hd,
                                    "source": e["recruiting_source"]})
                next_id += 1

    df = pd.DataFrame(list(employees.values()))
    return df, hire_events


# --------------------------------------------------------------------------
# 2. Internal mobility
# --------------------------------------------------------------------------
def build_internal_moves(emp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pool = emp[emp["termination_date"].isna()].sample(frac=0.16, random_state=7)
    for i, (_, e) in enumerate(pool.iterrows()):
        move_date = WINDOW_START + timedelta(days=int(RNG.integers(0, (AS_OF - WINDOW_START).days)))
        if move_date < e["hire_date"] + timedelta(days=200):
            continue
        mtype = str(RNG.choice(["Promotion", "Lateral", "Transfer"], p=[0.46, 0.34, 0.20]))
        cur = LEVELS.index(e["job_level"])
        to_level = LEVELS[min(cur + 1, len(LEVELS) - 1)] if mtype == "Promotion" else e["job_level"]
        to_bu = e["business_unit"]
        if mtype == "Transfer":
            to_bu = str(RNG.choice([b for b in STRUCTURE if b != e["business_unit"]]))
        rows.append({
            "move_id": f"M{i:06d}", "employee_id": e["employee_id"], "move_date": move_date,
            "move_type": mtype, "from_business_unit": e["business_unit"], "to_business_unit": to_bu,
            "from_job_level": e["job_level"], "to_job_level": to_level,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 3. Requisitions + candidate funnel
# --------------------------------------------------------------------------
STAGES = ["Applied", "Recruiter Screen", "Hiring Manager Screen",
          "Onsite Interview", "Offer Extended", "Offer Accepted"]

SOURCE_QUALITY = {
    "Employee Referral": 1.55, "Internal Mobility": 1.70, "Sourced - Recruiter": 1.20,
    "LinkedIn": 1.00, "Agency": 1.05, "University": 0.80,
    "Direct Application": 0.62, "Job Board": 0.48,
}


def build_recruiting(hire_events: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    reqs, cands = [], []
    rid = cid = 0

    def make_candidates(req: dict, filled: bool):
        """Simulate one requisition's candidate pool.

        Offers are modelled the way a real process works: recruiters extend one
        offer at a time and only extend another if the first is declined. That
        keeps offer-acceptance rate meaningful instead of being an artefact of
        how many people happened to reach the final stage.
        """
        nonlocal cid
        n_app = int(np.clip(RNG.normal(46 if req["business_unit"] != "Engineering" else 72, 16), 8, 190))
        batch: list[dict] = []
        p_accept_req = 0.66 if req["business_unit"] in ("Engineering", "Commerce") else 0.80
        for _ in range(n_app):
            cid += 1
            src = str(RNG.choice(SOURCES, p=SOURCE_W))
            q = SOURCE_QUALITY[src]
            applied = req["opened_date"] + timedelta(days=int(RNG.integers(0, 45)))
            # stage conversion probabilities
            p_screen = np.clip(0.34 * q, 0.05, 0.92)
            p_hm = np.clip(0.55 * q ** 0.5, 0.05, 0.92)
            # Engineering has a well-known onsite -> offer bottleneck
            p_onsite = np.clip((0.30 if req["business_unit"] == "Engineering" else 0.52) * q ** 0.4, 0.05, 0.92)

            stage_idx, dt = 0, applied
            dates = {"applied_date": applied, "screen_date": None, "hm_screen_date": None,
                     "onsite_date": None, "offer_date": None, "decision_date": None}
            for step, p, key in [
                (1, p_screen, "screen_date"), (2, p_hm, "hm_screen_date"),
                (3, p_onsite, "onsite_date"),
            ]:
                if RNG.random() > p:
                    break
                dt = dt + timedelta(days=int(np.clip(RNG.normal(9, 4), 1, 40)))
                dates[key] = dt
                stage_idx = step
            batch.append({
                "candidate_id": f"C{cid:07d}",
                "requisition_id": req["requisition_id"],
                "business_unit": req["business_unit"],
                "department": req["department"],
                "job_family": req["job_family"],
                "job_level": req["job_level"],
                "source": src,
                "source_quality": q,
                "stage_reached": STAGES[stage_idx],
                "stage_index": stage_idx,
                "offer_status": None,
                **dates,
            })

        # ---- offer stage: sequential, one at a time
        pool = [c for c in batch if c["stage_index"] == 3]
        pool.sort(key=lambda c: -c["source_quality"])
        if filled:
            n_offers, accepted = 0, False
            while n_offers < min(3, len(pool)):
                c = pool[n_offers]
                n_offers += 1
                p_acc = float(np.clip(p_accept_req * c["source_quality"] ** 0.25, 0.15, 0.97))
                accepted = RNG.random() < p_acc or n_offers == min(3, len(pool))
                c["stage_index"] = 5 if accepted else 4
                c["stage_reached"] = STAGES[c["stage_index"]]
                c["offer_status"] = "Accepted" if accepted else "Declined"
                c["offer_date"] = c["onsite_date"] + timedelta(days=int(np.clip(RNG.normal(8, 3), 1, 30)))
                c["decision_date"] = c["offer_date"] + timedelta(days=int(np.clip(RNG.normal(6, 3), 1, 25)))
                if accepted:
                    break
        elif pool and RNG.random() < 0.26:
            c = pool[0]
            c["stage_index"] = 4
            c["stage_reached"] = STAGES[4]
            c["offer_date"] = c["onsite_date"] + timedelta(days=int(np.clip(RNG.normal(8, 3), 1, 30)))
            if RNG.random() < 0.45:
                c["offer_status"] = "Declined"
                c["decision_date"] = c["offer_date"] + timedelta(days=int(np.clip(RNG.normal(6, 3), 1, 25)))
            else:
                c["offer_status"] = "Pending"

        for c in batch:
            c.pop("source_quality", None)
        cands.extend(batch)

    # ---- filled requisitions, one per external hire
    for h in hire_events:
        if h["source"] == "Internal Mobility":
            continue
        rid += 1
        ttf = base_time_to_fill(h["business_unit"], h["job_family"], h["job_level"])
        opened = h["hire_date"] - timedelta(days=int(max(np.clip(RNG.normal(ttf, 14), 8, 240), 8)) + 30)
        ttf_actual = int(np.clip(RNG.normal(ttf + ttf_trend(h["business_unit"], opened), 13), 7, 260))
        accept_date = opened + timedelta(days=ttf_actual)
        req = {
            "requisition_id": f"R{rid:06d}", "business_unit": h["business_unit"],
            "department": h["department"], "job_family": h["job_family"],
            "job_level": h["job_level"], "location": str(RNG.choice(LOCATIONS, p=LOCATION_W)),
            "opened_date": opened, "status": "Filled",
            "offer_accepted_date": accept_date, "closed_date": h["hire_date"],
            "is_critical": bool(RNG.random() < (0.30 if h["business_unit"] == "Engineering" else 0.16)),
            "time_to_fill_days": ttf_actual,
        }
        reqs.append(req)
        make_candidates(req, filled=True)

    # ---- currently open requisitions
    open_plan = {"Engineering": 48, "Commerce": 26, "Operations": 30,
                 "Customer Success": 19, "Marketing": 9, "Corporate": 11}
    for bu, n in open_plan.items():
        cfg = STRUCTURE[bu]
        for _ in range(n):
            rid += 1
            fam = str(RNG.choice(cfg["job_families"]))
            lvl = str(RNG.choice(LEVELS, p=LEVEL_WEIGHTS))
            age = int(np.clip(RNG.gamma(2.3, 26 if bu != "Engineering" else 38), 3, 300))
            opened = AS_OF - timedelta(days=age)
            crit_p = 0.42 if bu == "Engineering" else 0.18
            req = {
                "requisition_id": f"R{rid:06d}", "business_unit": bu,
                "department": str(RNG.choice(cfg["departments"])), "job_family": fam,
                "job_level": lvl, "location": str(RNG.choice(LOCATIONS, p=LOCATION_W)),
                "opened_date": opened,
                "status": "Open" if RNG.random() > 0.10 else "On Hold",
                "offer_accepted_date": None, "closed_date": None,
                "is_critical": bool(RNG.random() < crit_p),
                "time_to_fill_days": None,
            }
            reqs.append(req)
            make_candidates(req, filled=False)

    # ---- a few cancelled reqs
    for _ in range(38):
        rid += 1
        bu = str(RNG.choice(list(STRUCTURE)))
        cfg = STRUCTURE[bu]
        opened = AS_OF - timedelta(days=int(RNG.integers(40, 500)))
        reqs.append({
            "requisition_id": f"R{rid:06d}", "business_unit": bu,
            "department": str(RNG.choice(cfg["departments"])),
            "job_family": str(RNG.choice(cfg["job_families"])),
            "job_level": str(RNG.choice(LEVELS, p=LEVEL_WEIGHTS)),
            "location": str(RNG.choice(LOCATIONS, p=LOCATION_W)),
            "opened_date": opened, "status": "Cancelled",
            "offer_accepted_date": None,
            "closed_date": opened + timedelta(days=int(RNG.integers(20, 160))),
            "is_critical": False, "time_to_fill_days": None,
        })

    return pd.DataFrame(reqs), pd.DataFrame(cands)


# --------------------------------------------------------------------------
# 4. Engagement survey + open text
# --------------------------------------------------------------------------
THEMES = {
    "Workload & Capacity": (
        ["The team is carrying too much with the headcount we have.",
         "Volumes keep going up but we have not added people.",
         "I am covering two roles since the last round of exits.",
         "Weekend coverage is becoming the norm rather than the exception."],
        ["Workload has been manageable this quarter.",
         "Staffing levels finally feel about right for the volume."]),
    "Career Growth": (
        ["It is not clear what the path to the next level looks like.",
         "Promotion decisions feel opaque and slow.",
         "I would like more stretch work but there is no bandwidth for it."],
        ["My manager built a real development plan with me.",
         "I moved into a bigger scope role this year."]),
    "Manager Effectiveness": (
        ["My manager is stretched too thin to give useful feedback.",
         "Direction changes week to week without explanation."],
        ["My manager is genuinely supportive and unblocks me quickly.",
         "One-to-ones are consistent and useful."]),
    "Compensation & Benefits": (
        ["Pay has not kept pace with the market for this role.",
         "The bonus structure is hard to understand."],
        ["The benefits package is competitive.",
         "This year's review felt fair."]),
    "Tools & Process": (
        ["Too many handoffs between systems to get anything done.",
         "Our tooling makes routine tasks slower than they should be."],
        ["The new workflow tooling saved us real time.",
         "Process changes this quarter actually reduced rework."]),
    "Recognition": (
        ["Good work goes unnoticed outside the immediate team.",
         "Recognition is inconsistent between teams."],
        ["Leadership called out the team's work publicly.",
         "Peer recognition here is strong."]),
    "Work Location Flexibility": (
        ["The onsite expectation does not match how the team actually works.",
         "Commute requirements are hard with the current schedule."],
        ["Hybrid arrangements work well for my team.",
         "Flexibility is one of the best things about working here."]),
    "Cross-team Collaboration": (
        ["Dependencies on other teams stall our work for weeks.",
         "Priorities are not aligned across functions."],
        ["Partnership with the other functions has improved a lot.",
         "Cross-team planning is much clearer this quarter."]),
}


def survey_periods() -> list[str]:
    out = []
    y, q = AS_OF.year, (AS_OF.month - 1) // 3 + 1
    for _ in range(8):
        out.append(f"{y}Q{q}")
        q -= 1
        if q == 0:
            q, y = 4, y - 1
    return list(reversed(out))


def build_survey(emp: pd.DataFrame) -> pd.DataFrame:
    rows, rid = [], 0
    periods = survey_periods()
    active = emp[emp["termination_date"].isna()]
    for pi_, period in enumerate(periods):
        prog = pi_ / (len(periods) - 1)
        sample = active.sample(frac=0.34, random_state=100 + pi_)
        for _, e in sample.iterrows():
            bu = e["business_unit"]
            base = {"Commerce": 3.92, "Operations": 3.55, "Engineering": 3.80,
                    "Customer Success": 3.86, "Marketing": 3.74, "Corporate": 3.95}[bu]
            drift = 0.0
            if bu == "Customer Success":
                drift = -0.42 * prog          # engagement declining
            elif bu == "Operations":
                drift = -0.22 * prog
            elif bu == "Commerce":
                drift = 0.14 * prog
            eng = float(np.clip(RNG.normal(base + drift, 0.62), 1, 5))

            workload = float(np.clip(RNG.normal(
                eng - (0.85 if bu in ("Customer Success", "Operations") else 0.25), 0.6), 1, 5))
            manager = float(np.clip(RNG.normal(eng + 0.12, 0.65), 1, 5))
            growth = float(np.clip(RNG.normal(eng - 0.25, 0.7), 1, 5))
            enps = int(np.clip(round(eng * 2.1 + RNG.normal(0, 1.6)), 0, 10))

            # theme selection is conditioned on the unit and the scores
            theme_w = {t: 1.0 for t in THEMES}
            if bu in ("Customer Success", "Operations"):
                theme_w["Workload & Capacity"] = 3.4
            if bu == "Engineering":
                theme_w["Career Growth"] = 2.0
                theme_w["Tools & Process"] = 1.9
            if bu == "Marketing":
                theme_w["Cross-team Collaboration"] = 2.1
            if bu == "Operations":
                theme_w["Manager Effectiveness"] = 2.2
            keys = list(theme_w)
            probs = np.array([theme_w[k] for k in keys], dtype=float)
            theme = str(RNG.choice(keys, p=probs / probs.sum()))

            neg, pos = THEMES[theme]
            p_neg = float(np.clip(0.94 - 0.19 * eng, 0.05, 0.92))
            negative = RNG.random() < p_neg
            comment = str(RNG.choice(neg if negative else pos))
            sentiment = "Negative" if negative else ("Positive" if eng >= 3.6 else "Neutral")

            rid += 1
            rows.append({
                "response_id": f"S{rid:07d}", "survey_period": period,
                "business_unit": bu, "department": e["department"],
                "job_family": e["job_family"], "job_level": e["job_level"],
                "location": e["location"], "tenure_band": tenure_band(e["hire_date"], AS_OF),
                "engagement_score": round(eng, 2), "manager_score": round(manager, 2),
                "workload_score": round(workload, 2), "growth_score": round(growth, 2),
                "enps_score": enps, "theme": theme, "sentiment": sentiment,
                "comment_text": comment,
            })
    return pd.DataFrame(rows)


def tenure_band(hire: date, asof: date) -> str:
    yrs = (asof - hire).days / 365.25
    if yrs < 1:
        return "<1 yr"
    if yrs < 2:
        return "1-2 yrs"
    if yrs < 5:
        return "2-5 yrs"
    if yrs < 10:
        return "5-10 yrs"
    return "10+ yrs"


# --------------------------------------------------------------------------
# 5. Seeded data-quality defects
# --------------------------------------------------------------------------
def inject_defects(emp: pd.DataFrame, cand: pd.DataFrame, req: pd.DataFrame,
                   survey: pd.DataFrame):
    log = {}

    # missing department
    idx = emp.sample(31, random_state=11).index
    emp.loc[idx, "department"] = None
    log["missing_department"] = len(idx)

    # terminated but no termination date
    term = emp[emp["termination_type"].notna()]
    idx = term.sample(42, random_state=12).index
    emp.loc[idx, "termination_date"] = pd.NaT
    log["missing_termination_date"] = len(idx)

    # termination before hire
    idx = term.drop(idx).sample(8, random_state=13).index
    emp.loc[idx, "termination_date"] = [
        h - timedelta(days=int(d))
        for h, d in zip(emp.loc[idx, "hire_date"], RNG.integers(5, 400, size=len(idx)))
    ]
    log["termination_before_hire"] = len(idx)

    # duplicate employee ids
    dupes = emp.sample(12, random_state=14).copy()
    dupes["location"] = "Unknown"
    emp = pd.concat([emp, dupes], ignore_index=True)
    log["duplicate_employee_id"] = len(dupes)

    # future-dated hires
    idx = emp.sample(6, random_state=15).index
    emp.loc[idx, "hire_date"] = [
        AS_OF + timedelta(days=int(d)) for d in RNG.integers(10, 90, size=len(idx))
    ]
    log["future_hire_date"] = len(idx)

    # duplicate candidate applications
    cdupes = cand.sample(17, random_state=16).copy()
    cand = pd.concat([cand, cdupes], ignore_index=True)
    log["duplicate_candidate"] = len(cdupes)

    # absurd time-to-fill outliers
    filled = req[req["status"] == "Filled"]
    idx = filled.sample(9, random_state=17).index
    req.loc[idx, "time_to_fill_days"] = RNG.integers(420, 900, size=len(idx))
    log["time_to_fill_outlier"] = len(idx)

    # missing engagement scores
    idx = survey.sample(64, random_state=18).index
    survey.loc[idx, "engagement_score"] = np.nan
    log["missing_engagement_score"] = len(idx)

    return emp, cand, req, survey, log


# --------------------------------------------------------------------------
# 6. Build the warehouse
# --------------------------------------------------------------------------
DDL_VIEWS = f"""
-- ============================================================================
-- Governed views. Every agent query runs against these, never the raw tables.
-- Records failing a hard data-quality rule are quarantined here exactly once,
-- so that every metric in the product reconciles to the same population.
-- ============================================================================

CREATE OR REPLACE VIEW v_employee_exclusions AS
WITH dup AS (
    SELECT employee_id FROM employees GROUP BY employee_id HAVING COUNT(*) > 1
)
SELECT DISTINCT e.employee_id,
       CASE
         WHEN e.employee_id IN (SELECT employee_id FROM dup) THEN 'Duplicate employee_id'
         WHEN e.hire_date IS NULL                            THEN 'Missing hire date'
         WHEN e.hire_date > DATE '{config.AS_OF_DATE}'       THEN 'Future-dated hire'
         WHEN e.termination_date IS NOT NULL
              AND e.termination_date < e.hire_date           THEN 'Termination before hire'
         WHEN e.termination_type IS NOT NULL
              AND e.termination_date IS NULL                 THEN 'Terminated with no termination date'
       END AS exclusion_reason
FROM employees e
WHERE e.employee_id IN (SELECT employee_id FROM dup)
   OR e.hire_date IS NULL
   OR e.hire_date > DATE '{config.AS_OF_DATE}'
   OR (e.termination_date IS NOT NULL AND e.termination_date < e.hire_date)
   OR (e.termination_type IS NOT NULL AND e.termination_date IS NULL);

CREATE OR REPLACE VIEW v_employees AS
SELECT e.* EXCLUDE (department),
       COALESCE(e.department, 'Unassigned')                     AS department,
       (e.department IS NULL)                                   AS department_imputed,
       (e.termination_date IS NULL)                             AS is_active,
       DATE_DIFF('day', e.hire_date,
                 COALESCE(e.termination_date, DATE '{config.AS_OF_DATE}')) / 365.25 AS tenure_years,
       CASE
         WHEN DATE_DIFF('day', e.hire_date, COALESCE(e.termination_date, DATE '{config.AS_OF_DATE}'))/365.25 < 1  THEN '<1 yr'
         WHEN DATE_DIFF('day', e.hire_date, COALESCE(e.termination_date, DATE '{config.AS_OF_DATE}'))/365.25 < 2  THEN '1-2 yrs'
         WHEN DATE_DIFF('day', e.hire_date, COALESCE(e.termination_date, DATE '{config.AS_OF_DATE}'))/365.25 < 5  THEN '2-5 yrs'
         WHEN DATE_DIFF('day', e.hire_date, COALESCE(e.termination_date, DATE '{config.AS_OF_DATE}'))/365.25 < 10 THEN '5-10 yrs'
         ELSE '10+ yrs'
       END AS tenure_band
FROM employees e
WHERE e.employee_id NOT IN (SELECT employee_id FROM v_employee_exclusions);

CREATE OR REPLACE VIEW v_month_spine AS
SELECT DISTINCT (DATE_TRUNC('month', d) + INTERVAL 1 MONTH - INTERVAL 1 DAY)::DATE AS month_end
FROM (SELECT UNNEST(GENERATE_SERIES(DATE '{WINDOW_START.isoformat()}',
                                    DATE '{config.AS_OF_DATE}',
                                    INTERVAL 1 DAY)) AS d);

-- Employee-month fact: one row per active employee per month end.
CREATE OR REPLACE VIEW v_headcount_monthly AS
SELECT s.month_end, e.business_unit, e.department,
       e.job_family, e.job_level, e.location, e.work_model, e.tenure_band, e.employee_id
FROM v_month_spine s
JOIN v_employees e
  ON e.hire_date <= s.month_end
 AND (e.termination_date IS NULL OR e.termination_date > s.month_end);

-- Movement fact: hires and exits by month.
CREATE OR REPLACE VIEW v_movement_monthly AS
SELECT (DATE_TRUNC('month', hire_date) + INTERVAL 1 MONTH - INTERVAL 1 DAY)::DATE AS month_end,
       business_unit, department, job_family, job_level,
       'Hire' AS event_type, NULL AS termination_type, employee_id
FROM v_employees
WHERE hire_date >= DATE '{WINDOW_START.isoformat()}'
UNION ALL
SELECT (DATE_TRUNC('month', termination_date) + INTERVAL 1 MONTH - INTERVAL 1 DAY)::DATE AS month_end,
       business_unit, department, job_family, job_level,
       'Exit' AS event_type, termination_type, employee_id
FROM v_employees
WHERE termination_date >= DATE '{WINDOW_START.isoformat()}';

CREATE OR REPLACE VIEW v_requisitions AS
SELECT r.*,
       CASE WHEN r.status IN ('Open', 'On Hold')
            THEN DATE_DIFF('day', r.opened_date, DATE '{config.AS_OF_DATE}') END AS req_age_days,
       (r.time_to_fill_days IS NOT NULL AND r.time_to_fill_days > 365) AS is_ttf_outlier
FROM requisitions r;

CREATE OR REPLACE VIEW v_candidates AS
SELECT * EXCLUDE (rn) FROM (
  SELECT c.*,
         ROW_NUMBER() OVER (PARTITION BY c.candidate_id ORDER BY c.applied_date) AS rn
  FROM candidates c
) WHERE rn = 1;

CREATE OR REPLACE VIEW v_survey AS
SELECT * FROM survey_responses WHERE engagement_score IS NOT NULL;

CREATE OR REPLACE VIEW v_internal_moves AS
SELECT m.* FROM internal_moves m
JOIN v_employees e USING (employee_id);
"""


def main() -> None:
    print("Simulating workforce ...")
    emp, hires = simulate_workforce()
    print(f"  {len(emp):,} employee records, {len(hires):,} hire events")

    print("Building recruiting funnel ...")
    req, cand = build_recruiting(hires)
    print(f"  {len(req):,} requisitions, {len(cand):,} candidate records")

    print("Building engagement survey ...")
    survey = build_survey(emp)
    print(f"  {len(survey):,} survey responses")

    print("Building internal mobility ...")
    moves = build_internal_moves(emp)
    print(f"  {len(moves):,} internal moves")

    print("Injecting data-quality defects ...")
    emp, cand, req, survey, defects = inject_defects(emp, cand, req, survey)
    for k, v in defects.items():
        print(f"  {k}: {v}")

    for df, cols in [(emp, ["hire_date", "termination_date"]),
                     (req, ["opened_date", "closed_date", "offer_accepted_date"]),
                     (cand, ["applied_date", "screen_date", "hm_screen_date",
                             "onsite_date", "offer_date", "decision_date"]),
                     (moves, ["move_date"])]:
        for c in cols:
            df[c] = pd.to_datetime(df[c]).dt.date

    config.WAREHOUSE.parent.mkdir(parents=True, exist_ok=True)
    if config.WAREHOUSE.exists():
        config.WAREHOUSE.unlink()
    con = duckdb.connect(str(config.WAREHOUSE))
    for name, df in [("employees", emp), ("requisitions", req), ("candidates", cand),
                     ("survey_responses", survey), ("internal_moves", moves)]:
        con.register("_tmp", df)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM _tmp")
        con.unregister("_tmp")
    con.execute(f"CREATE TABLE meta_asof AS SELECT DATE '{config.AS_OF_DATE}' AS as_of_date")
    con.execute(DDL_VIEWS)

    hc = con.execute("SELECT COUNT(*) FROM v_employees WHERE is_active").fetchone()[0]
    att = con.execute("""
        SELECT ROUND(100.0 * SUM(CASE WHEN termination_type='Voluntary' THEN 1 ELSE 0 END)
               / NULLIF((SELECT COUNT(*) FROM v_employees WHERE is_active),0), 1)
        FROM v_employees
        WHERE termination_date >= DATE '""" + (AS_OF - timedelta(days=365)).isoformat() + "'").fetchone()[0]
    print(f"\nWarehouse written -> {config.WAREHOUSE}")
    print(f"  active headcount: {hc:,}   trailing-12m voluntary attrition: {att}%")
    print(con.execute("""
        SELECT business_unit,
               COUNT(*) FILTER (WHERE is_active) AS headcount,
               COUNT(*) FILTER (WHERE termination_type='Voluntary'
                                AND termination_date >= DATE '""" + (AS_OF - timedelta(days=365)).isoformat() + """') AS vol_exits
        FROM v_employees GROUP BY 1 ORDER BY 1""").df().to_string(index=False))
    con.close()


if __name__ == "__main__":
    main()
