"""Generator of a LARGE connected corpus, tuned to show the value of the graph.

Idea: many documents (~120) + long relation chains split across separate files
+ lots of look-alike distractors. On such a corpus:
  - vector (top-k=5) can't reach the far end of a chain — the needed document is
    not lexically similar to the question and drowns among the distractors;
  - graph follows the explicit edges and finds the answer.

Each fact is a short sentence in its own file (reliably extracted into graph
edges by the LLM). Questions and gold answers are derived from the structure —
so they are correct by construction.

Run:  python make_dataset.py   (overwrites data/astrolab/ and questions.yaml)
"""
import random
import shutil
from collections import defaultdict
from pathlib import Path

import yaml

SEED = 42
random.seed(SEED)

HERE = Path(__file__).parent
DATA = HERE / "astrolab_tmp"  # temp, moved into data/astrolab below
OUT = HERE / "data" / "astrolab"

N_VENDORS, N_TECHS, N_TEAMS, N_PROJECTS = 12, 24, 10, 40

VENDORS = ["Orbitek", "Supplyra", "Vendorix", "Greenflow", "Nordstock", "Quantline",
           "Ferrolab", "Helix-Trade", "Astra-Prom", "Billion-Parts", "Kronos", "Delta-Materials"]
TEAMS = ["Nebula", "Pulsar", "Quasar", "Comet", "Aurora", "Helios", "Titan", "Orion", "Vega", "Lyra"]
CITIES = ["Helsinki", "Warsaw", "Prague", "Tallinn", "Riga", "Vilnius", "Krakow", "Gdansk"]
TSPEC = ["sensors", "actuators", "batteries", "radio modules", "optics", "controllers"]
TLABELS = ["SLAM module", "orientation library", "drive controller", "telemetry stack",
           "path planner", "Kalman filter", "video codec", "mesh protocol",
           "power system", "battery model", "sensor calibration", "odometry"]
PLABELS = ["navigation", "telemetry", "power supply", "chassis", "communications",
           "mapping", "diagnostics", "autopilot", "charging", "sensing"]
FIRST = ["James", "Olivia", "Liam", "Emma", "Noah", "Ava", "William", "Sophia", "Mason", "Isabella",
         "Ethan", "Mia", "Lucas", "Charlotte", "Henry", "Amelia", "Jack", "Harper", "Owen", "Evelyn",
         "Daniel", "Abigail", "Leo", "Emily", "Nathan", "Grace", "Adam", "Chloe", "Ryan", "Zoe", "Mark"]
LAST = ["Smith", "Johnson", "Brown", "Taylor", "Wilson", "Davies", "Clark", "Hall", "Walker", "Young",
        "Allen", "King", "Wright", "Scott", "Green", "Baker", "Adams", "Nelson", "Hill", "Campbell",
        "Mitchell", "Roberts", "Carter", "Phillips", "Evans", "Turner", "Parker", "Collins", "Edwards", "Morris", "Cooper"]


def pick_people(n):
    combos = [f"{f} {l}" for f in FIRST for l in LAST]
    random.shuffle(combos)
    return combos[:n]


def build():
    people = pick_people(N_TEAMS + N_TEAMS * 2)  # leads + 2 engineers each
    leads = people[:N_TEAMS]
    engineers = people[N_TEAMS:]
    team_lead = {TEAMS[i]: leads[i] for i in range(N_TEAMS)}
    team_eng = {TEAMS[i]: [engineers[2 * i], engineers[2 * i + 1]] for i in range(N_TEAMS)}

    techs = [f"T{n:02d}" for n in range(1, N_TECHS + 1)]
    tech_label = {t: random.choice(TLABELS) for t in techs}
    tech_vendor = {t: random.choice(VENDORS) for t in techs}

    projects = [f"P{n:02d}" for n in range(1, N_PROJECTS + 1)]
    proj_label = {p: random.choice(PLABELS) for p in projects}
    proj_team = {p: random.choice(TEAMS) for p in projects}
    proj_tech = {p: random.choice(techs) for p in projects}
    proj_deps = {}
    for i, p in enumerate(projects):
        k = random.choice([0, 1, 1, 2, 2]) if i > 0 else 0
        proj_deps[p] = sorted(random.sample(projects[:i], min(k, i)))

    team_projects = defaultdict(list)
    for p in projects:
        team_projects[proj_team[p]].append(p)

    def closure(p, seen=None):
        seen = seen if seen is not None else set()
        for d in proj_deps[p]:
            if d not in seen:
                seen.add(d)
                closure(d, seen)
        return seen

    proj_closure = {p: closure(p) for p in projects}
    dependents = defaultdict(set)
    for p in projects:
        for x in proj_closure[p]:
            dependents[x].add(p)

    vendor_projects = defaultdict(set)
    for p in projects:
        vendor_projects[tech_vendor[proj_tech[p]]].add(p)

    d = dict(**locals())
    d["TEAMS"], d["VENDORS"] = TEAMS, VENDORS
    return d


def write_docs(g):
    if DATA.exists():
        shutil.rmtree(DATA)
    DATA.mkdir(parents=True)

    def w(name, text):
        (DATA / name).write_text(text.strip() + "\n", encoding="utf-8")

    w("00_overview.md",
      f"# Astrolab — overview\n\nAstrolab is a fictional company (demo corpus). It has "
      f"{N_TEAMS} teams, {N_PROJECTS} projects and {N_TECHS} technologies from {N_VENDORS} "
      f"vendors. Robots are assembled from many interdependent projects; the critical links "
      f"are split across separate documents.")

    for t in g["TEAMS"]:
        eng = ", ".join(g["team_eng"][t])
        projs = ", ".join(sorted(g["team_projects"][t])) or "—"
        w(f"team_{t}.md",
          f"# Team {t}\n\nThe lead of team {t} is {g['team_lead'][t]}.\n"
          f"Engineers on team {t}: {eng}.\nTeam {t} owns projects: {projs}.")

    for t in g["TEAMS"]:
        for e in g["team_eng"][t]:
            w(f"person_{e.replace(' ', '_')}.md",
              f"# {e}\n\n{e} is an engineer on team {t}. Their lead is {g['team_lead'][t]}.")

    for p in g["projects"]:
        deps = ", ".join(g["proj_deps"][p]) or "no other projects"
        w(f"project_{p}.md",
          f"# Project {p} ({g['proj_label'][p]})\n\nProject {p} is in the area of "
          f"{g['proj_label'][p]}. Owner — team {g['proj_team'][p]}. Project {p} uses "
          f"technology {g['proj_tech'][p]}. Project {p} depends on projects: {deps}.")

    for t in g["techs"]:
        w(f"tech_{t}.md",
          f"# Technology {t} ({g['tech_label'][t]})\n\nTechnology {t} is a "
          f"{g['tech_label'][t]}. The supplier of technology {t} is vendor {g['tech_vendor'][t]}.")

    for v in g["VENDORS"]:
        w(f"vendor_{v.replace(' ', '_')}.md",
          f"# Vendor {v}\n\nVendor {v} is based in {random.choice(CITIES)}. "
          f"Specialisation — {random.choice(TSPEC)}.")

    return len(list(DATA.glob('*.md')))


def build_questions(g):
    qs = []
    # factual (control — vector usually fine here)
    for t in random.sample(g["TEAMS"], 2):
        qs.append({"text": f"Who leads team {t}?", "type": "factual",
                   "reference": f"{g['team_lead'][t]}."})

    # bridge: project -> technology -> vendor (answer is 2 docs from the named project)
    bridge = random.sample(g["projects"], 5)
    for p in bridge:
        v = g["tech_vendor"][g["proj_tech"][p]]
        qs.append({"text": f"Which vendor supplies the technology used by project {p}?",
                   "type": "multihop",
                   "reference": f"Vendor {v} (via technology {g['proj_tech'][p]})."})

    # transitive dependencies (vector only sees direct deps from one document)
    deep = sorted(g["projects"], key=lambda p: len(g["proj_closure"][p]), reverse=True)[:3]
    for p in deep:
        cl = sorted(g["proj_closure"][p])
        qs.append({"text": f"Which projects does project {p} depend on, directly or indirectly?",
                   "type": "multihop",
                   "reference": ("Project {p} depends (directly or transitively) on: {lst}."
                                 .format(p=p, lst=", ".join(cl) if cl else "no projects"))})

    # global: aggregation over the whole graph
    top_vendor = max(g["vendor_projects"], key=lambda v: len(g["vendor_projects"][v]))
    qs.append({"text": "Which vendor's technologies are used by the most projects?",
               "type": "global",
               "reference": f"Vendor {top_vendor} — its technologies are used by "
                            f"{len(g['vendor_projects'][top_vendor])} projects."})
    pof = max(g["dependents"], key=lambda x: len(g["dependents"][x]))
    qs.append({"text": "Which project is the single point of failure — the one the most projects "
                       "depend on, directly or indirectly?",
               "type": "global",
               "reference": f"Project {pof} — {len(g['dependents'][pof])} projects depend on it "
                            f"(transitively)."})
    return qs


def main():
    g = build()
    n = write_docs(g)
    if OUT.exists():
        shutil.rmtree(OUT)
    shutil.move(str(DATA), str(OUT))
    qs = build_questions(g)
    (HERE / "questions.yaml").write_text(
        "# Auto-generated by make_dataset.py (seed=%d). Gold answers derived from the graph.\n" % SEED
        + yaml.safe_dump({"questions": qs}, allow_unicode=True, sort_keys=False),
        encoding="utf-8")
    print(f"Documents: {n}  ->  {OUT}")
    print(f"Questions: {len(qs)}  ->  {HERE/'questions.yaml'}")
    print("\nSample questions:")
    for q in qs[:4] + qs[-2:]:
        print(f"  [{q['type']:8}] {q['text']}")
        print(f"            -> {q['reference']}")


if __name__ == "__main__":
    main()
