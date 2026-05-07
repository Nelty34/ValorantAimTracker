import json
import matplotlib.pyplot as plt

# Load data
with open("users.json", "r") as f:
    data = json.load(f)

test_group = ["Shortbread", "Natt", "Nut"]
control_group = ["Netly", "Speegy", "Nucleic"]

def get_first_last(user, metric):
    matches = [m for m in data[user]["matches"] if m]
    if len(matches) < 2:
        return None, None
    return matches[0].get(metric, 0), matches[-1].get(metric, 0)

def get_eff(match):
    if match["total_engagements"] > 0:
        return match["engagements_with_shots"] / match["total_engagements"]
    return 0

def get_first_last_eff(user):
    matches = [m for m in data[user]["matches"] if m]
    if len(matches) < 2:
        return None, None
    return get_eff(matches[0]), get_eff(matches[-1])

def avg(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals)/len(vals) if vals else 0

def build_graph(metric_name, get_func, ylabel, filename):
    test_initial = []
    test_final = []
    control_initial = []
    control_final = []

    for u in test_group:
        i, f = get_func(u)
        if i is not None:
            test_initial.append(i)
            test_final.append(f)

    for u in control_group:
        i, f = get_func(u)
        if i is not None:
            control_initial.append(i)
            control_final.append(f)

    labels = ["Initial", "Final"]

    test_vals = [avg(test_initial), avg(test_final)]
    control_vals = [avg(control_initial), avg(control_final)]

    x = range(len(labels))

    plt.figure()
    plt.plot(x, test_vals, marker='o', label="Test Group")
    plt.plot(x, control_vals, marker='o', label="Control Group")

    plt.xticks(x, labels)
    plt.title(f"{metric_name} Improvement")
    plt.ylabel(ylabel)
    plt.legend()

    plt.savefig(filename)

# -------- GRAPHS --------

build_graph(
    "ACS",
    lambda u: get_first_last(u, "acs"),
    "ACS",
    "acs_improvement.png"
)

build_graph(
    "Kills",
    lambda u: get_first_last(u, "kills"),
    "Kills",
    "kills_improvement.png"
)

build_graph(
    "Reaction Time",
    lambda u: get_first_last(u, "avg_reaction_time"),
    "Seconds",
    "reaction_time_improvement.png"
)

build_graph(
    "Engagement Efficiency",
    get_first_last_eff,
    "Shots per Engagement",
    "engagement_improvement.png"
)

print("Improvement graphs generated.")