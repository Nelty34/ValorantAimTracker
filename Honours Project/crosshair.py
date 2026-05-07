import json
from collections import Counter

INPUT_FILE = "users.json"

TEST_GROUP = ["Shortbread", "Natt", "Nut"]
CONTROL_GROUP = ["Netly", "Speegy", "Nucleic"]

PLACEMENTS = [
    "LOWER than head",
    "ON LEVEL with head",
    "HIGHER than head"
]


def load_data():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def placement_percentages(match):
    placements = match.get("crosshair_placements", [])
    total = len(placements)

    if total == 0:
        return {p: 0 for p in PLACEMENTS}

    counts = Counter(placements)

    return {
        p: round((counts.get(p, 0) / total) * 100, 2)
        for p in PLACEMENTS
    }


def get_user_improvement(data, username):
    matches = data[username].get("matches", [])

    if len(matches) < 2:
        return None

    first = placement_percentages(matches[0])
    final = placement_percentages(matches[-1])

    improvement = {
        p: round(final[p] - first[p], 2)
        for p in PLACEMENTS
    }

    return {
        "first": first,
        "final": final,
        "improvement": improvement
    }


def average_group_improvement(results):
    avg = {}

    for placement in PLACEMENTS:
        values = [
            r["improvement"][placement]
            for r in results
            if r is not None
        ]

        avg[placement] = round(sum(values) / len(values), 2) if values else 0

    return avg


def print_group_results(group_name, users, data):
    print(f"\n{group_name}")
    print("-" * len(group_name))

    results = []

    for user in users:
        result = get_user_improvement(data, user)
        results.append(result)

        if result is None:
            print(f"\n{user}: Not enough matches")
            continue

        print(f"\n{user}")
        print(f"Initial:     {result['first']}")
        print(f"Final:       {result['final']}")
        print(f"Improvement: {result['improvement']}")

    avg = average_group_improvement(results)

    print(f"\n{group_name} Average Improvement:")
    print(avg)

    return avg

import matplotlib.pyplot as plt


def create_crosshair_graph(test_avg, control_avg):
    labels = PLACEMENTS

    test_values = [test_avg[p] for p in labels]
    control_values = [control_avg[p] for p in labels]

    x = range(len(labels))
    width = 0.35

    plt.figure(figsize=(9, 6))

    plt.bar([i - width / 2 for i in x], test_values, width, label="Feedback Group")
    plt.bar([i + width / 2 for i in x], control_values, width, label="Control Group")

    plt.axhline(0, linewidth=1)

    plt.xticks(x, labels, rotation=15)
    plt.ylabel("Average Percentage Change")
    plt.title("Crosshair Placement Improvement by Group")
    plt.legend()
    plt.tight_layout()

    plt.savefig("crosshair_placement_improvement.png", dpi=300)
    plt.show()


def main():
    data = load_data()

    test_avg = print_group_results("Test Group", TEST_GROUP, data)
    control_avg = print_group_results("Control Group", CONTROL_GROUP, data)

    print("\nGroup Comparison")
    print("----------------")

    for placement in PLACEMENTS:
        difference = round(test_avg[placement] - control_avg[placement], 2)
        print(f"{placement}: Test - Control = {difference}%")

    create_crosshair_graph(test_avg, control_avg)

    print("\nGraph saved as crosshair_placement_improvement.png")


if __name__ == "__main__":
    main()