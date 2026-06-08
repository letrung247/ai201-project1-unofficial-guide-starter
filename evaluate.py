"""
Milestone 6 — Run the 5 evaluation-plan questions end to end and print the
question, the system's answer, the sources cited, and the top retrieval
distances so the results can be recorded and judged in the README.

Run:  python evaluate.py
"""

from query import ask

EVAL_QUESTIONS = [
    "What are the dining hall hours on weekends?",
    "What do students say about food quality in the dining halls?",
    "What meal plan options are available for the 2025-26 academic year?",
    "Which dining location has the shortest wait times according to student reviews?",
    "What accommodations are available for dietary restrictions like vegetarian or gluten-free meals?",
]


def main():
    for i, q in enumerate(EVAL_QUESTIONS, 1):
        result = ask(q)
        print(f"\n{'='*74}\nQ{i}: {q}\n{'='*74}")
        print("ANSWER:\n" + result["answer"].encode("ascii", "replace").decode())
        print("\nSOURCES:", result["sources"])
        print("TOP RETRIEVED (source #idx : distance):")
        for c in result["chunks"][:6]:
            print(f"  {c['source']} #{c['chunk_index']} : {c['distance']:.3f}")


if __name__ == "__main__":
    main()
