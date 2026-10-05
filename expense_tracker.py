"""
Expense Tracker - Python Mini Project
Features: add, view, delete expenses, category summary, saved to a JSON file.
Run with:  python expense_tracker.py
"""

import json
import os
from datetime import date

DATA_FILE = "expenses.json"


# ---------- File handling ----------
def load_expenses():
    """Load expenses from disk. Returns an empty list if no file exists."""
    if not os.path.exists(DATA_FILE):
        return []
    try:
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        print("Warning: could not read saved data. Starting fresh.")
        return []


def save_expenses(expenses):
    """Write expenses to disk."""
    with open(DATA_FILE, "w") as f:
        json.dump(expenses, f, indent=2)


# ---------- Input helpers ----------
def get_amount(prompt):
    """Keep asking until the user enters a positive number."""
    while True:
        try:
            value = float(input(prompt))
            if value > 0:
                return value
            print("Amount must be greater than 0.")
        except ValueError:
            print("Please enter a valid number.")


# ---------- Core features ----------
def add_expense(expenses):
    description = input("Description: ").strip() or "Unnamed"
    category = input("Category (e.g. Food, Travel, Bills): ").strip().title() or "Other"
    amount = get_amount("Amount: ")
    expenses.append({
        "date": str(date.today()),
        "description": description,
        "category": category,
        "amount": amount,
    })
    save_expenses(expenses)
    print(f"Added: {description} - {amount:.2f} ({category})")


def view_expenses(expenses):
    if not expenses:
        print("No expenses recorded yet.")
        return
    print(f"\n{'#':<4}{'Date':<12}{'Description':<22}{'Category':<12}{'Amount':>10}")
    print("-" * 60)
    for i, e in enumerate(expenses, start=1):
        print(f"{i:<4}{e['date']:<12}{e['description'][:20]:<22}"
              f"{e['category']:<12}{e['amount']:>10.2f}")
    print("-" * 60)
    print(f"{'Total':<50}{sum(e['amount'] for e in expenses):>10.2f}")


def delete_expense(expenses):
    view_expenses(expenses)
    if not expenses:
        return
    try:
        num = int(input("Enter the # to delete (0 to cancel): "))
    except ValueError:
        print("Invalid input.")
        return
    if 1 <= num <= len(expenses):
        removed = expenses.pop(num - 1)
        save_expenses(expenses)
        print(f"Deleted: {removed['description']}")
    elif num != 0:
        print("No expense with that number.")


def show_summary(expenses):
    if not expenses:
        print("No expenses recorded yet.")
        return
    totals = {}
    for e in expenses:
        totals[e["category"]] = totals.get(e["category"], 0) + e["amount"]
    grand_total = sum(totals.values())

    print("\nSpending by category")
    print("-" * 40)
    for cat, amt in sorted(totals.items(), key=lambda x: x[1], reverse=True):
        percent = amt / grand_total * 100
        bar = "#" * int(percent // 5)
        print(f"{cat:<12}{amt:>9.2f}  {percent:5.1f}%  {bar}")
    print("-" * 40)
    print(f"{'Total':<12}{grand_total:>9.2f}")


# ---------- Main menu ----------
def main():
    expenses = load_expenses()
    actions = {
        "1": ("Add expense", add_expense),
        "2": ("View expenses", view_expenses),
        "3": ("Delete expense", delete_expense),
        "4": ("Category summary", show_summary),
    }

    while True:
        print("\n=== Expense Tracker ===")
        for key, (label, _) in actions.items():
            print(f"{key}. {label}")
        print("5. Exit")

        choice = input("Choose an option: ").strip()
        if choice == "5":
            print("Goodbye!")
            break
        elif choice in actions:
            actions[choice][1](expenses)
        else:
            print("Invalid choice, try again.")


if __name__ == "__main__":
    main()
