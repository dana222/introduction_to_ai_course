"""
Knowledge Base - Logic Reasoning Module
SE444 - Artificial Intelligence Course Project

Upgraded to support:
- Propositional logic (original)
- First-Order Logic (BONUS)
    ✓ Predicates
    ✓ Variables
    ✓ Unification
    ✓ FOL forward chaining inference
"""

from typing import Set, List, Dict, Tuple
import re


# --------------------------
# Predicate Parsing Utilities
# --------------------------

def parse_predicate(fact: str) -> Tuple[str, Tuple]:
    """
    Parse a predicate string into (name, args).
    Example: "Safe(2,3)" → ("Safe", ("2", "3"))
             "Free(x,y)" → ("Free", ("x", "y"))
    """
    match = re.match(r"(\w+)\((.*)\)", fact)
    if not match:
        return fact, ()  # simple propositional atom

    name = match.group(1)
    args = tuple(arg.strip() for arg in match.group(2).split(","))
    return name, args


def is_variable(x: str) -> bool:
    """
    Variables start with lowercase letters.
    Example: x, y, pos, a
    """
    return isinstance(x, str) and x and x[0].islower()


def unify(a, b, subs=None):
    """
    Unify two predicate argument lists.
    Example:
        a = ("x", "3")
        b = ("2", "3")
        → subs = { "x": "2" }
    """
    if subs is None:
        subs = {}

    for arg_a, arg_b in zip(a, b):

        if arg_a == arg_b:  # identical constants
            continue

        # arg_a is variable
        if is_variable(arg_a):
            if arg_a in subs:
                if subs[arg_a] != arg_b:
                    return None
            else:
                subs[arg_a] = arg_b
            continue

        # arg_b is variable
        if is_variable(arg_b):
            if arg_b in subs:
                if subs[arg_b] != arg_a:
                    return None
            else:
                subs[arg_b] = arg_a
            continue

        # mismatch constants → cannot unify
        return None

    return subs


# --------------------------
# Knowledge Base
# --------------------------

class KnowledgeBase:
    """
    Knowledge base supporting:
    - Propositional logic
    - First-Order Logic (BONUS)
    """

    def __init__(self):
        self.facts = set()        # stores strings
        self.rules = []           # (premises, conclusion)

    # --------------------------
    # Adding facts
    # --------------------------
    def tell(self, fact: str):
        if fact not in self.facts:
            self.facts.add(fact)
            self.infer()

    # --------------------------
    # Adding rules
    # --------------------------
    def add_rule(self, premises: List[str], conclusion: str):
        self.rules.append((premises, conclusion))

    # --------------------------
    # ASK (propositional)
    # --------------------------
    def ask(self, query: str) -> bool:
        return query in self.facts

    # --------------------------
    # Forward Chaining (FOL Version)
    # --------------------------
    def infer(self):
        """
        BONUS:
        Full FOL forward chaining using unification.
        Example:
            Rule: Free(x,y) AND Safe(x,y) → CanMove(x,y)
            Facts: Free(1,3), Safe(1,3)
            Derived: CanMove(1,3)
        """
        changed = True
        while changed:
            changed = False

            for premises, conclusion in self.rules:

                # Try to find substitutions that satisfy all premises
                subs_list = [{}]  # list of possible substitutions

                for prem in premises:
                    pred_name, pred_args = parse_predicate(prem)

                    new_subs_list = []

                    for fact in self.facts:
                        f_name, f_args = parse_predicate(fact)

                        if pred_name != f_name or len(pred_args) != len(f_args):
                            continue

                        for subs in subs_list:
                            new_subs = unify(
                                tuple(subs.get(a, a) for a in pred_args),
                                f_args,
                                subs.copy()
                            )
                            if new_subs is not None:
                                new_subs_list.append(new_subs)

                    subs_list = new_subs_list

                # No valid variable bindings → skip rule
                if not subs_list:
                    continue

                # Apply substitutions to conclusion
                conc_name, conc_args = parse_predicate(conclusion)

                for subs in subs_list:
                    instantiated_args = tuple(subs.get(a, a) for a in conc_args)
                    new_fact = f"{conc_name}({','.join(instantiated_args)})"

                    if new_fact not in self.facts:
                        self.facts.add(new_fact)
                        changed = True

    # --------------------------
    # Helpers
    # --------------------------
    def clear_facts(self):
        self.facts.clear()

    def get_facts(self) -> Set[str]:
        return self.facts.copy()

    def get_rules(self) -> List:
        return self.rules.copy()

    def __str__(self):
        facts_str = "\n  ".join(sorted(self.facts))
        rules_str = "\n  ".join(
            [f"{' AND '.join(p)} → {c}" for p, c in self.rules]
        )
        return f"Knowledge Base:\nFacts ({len(self.facts)}):\n  {facts_str}\nRules ({len(self.rules)}):\n  {rules_str}"
