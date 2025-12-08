"""
Knowledge Base with First-Order Logic (Supports Variables)
SE444 - AI Course Project Bonus Extension
"""

from typing import List, Dict, Optional, Tuple
import re


def parse_predicate(expr: str) -> Tuple[str, Tuple[str]]:
    """
    Convert 'Safe(1,2)' → ('Safe', ('1','2'))
    """
    name, args = expr.split("(")
    args = args[:-1]  # remove ')'
    args = tuple(a.strip() for a in args.split(","))
    return name, args


def unify(a, b, subst: Dict[str, str]) -> Optional[Dict[str, str]]:
    """
    Unify arguments with variables.
    Example:
        unify(("X","Y"), ("1","2")) → {"X":"1","Y":"2"}
    """
    if subst is None:
        return None

    if a == b:
        return subst

    # Variable case
    if re.fullmatch(r"[A-Z]+", a):  # variable is uppercase
        return unify_var(a, b, subst)

    if re.fullmatch(r"[A-Z]+", b):
        return unify_var(b, a, subst)

    return None


def unify_var(var, value, subst):
    """Bind a variable."""
    if var in subst:
        return unify(subst[var], value, subst)
    elif value in subst:
        return unify(var, subst[value], subst)
    else:
        new_subst = subst.copy()
        new_subst[var] = value
        return new_subst


def unify_tuple(args1, args2, subst):
    """Unify tuple arguments (X,Y) with (1,2)."""
    for a, b in zip(args1, args2):
        subst = unify(a, b, subst)
        if subst is None:
            return None
    return subst


class KnowledgeBase:
    """First-Order Logic Knowledge Base with forward chaining."""

    def __init__(self):
        self.facts: List[str] = []  # factual predicates
        self.rules: List[Tuple[List[str], str]] = []  # (premises, conclusion)

    # --- Facts Methods ---
    def tell(self, fact: str):
        """Add a fact."""
        if fact not in self.facts:
            self.facts.append(fact)

    def clear_facts(self):
        """Remove all facts (rules stay)."""
        self.facts = []

    def get_facts(self) -> List[str]:
        return list(self.facts)

    # --- Rules Methods ---
    def add_rule(self, premises: List[str], conclusion: str):
        """Add a logical rule: premises → conclusion"""
        self.rules.append((premises, conclusion))

    def get_rules(self) -> List[Tuple[List[str], str]]:
        return list(self.rules)

    # --- Inference ---
    def ask(self, query: str) -> bool:
        """Check if KB can infer the query."""
        derived = self.forward_chain()
        return query in derived

    def forward_chain(self) -> List[str]:
        """Perform forward chaining to infer all possible facts."""
        inferred = set(self.facts)
        added = True

        while added:
            added = False

            for premises, conclusion in self.rules:
                # Start with empty substitution
                matches = [{}]

                for premise in premises:
                    new_matches = []
                    p_name, p_args = parse_predicate(premise)

                    for fact in self.facts:
                        f_name, f_args = parse_predicate(fact)
                        if p_name != f_name:
                            continue

                        for subst in matches:
                            result = unify_tuple(p_args, f_args, subst)
                            if result is not None:
                                new_matches.append(result)

                    matches = new_matches

                # Apply substitutions to conclusion
                for subst in matches:
                    c_name, c_args = parse_predicate(conclusion)
                    grounded = f"{c_name}({','.join(subst.get(arg, arg) for arg in c_args)})"

                    if grounded not in inferred:
                        inferred.add(grounded)
                        self.facts.append(grounded)
                        added = True

        return inferred
