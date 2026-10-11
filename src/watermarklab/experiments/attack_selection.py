"""Deterministic diversity-preserving subset of reviewer attack strengths.

For a limit of 7, select one strongest listed strength in each of seven
attack families; for limit of 14, add the lightest in each family. A limit
of zero selects the entire original suite, in original order.
"""
from __future__ import annotations
from collections import defaultdict


def stratified_attack_subset(attacks, maximum):
    attacks=list(attacks)
    if maximum is None or int(maximum)==0 or int(maximum)>=len(attacks):
        return attacks
    if int(maximum)<0:raise ValueError('max_attacks must be nonnegative')
    groups=defaultdict(list)
    for attack in attacks: groups[str(attack.group)].append(attack)
    ordered=[]
    depth=0
    while len(ordered)<maximum:
        added=False
        for group in groups.values():
            if depth<len(group):
                # Sample extreme strength first, then mild, then inward.
                index=(-1-depth//2) if depth%2==0 else (depth//2)
                cand=group[index]
                if cand not in ordered:
                    ordered.append(cand);added=True
                    if len(ordered)>=maximum:break
        if not added: break
        depth+=1
    return ordered
