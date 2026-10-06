from atomspace import STV, Atom

EPS = 1e-9                     # tolerance for floating point comparisons
MAX_CONF = 1.0 - 1e-9          # cap used when turning confidence into weight
SB_ONE = 0.9999                # deduction: sB above this counts as 1
INVERSION_FACTOR = 0.6         # confidence factor of inversion (from the repo)
INFERENCE_LINK_TYPES = {"InheritanceLink", "ImplicationLink"}

def clamp01(x):
    return max(0.0, min(1.0, x))

def no_info():
    """Truth value meaning 'no information' (strength is a placeholder)."""
    return STV(1.0, 0.0)

def is_no_info(stv):
    return stv.confidence == 0.0

def c2w(c):
    """Confidence to evidence weight: w = c / (1 - c)."""
    c = min(c, MAX_CONF)
    return c / (1.0 - c)

def w2c(w):
    """Evidence weight to confidence: c = w / (w + 1)."""
    return w / (w + 1.0)

def conditional_probability_consistent(sA, sB, sAB):
    """True if P(B|A) = sAB is possible given the base rates sA and sB."""
    if sA <= 0.0:
        return False
    smallest = clamp01((sA + sB - 1.0) / sA)
    largest = clamp01(sB / sA)
    return smallest - EPS <= sAB <= largest + EPS

def deduction(A, B, C, AB, BC):
    """A->B and B->C give A->C."""
    if not (conditional_probability_consistent(A.strength, B.strength, AB.strength)
            and conditional_probability_consistent(B.strength, C.strength, BC.strength)):
        return no_info()
    if B.strength > SB_ONE:
        s = C.strength
    else:
        s = (AB.strength * BC.strength
             + (1.0 - AB.strength) * (C.strength - B.strength * BC.strength)
             / (1.0 - B.strength))
    c = AB.strength * BC.strength * AB.confidence * BC.confidence
    return STV(clamp01(s), clamp01(c))

def induction(A, B, C, BA, BC):
    """B->A and B->C give A->C (B is the shared source)."""
    sA, sB, sC = A.strength, B.strength, C.strength
    if sA <= 0.0 or 1.0 - sB <= EPS:
        return no_info()
    t1 = BA.strength * BC.strength * sB / sA
    t2 = ((1.0 - BA.strength * sB / sA)
          * (sC - sB * BC.strength) / (1.0 - sB))
    x = BC.strength * BC.confidence * BA.confidence
    return STV(clamp01(t1 + t2), clamp01(w2c(x)))

def abduction(A, B, C, AB, CB):
    """A->B and C->B give A->C (B is the shared target). A is unused."""
    sB, sC = B.strength, C.strength
    if sB <= 0.0 or 1.0 - sB <= EPS:
        return no_info()
    s = (AB.strength * CB.strength * sC / sB
         + sC * (1.0 - AB.strength) * (1.0 - CB.strength) / (1.0 - sB))
    x = AB.strength * AB.confidence * CB.confidence
    return STV(clamp01(s), clamp01(w2c(x)))

def revision(t1, t2):
    """Merge two truth values of the same statement (independent evidence)."""
    w1 = c2w(t1.confidence)
    w2 = c2w(t2.confidence)
    w = w1 + w2
    if w <= 0.0:
        return STV(clamp01((t1.strength + t2.strength) / 2.0), 0.0)
    s = (w1 * t1.strength + w2 * t2.strength) / w
    c = min(1.0, max(w2c(w), t1.confidence, t2.confidence))
    return STV(clamp01(s), clamp01(c))

def inversion(A, B, AB):
    """From A->B get B->A using Bayes: sBA = sAB * sA / sB."""
    if B.strength <= 0.0:
        return no_info()
    s = AB.strength * A.strength / B.strength
    c = B.confidence * AB.confidence * INVERSION_FACTOR
    return STV(clamp01(s), clamp01(c))

def _is_binary_inference_link(atom):
    return (atom.is_link
            and len(atom.targets) == 2
            and atom.type in INFERENCE_LINK_TYPES)

def _chain_type(first_type, second_type):
    """Link type of a deduction result, or None if the chain is not allowed."""
    if first_type == second_type:
        return first_type
    if first_type == "ImplicationLink" and second_type == "InheritanceLink":
        return "ImplicationLink"      # option A
    return None

def deduction_link(ab, bc):
    """A->B and B->C give A->C. Returns an Atom, or None if the rule does not apply."""
    if not (_is_binary_inference_link(ab) and _is_binary_inference_link(bc)):
        return None
    A, B1 = ab.targets
    B2, C = bc.targets
    if B1 != B2 or A == C or A == B1 or B1 == C:
        return None
    result_type = _chain_type(ab.type, bc.type)
    if result_type is None:
        return None
    stv = deduction(A.stv, B1.stv, C.stv, ab.stv, bc.stv)
    return Atom(result_type, targets=(A, C), stv=stv)

def induction_link(ba, bc):
    """B->A and B->C give A->C. Both links must have the same type."""
    if not (_is_binary_inference_link(ba) and _is_binary_inference_link(bc)):
        return None
    if ba.type != bc.type:
        return None
    B1, A = ba.targets
    B2, C = bc.targets
    if B1 != B2 or A == C or A == B1 or C == B1:
        return None
    stv = induction(A.stv, B1.stv, C.stv, ba.stv, bc.stv)
    return Atom(ba.type, targets=(A, C), stv=stv)

def abduction_link(ab, cb):
    """A->B and C->B give A->C. Both links must have the same type."""
    if not (_is_binary_inference_link(ab) and _is_binary_inference_link(cb)):
        return None
    if ab.type != cb.type:
        return None
    A, B1 = ab.targets
    C, B2 = cb.targets
    if B1 != B2 or A == C or A == B1 or C == B1:
        return None
    stv = abduction(A.stv, B1.stv, C.stv, ab.stv, cb.stv)
    return Atom(ab.type, targets=(A, C), stv=stv)

def inversion_link(ab):
    """A->B gives B->A (same link type)."""
    if not _is_binary_inference_link(ab):
        return None
    A, B = ab.targets
    if A == B:
        return None
    stv = inversion(A.stv, B.stv, ab.stv)
    return Atom(ab.type, targets=(B, A), stv=stv)

def revise_atoms(a1, a2):
    """Revise two atoms that are the same statement. Returns a new Atom or None."""
    if a1.key() != a2.key():
        return None
    stv = revision(a1.stv, a2.stv)
    if a1.is_node:
        return Atom(a1.type, name=a1.name, stv=stv)
    return Atom(a1.type, targets=a1.targets, stv=stv)

def revise_into(space, atom):
    existing = space.get(atom.type, name=atom.name, targets=atom.targets)
    if existing is None:
        return space.add(atom)
    existing.stv = revision(existing.stv, atom.stv)
    return existing