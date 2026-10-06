import json
import os
import tempfile

NODE_TYPES = {"ConceptNode", "PredicateNode"}
LINK_TYPES = {
    "InheritanceLink",
    "SubsetLink",
    "SimilarityLink",
    "EvaluationLink",
    "ImplicationLink",
}


class STV:
    """Simple Truth Value: strength and confidence, both in [0, 1]."""

    __slots__ = ("strength", "confidence")

    def __init__(self, strength, confidence):
        strength = float(strength)
        confidence = float(confidence)
        if not 0.0 <= strength <= 1.0:
            raise ValueError(f"strength must be in [0, 1], got {strength}")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {confidence}")
        self.strength = strength
        self.confidence = confidence

    def __eq__(self, other):
        return (
            isinstance(other, STV)
            and self.strength == other.strength
            and self.confidence == other.confidence
        )

    def __hash__(self):
        return hash((self.strength, self.confidence))

    def __repr__(self):
        return f"(STV {self.strength:.4f} {self.confidence:.4f})"


class Atom:
    """A node (type + name) or a link (type + ordered targets).

    Identity is (type, name, targets). The STV is NOT part of identity.
    """

    def __init__(self, type, name=None, targets=None, stv=None):
        if (name is None) == (targets is None):
            raise ValueError("an atom needs either a name (node) or targets (link)")
        if name is not None and type not in NODE_TYPES:
            raise ValueError(f"unknown node type: {type}")
        if targets is not None and type not in LINK_TYPES:
            raise ValueError(f"unknown link type: {type}")
        if targets is not None:
            targets = tuple(targets)
            if len(targets) == 0:
                raise ValueError("a link needs at least one target")
            if not all(isinstance(t, Atom) for t in targets):
                raise ValueError("link targets must be Atom objects")
        self.type = type
        self.name = name
        self.targets = targets
        self.stv = stv if stv is not None else STV(1.0, 0.0)

    @property
    def is_node(self):
        return self.name is not None

    @property
    def is_link(self):
        return self.targets is not None

    def key(self):
        if self.is_node:
            return (self.type, self.name)
        return (self.type, tuple(t.key() for t in self.targets))

    def __eq__(self, other):
        return isinstance(other, Atom) and self.key() == other.key()

    def __hash__(self):
        return hash(self.key())

    def __repr__(self):
        if self.is_node:
            return f"({self.type} {self.name})"
        inner = " ".join(repr(t) for t in self.targets)
        return f"({self.type} {inner})"


def Concept(name, stv=None):
    return Atom("ConceptNode", name=name, stv=stv)


def Predicate(name, stv=None):
    return Atom("PredicateNode", name=name, stv=stv)


def Link(type, *targets, stv=None):
    return Atom(type, targets=targets, stv=stv)


class AtomSpace:
    """Stores atoms, each exactly once, with indexes for fast lookup."""

    def __init__(self):
        self.atoms = {}          # key -> Atom
        self.by_type = {}        # type -> set of Atom
        self._first = {}         # Atom -> set of links whose first target is it
        self._last = {}          # Atom -> set of links whose last target is it
        self._containing = {}    # Atom -> set of links that contain it anywhere


    def add(self, atom):
        """Add an atom, or update the STV if it already exists.

        Returns the stored atom. Link targets are replaced by the stored
        atoms, and missing targets are added too.
        """
        existing = self.atoms.get(atom.key())
        if existing is not None:
            existing.stv = atom.stv
            return existing

        if atom.is_link:
            stored_targets = tuple(self._ensure(t) for t in atom.targets)
            stored = Atom(atom.type, targets=stored_targets, stv=atom.stv)
        else:
            stored = Atom(atom.type, name=atom.name, stv=atom.stv)

        self.atoms[stored.key()] = stored
        self.by_type.setdefault(stored.type, set()).add(stored)
        if stored.is_link:
            self._first.setdefault(stored.targets[0], set()).add(stored)
            self._last.setdefault(stored.targets[-1], set()).add(stored)
            for t in set(stored.targets):
                self._containing.setdefault(t, set()).add(stored)
        return stored

    def _ensure(self, atom):
        """Return the stored version of atom, adding it if missing."""
        found = self.atoms.get(atom.key())
        if found is not None:
            return found
        return self.add(atom)

    def get(self, type, name=None, targets=None):
        """Return the stored atom, or None."""
        if name is not None:
            return self.atoms.get((type, name))
        if targets is not None:
            return self.atoms.get((type, tuple(t.key() for t in targets)))
        return None

    def remove(self, atom):
        """Remove an atom and every link that contains it. Returns count removed."""
        stored = self.atoms.get(atom.key())
        if stored is None:
            return 0
        removed = 0
        for link in list(self._containing.get(stored, ())):
            removed += self.remove(link)
        if stored.key() not in self.atoms:
            return removed
        del self.atoms[stored.key()]
        self.by_type[stored.type].discard(stored)
        if stored.is_link:
            self._first[stored.targets[0]].discard(stored)
            self._last[stored.targets[-1]].discard(stored)
            for t in set(stored.targets):
                self._containing[t].discard(stored)
        self._containing.pop(stored, None)
        self._first.pop(stored, None)
        self._last.pop(stored, None)
        return removed + 1

    def all(self):
        return list(self.atoms.values())

    def of_type(self, type):
        return list(self.by_type.get(type, ()))

    def __len__(self):
        return len(self.atoms)

    def __contains__(self, atom):
        return atom.key() in self.atoms

    # ---- queries ---------------------------------------------------------

    def links_from(self, atom, type=None):
        """Links whose first target is atom."""
        stored = self.atoms.get(atom.key())
        if stored is None:
            return []
        links = self._first.get(stored, ())
        return [l for l in links if type is None or l.type == type]

    def links_to(self, atom, type=None):
        """Links whose last target is atom."""
        stored = self.atoms.get(atom.key())
        if stored is None:
            return []
        links = self._last.get(stored, ())
        return [l for l in links if type is None or l.type == type]

    def match(self, type=None, source=None, target=None):
        """Pattern query. None means any. source/target apply to links."""
        if source is not None:
            candidates = self.links_from(source)
        elif target is not None:
            candidates = self.links_to(target)
        elif type is not None:
            candidates = self.of_type(type)
        else:
            candidates = self.all()
        result = []
        for a in candidates:
            if type is not None and a.type != type:
                continue
            if source is not None and not (a.is_link and a.targets[0] == source):
                continue
            if target is not None and not (a.is_link and a.targets[-1] == target):
                continue
            result.append(a)
        return result

    @staticmethod
    def _ref(atom):
        if atom.is_node:
            return {"type": atom.type, "name": atom.name}
        return {"type": atom.type, "targets": [AtomSpace._ref(t) for t in atom.targets]}

    @staticmethod
    def _from_ref(ref, space):
        if "name" in ref:
            return space._ensure(Atom(ref["type"], name=ref["name"]))
        targets = [AtomSpace._from_ref(t, space) for t in ref["targets"]]
        return space._ensure(Atom(ref["type"], targets=targets))

    def save(self, path):
        nodes = [a for a in self.atoms.values() if a.is_node]
        links = [a for a in self.atoms.values() if a.is_link]
        data = {
            "atoms": [
                {
                    **self._ref(a),
                    "stv": [a.stv.strength, a.stv.confidence],
                }
                for a in nodes + links
            ]
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        space = cls()
        for entry in data["atoms"]:
            atom = cls._from_ref(entry, space)
            atom.stv = STV(*entry["stv"])
        return space