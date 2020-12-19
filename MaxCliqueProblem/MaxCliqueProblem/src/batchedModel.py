from docplex.mp.model import Model

class BatchedModel(Model):
    """Modified docplex.mp.model.Model
    which can bath constraints and apply it
    only before solution computing"""

    def __init__(self, name=None):
        """Remember constraints and
        apply in in the model only ones
        """
        super().__init__(name=name)
        self._bath_add_constr = []
        self._bath_remove_constr = []

    def add_constraint_bath(self, constr):
        """Add constraints to bath"""
        self._bath_add_constr.append(constr)
        return constr

    def remove_constraint_bath(self, constr):
        """Remove constraints from bath"""
        self._bath_remove_constr.append(constr)

    def apply_batch(self):
        """Apply all batched constraints"""
        if self._bath_add_constr:
            super().add_constraints(self._bath_add_constr)
            self._bath_add_constr = []

        if self._bath_remove_constr:
            super().remove_constraints(self._bath_remove_constr)
            self._bath_remove_constr = []

    def solve(self):
        """Solve model adding all bathed
        constraints before computing"""
        self.apply_batch()
        return super().solve()


