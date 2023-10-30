"""Stores a dict of all kernel names mapping each to a
class. Should be updated any time a new kernel is
added / any time one is removed."""
from .exact_quadratic import ExactQuadratic


KERNEL_NAME_TO_CLASS = {"ExactQuadratic":ExactQuadratic}
