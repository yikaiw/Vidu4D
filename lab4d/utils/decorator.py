# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
from functools import wraps


def train_only_fields(method):
    """Decorator to skip the method and return an empty field list if not in
    training mode.
    """

    @wraps(method)
    def _impl(self, *method_args, **method_kwargs):
        if self.training:
            return method(self, *method_args, **method_kwargs)
        else:
            return {}

    return _impl
