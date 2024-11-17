# Copyright (c) 2023 Jeff Tan, Carnegie Mellon University.
import inspect
import os
from contextlib import contextmanager
from functools import wraps
from sys import exc_info

import torch
from _testcapi import set_exc_info


class record_function(torch.profiler.record_function):
    """A context manager / function decorator that adds a label to a block of
    Python code (or function) when running autograd profiler. To avoid polluting
    the error messages, this class is invisible in the stack trace.

    Args:
        func_name (str): Name of the decorated function
    """

    def __call__(self, func, is_staticmethod=False, is_classmethod=False):
        if not inspect.isfunction(func):
            return func
        if hasattr(func, "_has_record_function_decorator"):
            # Avoid decorating the same function twice
            return func

        @wraps(func)
        def wrapped(*args, **kwargs):
            with self:
                try:
                    return func(*args, **kwargs)
                except:
                    # Remove the decorator itself from the stack trace (requires CPython)
                    # https://stackoverflow.com/questions/72146438/remove-decorator-from-stack-trace
                    tp, exc, tb = exc_info()
                    set_exc_info(tp, exc, tb.tb_next)
                    del tp, exc, tb
                    raise

        if is_staticmethod:
            # Without this, staticmethods will become non-static after wrapping,
            # so the `self` argument will get passed
            wrapped = staticmethod(wrapped)
        if is_classmethod:
            # Without this, classmethods will become non-class after wrapping,
            # so the `self` argument will get passed instead of `cls`
            wrapped = classmethod(wrapped)

        wrapped._has_record_function_decorator = True
        return wrapped


class record_class:
    """A class decorator that applies the @record_function decorator to every
    member function defined in the class

    Args:
        class_name (str): Name of the decorated class
    """

    def __init__(self, arg):
        assert isinstance(arg, str)
        self.class_name = arg

    def __call__(self, cls):
        if not inspect.isclass(cls):
            return cls

        # Find methods which are not defined in any parent class
        methods = set(inspect.getmembers(cls, predicate=inspect.isfunction))
        parent_methods = [
            inspect.getmembers(parent, predicate=inspect.isfunction)
            for parent in cls.__bases__
        ]
        unique_methods = methods.difference(*parent_methods)

        for name, func in unique_methods:
            func_name = f"{self.class_name}::{name}"
            is_staticmethod = isinstance(
                inspect.getattr_static(cls, name), staticmethod
            )
            is_classmethod = isinstance(inspect.getattr_static(cls, name), classmethod)
            setattr(
                cls,
                name,
                record_function(func_name)(func, is_staticmethod, is_classmethod),
            )

        return cls


def decorate_module(module):
    """Modifies a module in place to decorate every class with @record_class and
    every module with @record_function

    Args:
        module: Module to modify in-place
    Returns:
        module: The input module with classes and functions decorated
    """
    for name, func in inspect.getmembers(module, predicate=inspect.isfunction):
        if func.__module__ == module.__name__:
            setattr(module, name, record_function(name)(func))

    for name, cls in inspect.getmembers(module, predicate=inspect.isclass):
        if cls.__module__ == module.__name__:
            setattr(module, name, record_class(name)(cls))

    return module


@contextmanager
def torch_profile(save_dir, out_prefix, enabled=True):
    """Wrapper around torch.profiler.profile() that profiles CPU time, CUDA
    time, and memory usage. Writes output tables and Chrome traces to disk

    Args:
        save_dir (str): Directory to save output logs
        out_prefix (str): Prefix of output filenames
        enabled (bool): If False, this context manager does nothing
    """
    if not enabled:
        yield
        return

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        try:
            with record_function(out_prefix):
                yield
        except Exception as e:
            # Exit if the inner function raises an error
            import traceback

            print("".join(traceback.format_exception(None, e, e.__traceback__)))
            exit(1)

    prof_avgs = prof.key_averages(group_by_input_shape=True)

    os.makedirs(save_dir, exist_ok=True)
    for sort_by in [
        "cpu_time_total",
        "self_cpu_time_total",
        "cuda_time_total",
        "self_cuda_time_total",
        "cpu_memory_usage",
        "self_cpu_memory_usage",
        "cuda_memory_usage",
        "self_cuda_memory_usage",
    ]:
        with open(f"{save_dir}/{out_prefix}_{sort_by}.txt", "w") as f:
            f.write(prof_avgs.table(sort_by=sort_by))

    prof.export_chrome_trace(f"{save_dir}/{out_prefix}_trace.json")
