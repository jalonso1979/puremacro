"""Dynare macro-processor interface.

Provides pre-processing for Dynare ``.mod`` files supporting standard macro directives
including ``@#define``, ``@#undef``, ``@#for`` / ``@#endfor``, ``@#if`` / ``@#elseif`` /
``@#else`` / ``@#endif``, ``@#ifdef`` / ``@#ifndef``, ``@#include``, ``@#includepath``,
and inline ``@{...}`` interpolation.
"""
from __future__ import annotations

from puremacro.dsge._macro import (
    DynareMacroError,
    Scope,
    preprocess_macro,
)

__all__ = [
    "DynareMacroError",
    "Scope",
    "preprocess_macro",
]
