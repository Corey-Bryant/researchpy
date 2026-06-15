# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
import pandas as pd




@dataclass
class CoreDataclass:
    """Generic base dataclass providing common utility methods.

    All methods reference only ``self`` and dynamically inspect fields,
    so subclasses inherit them without needing to override.
    """

    def __post_init__(self):
        self.__name__ = "Researchpy.CoreDataclass"


    def to_dict(self, drop_none=True):
        dct = {}
        for f in fields(self):
            if drop_none and getattr(self, f.name) is not None:
                dct[f.name] = getattr(self, f.name)

        return dct


    def _get_summary(self, skip_raw_arrays=False):
        """Print a human-readable summary of all DataFrame/dict fields."""
        printed = []

        for f in fields(self):
            val = getattr(self, f.name)

            if val is None:
                continue

            if isinstance(val, pd.DataFrame):
                printed.append(val.to_string(index=False))

            elif isinstance(val, (list, np.ndarray)):
                if not skip_raw_arrays:
                    try:
                        val = pd.Series(val)
                        printed.append(val.to_string(index=False))

                    except:
                        try:
                            for x in val: printed.append(f"{x}")
                        except:
                            print(f"{f.name} could not be printed as a DataFrame or Series, and is not a simple list/array. Skipping.")
                            continue
                else:
                    continue # skip raw arrays

            else:
                printed.append(f"{f.name}: {val}")

        print("\n".join(printed))


    def info(self):
        class_name = type(self).__name__
        parts = [f"Class({class_name})"]

        for f in fields(self):
            val = getattr(self, f.name)

            if val is None:
                parts.append(f"  {f.name}=None")

            elif isinstance(val, pd.DataFrame):
                parts.append(f"  {f.name}=pd.DataFrame({val.shape[0]}x{val.shape[1]})")

            elif isinstance(val, dict):
                parts.append(f"  {f.name}=dict({len(val)} keys)")

            elif isinstance(val, (list, np.ndarray)):
                if isinstance(val, list):
                    length = len(val)
                    parts.append(f"  {f.name}={type(val).__name__}(len={length})")
                else:
                    parts.append(f"  {f.name}=np.ndarray(shape={val.shape})")

            else:
                parts.append(f"  {f.name}={val!r}")

        return "\n\n".join(parts) + "\n)"


    def __iter__(self):
        """Yield fields in order for tuple-style unpacking."""
        for f in fields(self):
            yield getattr(self, f.name)


    def __repr__(self):
        return self.info()

