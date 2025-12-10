#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 10:49:10 2025

@author: azelcer
"""
from __future__ import annotations
import pathlib as _pl
import datetime as _datetime


class FileNameServer:
    """Dummy class that delivers base filenames and dirs to modules.


    Es sólo un bosquejo.
    """

    _base_name: _pl.Path | None = None
    _base_dir: _pl.Path | None = None

    def __init__(
            self,
            base_dir: _pl.Path | str = None,
            base_name: _pl.Path | str = None
            ):
        self.set_base_dir(base_dir or self.get_home_path())
        if base_name:
            self.set_base_name(base_name)

    def set_base_name(self, base_name: _pl.Path | str):
        base = _pl.Path(base_name)
        if base.parent != _pl.Path("."):
            raise ValueError("El nombre base no debe tener directorios")
        self._base_name = base

    def get_base_name(self) -> _pl.Path:
        return _pl.Path(self._base_name)

    def set_base_dir(self, base_dir: _pl.Path | str):
        base = _pl.Path(base_dir)
        self._base_dir = base

    def get_base_dir(self) -> _pl.Path:
        return _pl.Path(self._base_dir)

    @classmethod
    def get_home_path(*args) -> _pl.Path:
        """Convenience function."""
        return _pl.Path.home()

    @classmethod
    def get_datetime_str(*args, time: bool = True) -> str:
        """Convenience function."""
        if time:
            return _datetime.datetime.now().isoformat(timespec="seconds").replace(":", "")
        else:
            return _datetime.date.today().isoformat().replace(":", "")
