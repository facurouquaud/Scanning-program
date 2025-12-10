#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generic utils for PyQt interfaces


Created on Tue Jan  7 11:16:32 2025

@author: azelcer
"""
from PyQt5.QtWidgets import (QLineEdit as _QLineEdit, QCheckBox as _QCheckBox)
from PyQt5 import QtGui as _QtGui


def create_float_lineedit(data: str = "", max_val: float = 200.) -> _QLineEdit:
    """
    Create and fill.

    Parameters
    ----------
    data : str
        Initial text to display in line edit. Default is empty string.

    """
    le = _QLineEdit(str(float(data)))
    le.setValidator(_QtGui.QDoubleValidator(0, max_val, 3))
    return le


def create_int_lineedit(data: str = "") -> _QLineEdit:
    """
    Create and fill.

    Parameters
    ----------
    data : str
        Initial text to display in line edit. Default is empty string.

    """
    le = _QLineEdit(str(int(data)))
    le.setValidator(_QtGui.QIntValidator(3, 2000))
    return le


class GroupedCheckBoxes:
    """Manages grouped CheckBoxes states.

    This is a helper class to ease GUI implementation. It implements an 'All'
    checkbox that controls and stays sinchronized with others. It should be used
    carefully to avoid state changes loops.
    """

    def __init__(self, all_checkbox: _QCheckBox, *other_checkboxes):
        """Init class.

        Parameters
        ----------
        all_checkbox : QCheckBox
            The checkbox that checks/unchecks all others.
        *other_checkboxes : Iterable[QCheckBox]
            All other checkboxes.
        """
        self.acb = all_checkbox
        self.others = other_checkboxes
        for _ in other_checkboxes:
            _.stateChanged.connect(self.on_state)
        all_checkbox.clicked.connect(self.on_click)

    def on_state(self, state: int):
        """Handle single items change."""
        self.acb.setChecked(all([_.isChecked() for _ in self.others]))

    def on_click(self, is_checked: bool):
        """Handle 'All' checkbox click."""
        for _ in self.others:
            _.setChecked(is_checked)
