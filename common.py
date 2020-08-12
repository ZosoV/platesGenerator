"""
Definitions that don't fit elsewhere.

"""

__all__ = (
    'DIGITS',
    'LETTERS',
    'CHARS',
    'sigmoid',
    'softmax',
)

import numpy

actual_format = 'BCDFGHJKLPRSTVWXYZ'
past_format_letter1 = 'ABCDEFGHKLNPRSTUVXYZWM'
past_format_letter2 = 'ABCDEFGHIJKLNPRSTUVXYZ'

DIGITS = "0123456789"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHARS = LETTERS + DIGITS

