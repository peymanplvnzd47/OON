import numpy as np


def lin2db(x):
    return 10*np.log10(x)


def db2lin(x):
    return 10**(x/10)



import math

def db_to_linear(db):
    """
    Convert decibels to linear scale.

    Parameters:
    - db (float): Decibel value.

    Returns:
    - linear (float): Corresponding linear value.
    """
    linear = 10 ** (db / 10)
    return linear

def linear_to_db(linear):
    """
    Convert linear scale to decibels.

    Parameters:
    - linear (float): Linear value.

    Returns:
    - db (float): Corresponding decibel value.
    """
    db = 10 * math.log10(linear)
    return db

# Example usage:
db_value = 20
linear_value = db_to_linear(db_value)
print(f"{db_value} dB is equal to {linear_value} in linear scale.")

linear_value_2 = 100
db_value_2 = linear_to_db(linear_value_2)
print(f"{linear_value_2} in linear scale is equal to {db_value_2} dB.")
