from __future__ import annotations

import pytest
from code_prototypes import (
    pulser_basic_mis_pulse_v1,
    pulser_basic_ramp_v2,
    pulser_basic_rx_v1,
    pyq_basic_diff_v1,
    pyq_basic_rx_v1,
    pyq_basic_training_v1,
    pyq_basic_training_v2,
)


def test_pyq_basic_diff_v1() -> None:
    pyq_basic_diff_v1()


def test_pyq_basic_training_v1() -> None:
    pyq_basic_training_v1()


def test_pyq_basic_training_v2() -> None:
    pyq_basic_training_v2()


@pytest.mark.xfail
def test_pyq_basic_rx_v1() -> None:
    """
    !!!note
        This code fails because Qadence 2 IR currently does not handle arbitrary
        Hamiltonians.
    """

    pyq_basic_rx_v1()


def test_pulser_basic_rx_v1() -> None:
    pulser_basic_rx_v1()


def test_pulser_basic_ramp_v2() -> None:
    pulser_basic_ramp_v2()


def test_pulser_basic_mis_pulse_v1() -> None:
    pulser_basic_mis_pulse_v1()
