"""Set and verify PUCQ4 Q6 flux bias using Yoko 4.

Displayed Q6 is Python index 5. Yoko 4 is the dedicated Q6 DC source.
Currents are always expressed in amperes; 1 mA is 1e-3 A.
"""

from malab_yokogawa_gs200 import YokogawaGS200


YOKO3_IP = "192.168.1.73"
YOKO4_IP = "192.168.1.77"
TARGET_CURRENT = 0.0
MAX_ABS_CURRENT = 0.015
VERIFY_TOLERANCE = 1e-7


def set_flux_current(ip_address: str, label: str,
                     target_current: float = TARGET_CURRENT) -> float:
    if abs(target_current) > MAX_ABS_CURRENT:
        raise ValueError("Q6 bias exceeds the configured 15 mA safety limit")

    with YokogawaGS200(ip_address, max_current=MAX_ABS_CURRENT) as yoko:
        print(f"{label} instrument:", yoko.get_id())
        print("Mode before:", yoko.get_mode())
        print("Output before:", yoko.get_output())
        print("Current before: %.9f A" % yoko.get_current())
        print("Range: %.9f A" % yoko.get_range())
        print("Voltage limit: %.6f V" % yoko.get_voltage_limit())

        # Lab procedure: 10 mA range, 0.5 mA/s ramp, then output off at zero.
        yoko.set_range(0.01)
        yoko.ramp_current(target_current, sweeprate=0.0005)
        yoko.set_output(False)
        measured = yoko.get_current()

        print("Output after:", yoko.get_output())
        print("Current after: %.9f A" % measured)
        if abs(measured - target_current) > VERIFY_TOLERANCE:
            raise RuntimeError(
                "Yoko 4 verification failed: requested %.9f A, read %.9f A"
                % (target_current, measured)
            )
        return measured


def set_q6_current(target_current: float = TARGET_CURRENT) -> float:
    return set_flux_current(YOKO4_IP, "Yoko 4 / Q6", target_current)


def set_all_known_pucq4_flux_lines_zero():
    """Set the two documented PUCQ4 DC lines to zero and disable outputs."""
    return {
        "Q4": set_flux_current(YOKO3_IP, "Yoko 3 / Q4"),
        "Q6": set_flux_current(YOKO4_IP, "Yoko 4 / Q6"),
    }


if __name__ == "__main__":
    set_all_known_pucq4_flux_lines_zero()
