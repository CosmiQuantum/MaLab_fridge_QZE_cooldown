"""
Yokogawa GS200 DC source driver -- flux bias for PUCQ4.

Adapted from HouckLab_QICK:
  WorkingProjects/Inductive_Coupler/Client_modules/PythonDrivers/YOKOGS200.py

The original is a good starting point but has four bugs that matter when the
thing on the other end is a dilution fridge. Each fix is marked FIX below.
Diff the two if you want to see exactly what moved.
"""

import sys
import time

import numpy as np
import pyvisa as visa

# FIX 1: the original starts with `import visa` (PyVISA < 1.5) followed by
# `import pyvisa as visa`. The first line raises ModuleNotFoundError on any
# modern install, so the file cannot be imported at all. Dropped it.


class YOKOGS200:
    """Yokogawa GS200 in current-source mode.

    Typical use:

        import pyvisa
        rm = pyvisa.ResourceManager()
        yoko = YOKOGS200('GPIB0::2::INSTR', rm)
        yoko.SetMode('current')
        yoko.SetCurrent(4e-3)      # amps -- 4 mA
        ...
        yoko.RampToZero()
        yoko.OutputOff()
    """

    def __init__(self, VISAaddress, rm, maxCurrent=0.015, maxVoltage=1.3):
        self.VISAaddress = VISAaddress
        self.maxVoltage = maxVoltage

        # FIX 2: the original has no current limit at all -- only SetVoltage
        # checks a maximum. A typo of 100 for 0.100 would push the GS200's full
        # 200 mA down a flux line. The student's scans only ever needed
        # +/-10 mA, so the default ceiling here is 15 mA. Raise it deliberately
        # if you actually need to, do not edit it away.
        self.maxCurrent = maxCurrent

        self.rampstep = 0.001       # [A or V] increment when ramping
        self.rampinterval = 0.01    # [s] dwell per step. Do NOT go below 0.001
                                    # or the fridge heats up.
        try:
            self.session = rm.open_resource(VISAaddress)
        except visa.Error:
            sys.stderr.write("Couldn't connect to '%s', exiting now..."
                             % VISAaddress)
            sys.exit()

    # ======================================================================= #

    def OutputOn(self):
        self.session.write('OUTPut 1')

    def OutputOff(self):
        self.session.write('OUTPut 0')

    # ======================================================================= #

    def _ramp(self, start, stop, write_level):
        """Shared ramp: always dwells, always lands exactly on the target."""
        # FIX 3a: the original computes steps as max(1, ...) in SetCurrent. With
        # num=1, np.linspace(start, stop, 1) returns [start] -- so any current
        # change smaller than ~0.5*rampstep silently did nothing and left the
        # bias at its old value. max(2, ...) guarantees the endpoint is hit.
        steps = max(2, int(round(abs(stop - start) / self.rampstep)) + 1)
        for level in np.linspace(start, stop, num=steps):
            write_level(level)
            # FIX 3b: SetCurrent in the original has no sleep in its loop, while
            # SetVoltage does -- despite the comment on rampinterval warning
            # that going too fast heats the fridge. Current mode is exactly the
            # case where that matters.
            time.sleep(self.rampinterval)

    def SetVoltage(self, voltage):
        """Ramp to `voltage` volts."""
        if np.abs(voltage) > self.maxVoltage:
            print('!!!! voltage greater than maximum voltage !!!! '
                  'Reduce or change max.')
            return
        start = self.GetVoltage()
        self.OutputOn()
        self._ramp(start, voltage,
                   lambda v: self.session.write(':SOURce:LEVel:AUTO %.8f' % v))

    def SetCurrent(self, current):
        """Ramp to `current` amps. Note the units: 4 mA is 4e-3, not 4."""
        if np.abs(current) > self.maxCurrent:
            print('!!!! current %.4f A exceeds maxCurrent %.4f A !!!! '
                  'Reduce, or raise maxCurrent deliberately.'
                  % (current, self.maxCurrent))
            return
        start = self.GetCurrent()
        self.OutputOn()
        self._ramp(start, current,
                   lambda i: self.session.write(':SOURce:LEVel:AUTO %.8f' % i))

    def RampToZero(self):
        """Walk the bias back to zero. Call before shutting down."""
        if self.GetMode() == 'current':
            self.SetCurrent(0.0)
        else:
            self.SetVoltage(0.0)

    def SetMode(self, mode):
        if mode not in ('voltage', 'current'):
            sys.stderr.write("Unknown output mode %s." % mode)
            return
        self.session.write('SOURce:FUNCtion %s' % mode)

    # ======================================================================= #

    # FIX 4: the original getters write `SOURce:FUNCtion VOLTage` / `CURRent`
    # before reading, so calling GetCurrent() on an instrument in voltage mode
    # silently switched its mode -- a getter with a side effect on a live flux
    # line. These query the mode instead of imposing it.

    def GetMode(self):
        result = self.session.query('SOURce:FUNCtion?').strip()
        return 'voltage' if result.startswith('VOLT') else 'current'

    def _GetLevel(self):
        return float(self.session.query('SOURce:LEVel?').strip())

    def GetVoltage(self):
        """Volts. Returns 0.0 if the instrument is in current mode."""
        if self.GetMode() != 'voltage':
            return 0.0
        return self._GetLevel()

    def GetCurrent(self):
        """Amps. Returns 0.0 if the instrument is in voltage mode."""
        if self.GetMode() != 'current':
            return 0.0
        return self._GetLevel()
