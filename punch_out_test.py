import sys
import os
import numpy as np
# sys.path.append(os.path.abspath("/home/qubituser/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux/")) # for QUIET
#sys.path.append(os.path.abspath("/home/nexusadmin/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux_nexus/")) # for NEXUS
from system_config import QICK_experiment
from section_003_punch_out_ge_mux import PunchOut
import datetime

number_of_qubits = 6  #currently 4 for NEXUS, 6 for QUIET

# sweep_DAC_attenuator1 =[] #np.linspace(5,20, 4)
# sweep_DAC_attenuator2 =[10]#[15,20,25,30] #np.linspace(5,20,4)

substudy = 'punchout'
outerFolder = os.path.join(f"M:/_Data/20250822 - Olivia/bob_run_started_Feb_11/squill/{substudy}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/")
outerfolder_plots = outerFolder + "/documentation/"

DAC_att_1=10
DAC_att_2=15
DAC_att=DAC_att_1+DAC_att_2
ADC_att=17
from expt_config import FRIDGE
experiment = QICK_experiment(outerfolder_plots, DAC_attenuator1 = DAC_att_1, DAC_attenuator2 = DAC_att_2, qubit_DAC_attenuator1 = 5 , qubit_DAC_attenuator2 = 4 ,ADC_attenuator = ADC_att, fridge=FRIDGE)
Qubit_index= 0 #starts at 0
Unmask = True
punch_out   = PunchOut(Qubit_index, number_of_qubits, outerfolder_plots, experiment, Unmask)

start_gain, stop_gain, num_points =  0.001, 1, 30 # for QUIET 0.55, 0.775, 5 #
#start_gain, stop_gain, num_points = 0.0, 0.8, 10 # for NEXUS

punch_out.run(experiment.soccfg, experiment.soc, start_gain, stop_gain, num_points, DAC_att, ADC_att, plot_Center_shift = True, plot_res_sweeps = True)

del punch_out
del experiment
