import sys
import os
import numpy as np
# sys.path.append(os.path.abspath("/home/qubituser/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux/")) # for QUIET
#sys.path.append(os.path.abspath("/home/nexusadmin/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux_nexus/")) # for NEXUS
sys.path.append(os.path.abspath("/home/kanyang/Github/4x2Loud_tprocV2"))
from system_config import QICK_experiment
from section_003_punch_out_ge_mux import PunchOut
import datetime

number_of_qubits = 6  #currently 4 for NEXUS, 6 for QUIET

# sweep_DAC_attenuator1 =[] #np.linspace(5,20, 4)
# sweep_DAC_attenuator2 =[10]#[15,20,25,30] #np.linspace(5,20,4)

################################################ Data Saving Setup ##################################################
# Folders

data_set = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

run_name = 'run5'
device_name = 'rfsoc-4x2-loopback'  # 'saph-6transmon'#  'sil-6transmon'
substudy_txt_notes = ('Test-Loopback')# ('This data was taken after reverting back to only 1 channel on the qick box. T1 shots saved as well as averaged IQ data.\n') # Initial qubit checkouts quiet run 8

study = 'tests-round_robin' #qubit_checkouts
sub_study ='tests'# 'source_on_25dBDAC' #pre_AB_paper_data_still_optimizing, two_photon_peak_search, AB_Paper_Data_24hrs, ABpaperdata3rdbatch_21dB_DACatten_Q1to5_t1shots_optional
#ABpaperdata3rdbatch_21dB_DACatten_Q1to6_t1shots_optional, ABpaperdata_21dB_DACatten_Q1to6_t1shots_optional_newopt, 18dB_DAC_testdata_allQs_exceptQ4, cooldown_run8b_19dB_DAC_allQs
data_set = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

if not os.path.exists(f"/data/QICK_data/{run_name}/"):
    os.makedirs(f"/data/QICK_data/{run_name}/")
if not os.path.exists(f"/data/QICK_data/{run_name}/{device_name}/"):
    os.makedirs(f"/data/QICK_data/{run_name}/{device_name}/")
studyFolder = os.path.join(f"/data/QICK_data/{run_name}/{device_name}/", study)
if not os.path.exists(studyFolder):
    os.makedirs(studyFolder)
subStudyFolder = os.path.join(studyFolder, sub_study)
if not os.path.exists(subStudyFolder):
    os.makedirs(subStudyFolder)

outerFolder = os.path.join(subStudyFolder, data_set)
optimizationFolder = os.path.join(outerFolder, 'optimization')
studyFolder = os.path.join(outerFolder, 'study_data')
outerfolder_plots= os.path.join(outerFolder, 'documentation')
subStudyDataFolder = os.path.join(outerFolder, 'study_data')
studyDocumentationFolder = os.path.join(outerFolder, 'documentation')
                                  
if not os.path.exists(studyDocumentationFolder):
    os.makedirs(studyDocumentationFolder)
if not os.path.exists(optimizationFolder):
    os.makedirs(optimizationFolder)
if not os.path.exists(subStudyDataFolder):
    os.makedirs(subStudyDataFolder)

file_path = os.path.join(outerfolder_plots, 'sub_study_notes.txt')
with open(file_path, "w", encoding="utf-8") as file:
    file.write(substudy_txt_notes)


#substudy = 'punchout_v5'
#outerFolder = os.path.join(f"M:/_Data/20250822 - Olivia/bob_run_started_Aug_23/squill/{substudy}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/")
#outerfolder_plots = outerFolder + "/documentation/"
#outerFolder = os.path.join("/home/nexusadmin/qick/NEXUS_sandbox/Data/Run30/", str(datetime.date.today())) # for NEXUS
# for att_1 in sweep_DAC_attenuator1:
#     for att_2 in sweep_DAC_attenuator2:
#         att_1 = round(att_1, 3)
#         att_2 = round(att_2, 3)
#         experiment = QICK_experiment(outerFolder, DAC_attenuator1 = att_1, DAC_attenuator2 = att_2)
#         punch_out   = PunchOut(outerFolder, experiment)
#
#         start_gain, stop_gain, num_points = 0.1, 1, 10
#         punch_out.run(experiment.soccfg, experiment.soc, start_gain, stop_gain, num_points, att_1, att_2, plot_Center_shift = True, plot_res_sweeps = True)
#
#         del punch_out
#         del experiment
#

DAC_att_1=10
DAC_att_2=15
DAC_att=DAC_att_1+DAC_att_2
ADC_att=17
from expt_config import FRIDGE
experiment = QICK_experiment(outerfolder_plots, DAC_attenuator1 = DAC_att_1, DAC_attenuator2 = DAC_att_2, qubit_DAC_attenuator1 = 5 , qubit_DAC_attenuator2 = 4 ,ADC_attenuator = ADC_att, fridge=FRIDGE)
Qubit_index= 3 #starts at 0
Unmask = True
punch_out   = PunchOut(Qubit_index, number_of_qubits, outerfolder_plots, experiment, Unmask)

start_gain, stop_gain, num_points =  0.0005, 0.005, 5 # for QUIET 0.55, 0.775, 5 #
#start_gain, stop_gain, num_points = 0.0, 0.8, 10 # for NEXUS

punch_out.run(experiment.soccfg, experiment.soc, start_gain, stop_gain, num_points, DAC_att, ADC_att, plot_Center_shift = True, plot_res_sweeps = True)

del punch_out
del experiment
