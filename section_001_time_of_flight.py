from qick.asm_v2 import AveragerProgramV2
import matplotlib.pyplot as plt
from build_state import *
from expt_config import *
from system_config import *

class TOFExperiment:
    def __init__(self, QubitIndex,  outerFolder, experiment, round_num = 1, save_figs = True, title = False, qick_verbose=True, unmasking_resgain = False):
        # every time a class instance is created, these definitions are set
        self.expt_name = "tof"
        self.QubitIndex = QubitIndex
        self.outerFolder = outerFolder
        self.Qubit = 'Q' + str(QubitIndex)
        self.exp_cfg = expt_cfg[self.expt_name]
        self.experiment = experiment
        self.save_figs = save_figs
        self.title = title
        self.qick_verbose=qick_verbose
        self.experiment.soc.reset_gens()
        if unmasking_resgain:
            self.exp_cfg["list_of_all_qubits"] = [QubitIndex]

        self.q_config = all_qubit_state(self.experiment,6)
        self.round_num = round_num
        if 'All' in self.Qubit:
            self.config = {**self.q_config['Q0'], **self.exp_cfg}
            print(f'Q {self.QubitIndex} Round {round_num} TOF configuration: ', self.config)
        else:
            self.config = {**self.q_config[self.Qubit], **self.exp_cfg}
            print(f'Q {self.QubitIndex + 1} Round {round_num} TOF configuration: ',self.config)


    def run(self):
        class MuxProgram(AveragerProgramV2):
            def _initialize(self, cfg):
                ro_chs = cfg['ro_ch']
                res_ch = cfg['res_ch']
                res_ch2 = cfg['res_ch']
                qubit_ch = cfg['qubit_ch']
                self.declare_gen(ch=res_ch, nqz=cfg['nqz_res'])
                self.declare_readout(ch=cfg['ro_ch'], length= 10)

                self.add_readoutconfig(ch=ro_chs, name="myro",
                                       freq=  cfg['res_freq_ge'],#cfg['res_freq_ge'], #cfg['res_freq_ge'],
                                       gen_ch=res_ch,
                                       outsel='product')

                self.send_readoutconfig(ch=cfg['ro_ch'], name="myro", t=0.0)
                # print(cfg["res_length"],cfg['ro_phase'],cfg['res_gain_ge'])
                self.add_pulse(ch=res_ch, name="res_pulse", ro_ch=ro_chs,
                               style="const",
                               length= cfg["res_length"],
                               freq= cfg['res_freq_ge'],#cfg['res_freq_ge'],#cfg['res_freq_ge'],
                               phase=cfg['ro_phase'],
                               gain= cfg['res_gain_ge']+0.3
                               )


                self.add_pulse(ch=res_ch2, name="res_pulse2", ro_ch=ro_chs,
                               style="const",
                               length= cfg["res_length"],
                               freq= cfg['res_freq_ge'],#cfg['res_freq_ge'],#cfg['res_freq_ge'],           
                               phase=cfg['ro_phase']+180,
                               gain= cfg['res_gain_ge']+0.3
                               )

                
                ## This is my attempt at a Two Stepped Pulse
                ##
                ##    A Few Notes
                ##
                ##  Because we delare a custom envelope it is safest to reset_gen before and after reach run,
                ##  this ensures we do not accidentally overlaod the board or at least it gives us an escape if we do
                ##
                ##  For a similar reason we add a delay at the end of the pulse/envelope delarations. This ensures the
                ##  FPGA has time to set properly.
                ##
                ##  Skipping these two lines is a dangerous game and has resulted in a board crash requiring manual reset
                ##  Unfortunately as the board lives in LOUD (which is far away) this is not optimal
                ##
                ##  Finally the current setting of the envelope power is sloppy given the board constraints. This can be
                ##  Handeled more rigerously however given the parameters of LOUD's Silicon Chip this is sufficient for now
                ##  If you have question on this ask me -Daniel Molenaar
                ##
                ###===================================================================================================                  
                #StartTime = 0.4
                v_smag = 16383  ## Max Power for arb signal                                                    \
                sigma2 = 0.02  ## Sam Suggest 0.01, TOF plots suggest 0.02 looks a lot smoother         
                alpha = 0.05  ### Normal Gain Units ## With this setup alpha + beta cannot be more than 0.5
                beta = 0.08  ### Normal Gain units  ## sneaking around this is tough so we will ignore 
                l_two = 0.01    ##Length of initial peak                                           
                l_base = 0.9     ##Dominated Readout length                                          
                stepsmall = 10000  ##This is probably over kill but it looks nice                              
                ti = 4*sigma2 #0 + StartTime                                                                              
                deslen = l_base+ti+ (4*sigma2)
                alpha =alpha*4
                beta = beta * 4
                fs_gen = self.soccfg['gens'][res_ch]['fs']
                stepscorr = (((int(fs_gen * deslen)) + 15) // 16) * 16
                t_new = np.linspace(0, deslen, stepscorr)
                # Defining the piecewise flat-top Gaussian function f(t, sigma, ti, L)                        
                def f_flat(t_new, sigma2, ti, L):

                    # Gaussian Rise                                                                          
                    rise = np.exp(-((t_new - ti)**2) / (2 * sigma2**2))
                    # Flat Top                                                                              

                    top = np.ones_like(t_new)
                    # Gaussian Fall                                                              

                    fall = np.exp(-((t_new - (ti + L))**2) / (2 * sigma2**2))
                    # Use np.select to apply conditions element-wise                                       
                    return np.select(
                        [t_new < ti, (t_new >= ti) & (t_new < ti + L), t_new >= ti + L],
                        [rise,top,fall]
                    )
                pulse_shape = v_smag  * (beta * f_flat(t_new, sigma2, ti, l_base) + alpha * f_flat(t_new, sigma2, ti, l_two))
                #pulse = v_smag * np.sin(omega*t) * (f_flat(t, sigma, ti, l_base) + alpha * f_flat(t, sigma, ti, l_two))         \
                    
                self.add_envelope(ch=res_ch, name="TwoStep",idata = pulse_shape)
                self.add_pulse(ch=res_ch, name="two_step", ro_ch=ro_chs,
                               style="arb",
                               envelope='TwoStep',
                               freq= cfg['res_freq_ge'],# cfg['res_freq_ge'],#cfg['res_freq_ge'],   
                               phase=cfg['ro_phase']-20,
                               gain= 1.0#cfg['res_gain_ge']+0.6                                                                   
                               )
                
                self.declare_gen(ch=res_ch, nqz=cfg['nqz_res'])
                #self.declare_readout(ch=cfg['ro_ch'], length= deslen) #*0.73)   

                


                self.add_gauss(ch=res_ch, name='ramp2',sigma = 0.15, length = 0.3, even_length=True)     
                self.add_pulse(ch=res_ch, name="Ramp", ro_ch=ro_chs,
                               style="flat_top",                                                              
                               envelope = "ramp2",                                                            
                               length= 1.0,#cfg["res_length"],                                                 
                               freq= cfg['res_freq_ge'],#cfg['res_freq_ge'],                                                 
                               phase=cfg['ro_phase'],                                                     
                               gain= 0.4#cfg['res_gain_ge']                                                    
                               )                      


                ## Here I try to use the DRAG pulse as defined in the QICK Code Base                                                      
                self.add_DRAG(ch=res_ch, name="Drag_o_clock", sigma=cfg['sigma'], length=4.0,
                              delta=-282.27, alpha=0.5, even_length=True)                    
                self.add_pulse(ch=res_ch, name="Drag", ro_ch=ro_chs,
                               style="arb",
                               envelope = "Drag_o_clock",
                               #length=cfg["res_length"],
                               freq=cfg['res_freq_ge'],#cfg['res_freq_ge'],
                               phase=cfg['ro_phase'],
                               gain=0.4#cfg['res_gain_ge']
                               )     


                #self.add_gauss(ch=res_ch, name="ramp", sigma=cfg['sigma'], length=cfg['sigma'] * 4, even_length=False)
                #self.add_pulse(ch=res_ch, name="qubit_pulse",
                #               style="arb",
                #               envelope="ramp",
                #               freq=cfg['qubit_freq_ge'],
                #               phase=cfg['qubit_phase'],
                #               gain=cfg['pi_amp'],
                #                )

                
                self.add_gauss(ch=qubit_ch, name="ramp", sigma=cfg['sigma'], length=cfg['sigma'] * 4, even_length=False)
                self.add_pulse(ch=qubit_ch, name="qubit_pulse",
                               style="arb",
                               envelope="ramp",
                               freq=cfg['qubit_freq_ge'],
                               phase=cfg['qubit_phase'],
                               gain=cfg['pi_amp'],
                               )


                
                self.delay(100.0)
                
            def _body(self, cfg):
                #self.pulse(ch=self.cfg["qubit_ch"], name = "qubit_pulse",t=0)
                #self.trigger(ros=[cfg['ro_ch']], pins=[0], t=0,  ddr4=True)
                self.delay_auto(0.0)
                #self.trigger(ros=[cfg['ro_ch']], pins=[0], t=0,  ddr4=True)
                self.pulse(ch=cfg['res_ch'], name="res_pulse", t=0)
                #self.trigger(ros=[cfg['ro_ch']], pins=[0], t=0,  ddr4=True)
                #self.delay_auto(1.0)
                #self.pulse(ch=cfg['res_ch'], name="res_pulse2", t=3)
                #self.pulse(ch=cfg['res_ch'], name="two_step", t=0)
                #self.pulse(ch=cfg['res_ch'], name = "Ramp", t=6)
                #self.pulse(ch=cfg['res_ch'], name = "Drag", t=8)
                self.trigger(ros=[cfg['ro_ch']], pins=[0], t=cfg['trig_time']-cfg['trig_time'],  ddr4=True)

                
                
        prog = MuxProgram(self.experiment.soccfg, reps=1, final_delay=0.0, cfg=self.config)
        iq_list = prog.acquire_decimated(self.experiment.soc, soft_avgs= 600)#self.config['soft_avgs'])
        t = prog.get_time_axis(ro_index=0)
        self.experiment.soc.reset_gens()
        if self.save_figs:
            (average_y_mag_values_last, average_y_mag_values_mid, average_y_mag_values_oct, DAC_attenuator1, DAC_attenuator2, ADC_attenuator) = self.plot_results(prog, iq_list)
        else:
            (average_y_mag_values_last, average_y_mag_values_mid, average_y_mag_values_oct, DAC_attenuator1, DAC_attenuator2, ADC_attenuator) = None, None, None, None, None, None,

        #print(t)
        #print(iq_list)
        return t, iq_list, self.config

#from qick.asm_v2 import AveragerProgramV2
#import matplotlib.pyplot as plt
#from build_state import *
#from expt_config import *
#from system_config import *
#

##StartTime = 2

#class TOFExperiment:
#    def __init__(self, QubitIndex,  outerFolder, experiment, round_num = 1, save_figs = True, title = False, qick_verbose=True, unmasking_resgain = False):
#        # CLEAR ENVELOPE MEMORY HERE                                                                              #  

#        # every time a class instance is created, these definitions are set
#        self.expt_name = "tof"
#        self.QubitIndex = QubitIndex
#        self.outerFolder = outerFolder
#        self.Qubit = 'Q' + str(QubitIndex)
#        self.exp_cfg = expt_cfg[self.expt_name]
#        self.experiment = experiment
#
#        # CLEAR ENVELOPE MEMORY HERE                                                                              #  
#        self.experiment.soc.reset_gens()
#        
#        self.save_figs = save_figs
#        self.title = title
#        self.qick_verbose=qick_verbose
#
#        if unmasking_resgain:
#            self.exp_cfg["list_of_all_qubits"] = [QubitIndex]
#
#        self.q_config = all_qubit_state(self.experiment,6)
#        self.round_num = round_num
#        if 'All' in self.Qubit:
#            self.config = {**self.q_config['Q0'], **self.exp_cfg}
#            print(f'Q {self.QubitIndex} Round {round_num} TOF configuration: ', self.config)
#        else:
#            self.config = {**self.q_config[self.Qubit], **self.exp_cfg}
#            print(f'Q {self.QubitIndex + 1} Round {round_num} TOF configuration: ',self.config)
#
#        self.config = {**self.q_config[self.Qubit], **self.exp_cfg}
#
#        ## CLEAR ENVELOPE MEMORY HERE
#        ##self.experiment.soc.reset_gens()
#
#            
#    def run(self):
#        soc = self.experiment.soc
#        soc.reset_gens()
#        #StartTime = 0.8                                                                                    
#        v_smag = 470
#        sigma = 0.04
#        alpha =1
#        l_two = 0.5
#        l_base =1.0
#        ti = 0.5 #0 + StartTime                                                                             
#        omega = 1##Test Freq                                                                                
#        #stepsmall = 20000                                                                                  
#        stepsmall = 900
#        #stepsmall = 16                                                                                     
#        deslen = 2
#        stepscorr = (((int(614.4 * deslen)) + 15) // 16) * 16
#        t_new = np.linspace(0, 2, stepscorr)
#        
#        # Defining the piecewise flat-top Gaussian function f(t, sigma, ti, L)                              
#        def f_flat(t_new, sigma, ti, L):
#            # Gaussian Rise                                                                                 
#            rise = np.exp(-((t_new - ti)**2) / (2 * sigma**2))
#            # Flat Top                                                                                      
#            top = np.ones_like(t_new)
#            # Gaussian Fall                                                                                 
#            fall = np.exp(-((t_new - (ti + L))**2) / (2 * sigma**2))
#
#            # Use np.select to apply conditions element-wise                                                
#            return np.select(
#                [t_new < ti, (t_new >= ti) & (t_new < ti + L), t_new >= ti + L],
#                [rise, top, fall]
#            )
#        pulse_shape = v_smag  * (f_flat(t_new, sigma, ti, l_base) + alpha * f_flat(t_new, sigma, ti, l_two))
#        #pulse = v_smag * np.sin(omega*t) * (f_flat(t, sigma, ti, l_base) + alpha * f_flat(t, sigma, ti, l_two))  #                                                                                                        
#        plt.plot(t_new,pulse_shape)
#        plt.show()
#
#        
#        class MuxProgram(AveragerProgramV2):
#            def _initialize(self, cfg):
#                ro_chs = cfg['ro_ch']
#                res_ch = cfg['res_ch']
#                # ro_ch = cfg['ro_ch']
#                # res_ch = cfg['res_ch']
#                qubit_ch = cfg['qubit_ch']
#                self.declare_gen(ch=res_ch, nqz=cfg['nqz_res'])
#                self.declare_readout(ch=cfg['ro_ch'], length=5.0)#cfg['res_length']*1.5)
#
#                self.add_readoutconfig(ch=ro_chs, name="myro",
#                                       freq = 5000, #freq=cfg['res_freq_ge'],
#                                       gen_ch=res_ch,
#                                       outsel='product')
#
#                #self.send_readoutconfig(ch=cfg['ro_ch'], name="myro", t=0)
#                # print(cfg["res_length"],cfg['ro_phase'],cfg['res_gain_ge'])
#                self.add_envelope(ch=res_ch, name="TwoStep", idata=pulse_shape)
#                
#                ##This is the normal signal
#                self.add_pulse(ch=res_ch, name="res_pulse4", ro_ch=ro_chs,
#                              style="const",
#                              length= 2, #cfg["res_length"],
#                              freq=5000,#cfg['res_freq_ge'],
#                              phase=cfg['ro_phase'],
#                              gain= 1.0   #cfg['res_gain_ge']+0.6
#                              )
#
#                #self.add_envelope(ch=res_ch, name="TwoStep",idata = pulse_shape)
#                
#                #self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_qubit'])
#
#                #self.add_gauss(ch=qubit_ch, name="ramp", sigma=cfg['sigma'], length=cfg['sigma'] * 4, even_length=False)
#
#                #self.add_pulse(ch=qubit_ch, name="qubit_pulse",
#                #       style="arb",
#                #       envelope="ramp",
#                #       freq=cfg['qubit_freq_ge'],
#                #       phase=cfg['qubit_phase'],
#                #       gain=cfg['pi_amp'],
#                #       )
##
#
#                ## This is my attempt at a Two Stepped Pulse 
#                ###===================================================================================================
#                #StartTime = 0.8
#                #v_smag = 470
#                #sigma = 0.04
#                #alpha =1
#                #l_two = 0.08
#                #l_base =0.25
#                #ti = 0.5 #0 + StartTime
#                #omega = 1##Test Freq
#
#                #stepsmall = 20000
#                #stepsmall = 900
#                #stepsmall = 16
#                #deslen = 2
#                
#                #stepscorr = (((int(614.4 * deslen)) + 15) // 16) * 16
#                
#                #t_new = np.linspace(0, 2, stepscorr)
#
#                # Defining the piecewise flat-top Gaussian function f(t, sigma, ti, L)
#                #def f_flat(t_new, sigma, ti, L):
#
#                #    # Gaussian Rise
#                #    rise = np.exp(-((t_new - ti)**2) / (2 * sigma**2))
#                #    # Flat Top
#                #    top = np.ones_like(t_new)
#                #    # Gaussian Fall
#                #    fall = np.exp(-((t_new - (ti + L))**2) / (2 * sigma**2))
#    
#                #    # Use np.select to apply conditions element-wise
#                #    return np.select(
#                #        [t_new < ti, (t_new >= ti) & (t_new < ti + L), t_new >= ti + L],
#                #        [rise, top, fall]
#                #    )
#                #pulse_shape = v_smag  * (f_flat(t_new, sigma, ti, l_base) + alpha * f_flat(t_new, sigma, ti, l_two))
#                #pulse = v_smag * np.sin(omega*t) * (f_flat(t, sigma, ti, l_base) + alpha * f_flat(t, sigma, ti, l_two))
#
#                #plt.plot(t_new,pulse_shape)
#                #plt.show()
#                
#                #self.add_envelope(ch=res_ch, name="TwoStep",idata = pulse_shape)
#                self.add_pulse(ch=res_ch, name="Step2", ro_ch=ro_chs,
#                                style="arb",
#                                envelope='TwoStep',
#                                freq= 5000,#cfg['res_freq_ge'],
#                                phase=cfg['ro_phase'],
#                                gain=cfg['res_gain_ge']+0.6 
#                                )
#                
#                ##===================================================================================================
#                             
#               
#            def _body(self, cfg):
#                #self.send_readoutconfig(ch=cfg['ro_ch'], name="myro", t=0)
#
#                #self.pulse(ch=self.cfg["qubit_ch"], name="qubit_pulse", t=0)  # play pulse
#                #self.delay_auto(0.0)
#                
#                #self.trigger(ros=[cfg['ro_ch']], pins=[0], t=0, ddr4=True)
#                #self.pulse(ch=cfg['res_ch'], name="res_pulse3", t=0)
#                self.pulse(ch=cfg['res_ch'], name="Step2", t=0)
#                #self.pulse(ch=cfg['res_ch'], name="res_pulse4", t=0)
#            
#                #self.pulse(ch=cfg['res_ch'], name="res_pulse", t=3)
#                #self.pulse(ch=cfg['res_ch'], name="res_pulse", t=0)
#               # self.trigger(ros=[cfg['ro_ch']], pins=[0], t=0, ddr4=True)
#                self.trigger(ros=[cfg['ro_ch']], pins=[0], t=0.0, ddr4=True)
#                #self.trigger(ros=[cfg['ro_ch']], pins=[0], t=0.7, ddr4=True)
 
#        prog = MuxProgram(self.experiment.soccfg, reps=1, final_delay=0.0, cfg=self.config)
#        iq_list = prog.acquire_decimated(self.experiment.soc, soft_avgs=self.config['soft_avgs'])
#        t = prog.get_time_axis(ro_index=0)
#        if self.save_figs:
#            (average_y_mag_values_last, average_y_mag_values_mid, average_y_mag_values_oct, DAC_attenuator1, DAC_attenuator2, ADC_attenuator) = self.plot_results(prog, iq_list)
#        else:
#            (average_y_mag_values_last, average_y_mag_values_mid, average_y_mag_values_oct, DAC_attenuator1, DAC_attenuator2, ADC_attenuator) = None, None, None, None, None, None,

#        return t, iq_list, self.config


    def plot_results(self, prog, iq_list):
        t = prog.get_time_axis(ro_index=0)
        fig, axes = plt.subplots(1, 1, figsize=(12, 12))
        phase_offsets=[]
        average_y_mag_values_mid = []
        average_y_I_values_mid = []
        average_y_Q_values_mid = []
        average_y_mag_values_oct = []
        average_y_I_values_oct = []
        average_y_Q_values_oct = []
        average_y_mag_values_last = []
        average_y_I_values_last = []
        average_y_Q_values_last = []

        plot = axes
        plot.plot(t, iq_list[0][:, 0], label="I value")
        plot.plot(t, iq_list[0][:, 1], label="Q value")
        magnitude = np.abs(iq_list[0].dot([1, 1j]))
        plot.plot(t, magnitude, label="magnitude")
        plot.legend()
        plot.set_ylabel("a.u.")
        plot.set_xlabel("us")
        #plot.set_ylim(-0,15)
        plot.axvline(0.75, c='r')

        phase_offset = np.angle(iq_list[0].dot([1, 1j]).sum(), deg=True)
        # print("measured phase %f degrees" % (phase_offset))
        phase_offsets.append(phase_offset)


        # Find indices of the middle three x-values
        mid_index = len(t) // 2
        indices_mid = [mid_index - 15, mid_index, mid_index + 15] #average 7 values

        one_eighth_index = len(t) // 15
        indices_oct = [one_eighth_index - 1, one_eighth_index, one_eighth_index + 1]

        indices_last = slice(-15, None)  # this will grab the last 7 elements

        # Calculate average y-values for I, Q, and magnitude
        avg_i_mid = np.mean(iq_list[0][indices_mid, 0])
        avg_q_mid = np.mean(iq_list[0][indices_mid, 1])
        avg_mag_mid = np.mean(magnitude[indices_mid])

        # Calculate average y-values for I, Q, and magnitude
        avg_i_oct = np.mean(iq_list[0][indices_oct, 0])
        avg_q_oct = np.mean(iq_list[0][indices_oct, 1])
        avg_mag_oct = np.mean(magnitude[indices_oct])

        # Calculate average y-values for I, Q, and magnitude
        avg_i_last = np.mean(iq_list[0][indices_last, 0])
        avg_q_last = np.mean(iq_list[0][indices_last, 1])
        avg_mag_last = np.mean(magnitude[indices_last])

        # Append the average magnitude to the list, you can change this to average I or Q.
        average_y_mag_values_mid.append(avg_mag_mid)
        average_y_I_values_mid.append(avg_i_mid)
        average_y_Q_values_mid.append(avg_q_mid)

        average_y_mag_values_oct.append(avg_mag_oct)
        average_y_I_values_oct.append(avg_i_oct)
        average_y_Q_values_oct.append(avg_q_oct)

        average_y_mag_values_last.append(avg_mag_last)
        average_y_I_values_last.append(avg_i_last)
        average_y_Q_values_last.append(avg_q_last)
        if self.title:
            plt.suptitle(f"TOF DAC_Att_1:{self.experiment.DAC_attenuator1} DAC_Att_2:{self.experiment.DAC_attenuator2} ADC_Att:{self.experiment.ADC_attenuator}", fontsize=24, y=0.95)


        # Save
        if self.save_figs:
            outerFolder_expt = os.path.join(self.outerFolder, self.expt_name)
            self.experiment.create_folder_if_not_exists(outerFolder_expt)
            now = datetime.datetime.now()
            formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
            if 'All' in self.Qubit:
                file_name = os.path.join(outerFolder_expt, f"R_{self.round_num}" + f"Q_{self.QubitIndex}" + f"{formatted_datetime}_" + self.expt_name + ".png")
            else:
                file_name = os.path.join(outerFolder_expt, f"R_{self.round_num}" + f"Q_{self.QubitIndex+1}" + f"{formatted_datetime}_" + self.expt_name + ".png")
            plt.savefig(file_name, dpi=50)
            #plt.show()
            plt.close(fig)

        return average_y_mag_values_last, average_y_mag_values_mid, average_y_mag_values_oct, self.experiment.DAC_attenuator1, self.experiment.DAC_attenuator2, self.experiment.ADC_attenuator



