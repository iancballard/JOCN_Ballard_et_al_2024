import sys, getopt
import time
import numpy as np
import pickle
import os.path as op
import pandas as pd
import os
import glob
from psychopy.monitors import Monitor
import warnings
from colormath.color_objects import LCHabColor, sRGBColor
from colormath.color_conversions import convert_color
from psychopy import core, visual, event, logging
from numpy.random import RandomState
pd.options.mode.chained_assignment = None  # suppress chained assignment warning

class Params(object):
    
    def __init__(self, mode, p_file='params'):
        """Initializer for the params object.
        Parameters
        ----------
        exp_name: string, name of the dict we want from the param file
        p_file: string, the name of a parameter file
        """
        

        self.mode = mode
        
        #read paramaters from file
        im = __import__(p_file)
        param_dict = getattr(im, mode)
        for key, val in param_dict.items():
            setattr(self, key, val)
        
        timestamp = time.localtime()
        self.timestamp = time.asctime(timestamp)
        self.date = time.strftime("%Y-%m-%d", timestamp)
        self.time = time.strftime("%H-%M-%S", timestamp)
        
    def set_by_cmdline(self,argv):
        #parse inputs
        help_str = 'rules.py -s <subject_id> -r <run> -c <counterbalance> -sess <sess> -scan <train_or_scan>'
        try:
          opts, args = getopt.getopt(argv,"s:r:c:d:t",["subject=", "run=", "cb=",'day=','scan='])
          if len(opts) == 0:
              print(help_str)
              sys.exit(2)          
        except getopt.GetoptError:
          print(help_str)
          sys.exit(2)
      
        for opt, arg in opts:
            if opt == '-h':
                print(help_str)
                sys.exit()
            elif opt in ("-s", "--subject"):
                self.sub = arg
            elif opt in ("-r", "--run"):
                self.run = arg
            elif opt in ("-c", "--cb"):
                self.cb = arg
            elif opt in ("-d", "--day"):
                self.session = arg
    
    def set_scan_mode(self):

        self.monitor_name = 'mbpro'
        self.init_wait_time = 2
        self.end_wait_time = 2
        self.isis = list(np.ones(len(self.isis), dtype=int))
        self.itis = list(np.ones(len(self.itis), dtype=int) + 2)                       
        self.update_coherence = False #continue calibration of RTs

    def randomize_shape_assignments(self):
        self.hash_sub_id = sum(map(ord, self.sub))
        
        rs = RandomState(self.hash_sub_id) #set random state
        cue_shapes = [3,4,5]
        rs.shuffle(cue_shapes)
        self.cues = dict(shape = cue_shapes[0],
                          color = cue_shapes[1],
                          motion = cue_shapes[2])
        self.cue_map = {3:'Triangle', 4:'Diamond', 5: 'Pentagon'}
         
    
    def randomize_rewarded_rule(self):
        rs = RandomState(self.hash_sub_id) #set random state
        rules = ['motion','color','shape']
        self.rewarded_rule = rs.choice(rules)
    
    def lch_to_rgb(self, p):
        """Convert the color values from Lch to RGB."""
        rgbs = []
        for lightness, hue in zip(p.lightnesses, p.hues):
            lch = LCHabColor(lightness, p.chroma, hue)
            rgb = convert_color(lch, sRGBColor).get_value_tuple()
            rgbs.append(rgb)
        
        rgbs = np.array(rgbs)
        rgbs = rgbs*2-1
        rgbs = [tuple(x) for x in rgbs]
        
        return np.array(rgbs)
             
    def set_subject_specific_params(self):
        
        f = 'sub_params'
        im = __import__(f)
        coh = getattr(im, 'coherences')
        self.coherence_floor = coh[self.sub]
    
    def run_info(self):

        #load run timings
        run_fname = op.join(op.abspath('timing'),'models','run' + str(int(self.run) - 1) + '.csv')
        self.run_timing = pd.read_csv(run_fname)
        run_fname = op.join(op.abspath('timing'),'designs','run' + str(int(self.run) - 1) + '.csv')
        self.run_design = pd.read_csv(run_fname)
    
        #update datastruct
        self.trial_type = self.run_design['trial_type']
        self.itis = self.run_design['iti']
        self.isis = self.run_design['isi']
        self.correct_resp = self.run_design['correct_resp']    
        self.color_direction = self.run_design['color_direction']    
        self.motion_direction = self.run_design['motion_direction']    
        self.mag = self.run_design['magnitude']    
        self.correct_dim = list(self.run_design['correct_dim'])
        self.correct_dim.append('color') #useful to have an extra trial for n+1 indexing below
    
        #determine correct responses for learning task
        self.motion_map = {'left':'3','right':'4'}
        self.color_map = {'left':'1','right':'2'}
        self.correct_resp_learn = []
        for color,motion in list(zip(self.color_direction, self.motion_direction)):
            self.correct_resp_learn.append([self.color_map[color],self.motion_map[motion]])
        
    def max_priority(self):
    
        #quit finder
        applescript="\'tell application \"Finder\" to quit\'"
        shellCmd = 'osascript -e '+applescript
        # os.system(shellCmd)
    
        #make processes maximum priority
        new_nice = -20
        # sysErr = os.system("sudo renice -n %s %s" % (new_nice, os.getpid()))
        # if sysErr:
        #     print('Warning: Failed to renice, probably you arent authorized as superuser')
        #
            
    def launch_window(self, test_refresh=True, test_tol=.5):
        """Load window info"""
        #taken from Mwaskom cregg
        try:
            mod = __import__("monitors")
        except ImportError:
            sys.exit("Could not import monitors.py in this directory.")

        try:
            minfo = getattr(mod, self.monitor_name)
        except IndexError:
            sys.exit("Monitor not found in monitors.py")

        fullscreen = self.full_screen
        size = minfo["size"] if fullscreen else (800, 600)

        monitor = Monitor(name=minfo["name"],
                          width=minfo["width"],
                          distance=minfo["distance"])
        monitor.setSizePix(minfo["size"])
        
        info = dict(units=self.monitor_units,
                    fullscr=fullscreen,
                    color=self.window_color,
                    size=size,
                    monitor=monitor)

        if "framerate" in minfo:
            self.framerate = minfo["framerate"]

        self.name = self.monitor_name
        self.__dict__.update(info)
        self.window_kwargs = info
        
        
        """Open up a presentation window and measure the refresh rate."""
        stated_refresh_hz = self.framerate

        # Initialize the Psychopy window object
        win = visual.Window(**self.window_kwargs)

        # Record the refresh rate we are currently achieving
        if self.test_refresh or stated_refresh_hz is None:
            win.setRecordFrameIntervals(True)
            logging.console.setLevel(logging.CRITICAL)
            flip_time, _, _ = visual.getMsPerFrame(win)
            observed_refresh_hz = 1000 / flip_time
            print('observed_refresh_hz',observed_refresh_hz)
            
        # Possibly test the refresh rate against what we expect
        if self.test_refresh and stated_refresh_hz is not None:
            refresh_error = np.abs(stated_refresh_hz - observed_refresh_hz)
            print('refresh_error',refresh_error)
            if refresh_error > test_tol:
                msg = ("Observed refresh rate differs from expected by {:.3f} Hz"
                       .format(refresh_error))
                raise RuntimeError(msg)

        # Set the refresh rate to use in the experiment
        if stated_refresh_hz is None:
            msg = "Monitor configuration does not have refresh rate information"
            warnings.warn(msg)
            win.framerate = observed_refresh_hz
        else:
            win.framerate = stated_refresh_hz

        return win
   
