import psychopy
from psychopy import core, visual, event, logging
from psychopy.visual import ShapeStim, Polygon
from psychopy.monitors import Monitor
import socket
import json
import glob
import numpy as np
import os.path as op
import sys, getopt
import time
import pickle
import datastruct 
import pandas as pd
import seaborn as sns
from textwrap import dedent
from trial_functions import check_abort, init_stims, present_dots_record_keypress, get_basic_objects, update_rule_names
      

def draw_stim(win, stim, nframes):    
    for frameN in range(int(nframes)):
        stim.draw()
        win.flip()
        check_abort(event.getKeys())

#annoying function because setting opacity for textstim doesn't work    
def draw_error(win, nframes, fixation_color):
    for frameN in range(int(nframes)):    
        error = visual.TextStim(win,
                color = fixation_color,
                text='+',
                opacity = float((frameN % 8) >= 4))
        error.draw()
        win.flip()
        check_abort(event.getKeys())
                
def experiment_module(p, win):

    ########################
    #### Instructions ####
    ########################
    if p.step_num == 0:
        for txt in p.instruct_text['intro']:
            message = visual.TextStim(win,
                height = p.text_height,
                text=dedent(txt))
            message.draw()
            win.flip()
            keys = event.waitKeys(keyList  = ['1'])
                
    if p.step_num < 3: #run through instructions for each feature the first time
        for txt in p.instruct_text[p.training_step]:
            message = visual.TextStim(win,
                height = p.text_height,
                text=dedent(txt))
            message.draw()
            win.flip()
            keys = event.waitKeys(keyList  = ['1'])     
    else:
        for txt in p.instruct_text['break_txt']:
            txt = txt.replace('COMPLETED',str(p.step_num))
            txt = txt.replace('TOTAL',str(p.total_trials))
            txt = txt.replace('FEATURE',p.training_step)
            
            message = visual.TextStim(win,
                height = p.text_height,
                text=dedent(txt))
            message.draw()
            win.flip()                
            

    ########################
    #### Visual Objects ####
    ########################
    
    #colors
    p.dot_colors = p.lch_to_rgb(p)

    #get fixation cross and feedback info
    fixation, reward = get_basic_objects(win, p)
                                                  
    ############################
    #### Set up Trial Order ####
    ############################
    p.dimension_val = {}
    for dimension in ['color','motion','shape']:
        
        if dimension == 'color':
            direction = ['green','pink'] * int(p.ntrials/2)
        elif dimension == 'shape':
            direction = ['circle','cross'] * int(p.ntrials/2)
        elif dimension == 'motion':
            direction = ['up','down'] * int(p.ntrials/2)
                        
        correct_resp = ['1','2'] * int(p.ntrials/2)
        
        #shuffle
        resp = list(zip(direction, correct_resp))
        np.random.shuffle(resp)
        
        p.dimension_val[dimension], correct_resp = zip(*resp)
        
        if p.training_step == dimension:
            p.correct_resp = correct_resp
    
    ########################
    #### Coherence #########
    ########################
    p.coherences = dict(color = [], motion = [], shape = [])
    for dimension in ['color','motion','shape']:

        coherence = np.linspace(p.coherence_floor[dimension],
                                p.coherence_floor[dimension] + p.coherence_range[dimension],
                                num=p.n_coherence_levels)
                                
        coherence = list(coherence)* int(p.ntrials/p.n_coherence_levels)
        np.random.shuffle(coherence)
        p.coherences[dimension].extend(list(coherence))
    print(p.coherences)
    
    ########################
    #### Run Experiment ####
    ########################
    
    # notify participant
    if p.step_num < 2: #after 2nd intro, "space to continue" is in instructions
        message = visual.TextStim(win,
            height = p.text_height,
            text='Press space to begin')
        message.draw()
        win.flip()

    #wait for scan trigger
    keys = event.waitKeys(keyList  = ['space'])

    #start timer
    clock = core.Clock()   
    
    #draw fixation
    draw_stim(win,
                fixation,
                1 * win.framerate)
                
    p.resp = []
    p.rt = []
    p.choice_times = []
    p.feedback_times = []
    p.incorrect = [] #only for switch
    p.bank = 0
    win.recordFrameIntervals = True
    num_correct= 0
    num_errors = 0
    
    for n in range(p.ntrials):
        
        ############################
        ###dot stim/choice period###
        ############################
        
        #set up coherences (before initializing dots)
        for rule in ['color','shape','motion']:
            p.coherence[rule] = p.coherences[rule][n]
            
        #initialize dots    
        dotstims, cue = init_stims(p, win)
        
        #set up response key and rt recording
        rt_clock = clock.getTime()
        p.choice_times.append(rt_clock)
        correct = True
        
        #color, motion and shape for this trial
        color_idx = p.dimension_val['color'][n]
        motion_idx = p.dimension_val['motion'][n]
        shape_idx= p.dimension_val['shape'][n]

        keys = present_dots_record_keypress(p,
                                            win,
                                            dotstims,
                                            cue,
                                            clock,
                                            color_idx, shape_idx, motion_idx,
                                            p.training_step)
        
        #record keypress
        if not keys:
            resp = np.NaN
            p.resp.append(resp)
            p.rt.append(np.NaN)
        else:
            resp = keys[0][0]
            p.resp.append(resp)
            p.rt.append(keys[0][1] - rt_clock)  
        
        #####################
        ###feedback period###
        #####################
        
        #show feedback    
        nframes = p.feedback_dur * win.framerate
            
        if np.isnan(p.rt[-1]): 
            correct = False
            draw_error(win, nframes, p.too_slow_color)
            
        elif str(resp) != str(p.correct_resp[n]):
            correct = False            
            draw_error(win, nframes, p.fixation_color)

        if not correct:
            num_errors +=1   
        
        ################
        ###iti period###
        ################
        
        draw_stim(win,
                    fixation,
                    p.iti * win.framerate)
    

    print('errors',num_errors, num_errors/p.ntrials)
    print('mean_rt',np.nanmean(p.rt))
    print('\nOverall, %i frames were dropped.\n' % win.nDroppedFrames)

    #save data
    out_f = op.join(p.outdir,p.sub + '_psychophys_' + p.training_step + '_' + str(p.step_num) + '_' + p.mode + '.pkl')
    while op.exists(out_f):
        out_f = out_f[:-4] + '+' + '.pkl'        

    with open(out_f, 'wb') as output:
        pickle.dump(p, output, pickle.HIGHEST_PROTOCOL)
                                      
def main(arglist):

    ##################################
    #### Parameter Initialization ####
    ##################################
    
    # Get the experiment parameters
    mode = arglist.pop(0)
    p = datastruct.Params(mode)
    p.set_by_cmdline(arglist)
    p.randomize_shape_assignments()
    
        
    ##################################
    #### Window Initialization ####
    ##################################
    
    # Open up the stimulus window
    win = p.launch_window(p)        
    logging.console.setLevel(logging.WARNING)
    
    #hide mouse
    event.Mouse(visible = False)
                      
    ########################
    #### Task Blocks ####
    ########################
    
    #loop through training steps
    p.total_trials = len(p.training_blocks)
    for n,training_step in enumerate(p.training_blocks):
        
        if n < 3: #short blocks in the beginning to get people back on track
            p.ntrials = p.ntrials_init
        else:
            p.ntrials = p.ntrials_test
            
        p.training_step = training_step
        p.step_num = n
        experiment_module(p, win)
        
    core.quit()
   
    
if __name__ == "__main__":
   main(sys.argv[1:])    
   

