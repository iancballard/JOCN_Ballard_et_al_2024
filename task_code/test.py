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
    update_rule_names(p)
    if p.step_num == 0:
        for txt in p.instruct_text['intro']:
            message = visual.TextStim(win,
                height = p.text_height,
                text=dedent(txt))
            message.draw()
            win.flip()
            keys = event.waitKeys(keyList  = ['space'])
                   
    else:
        for txt in p.instruct_text['break_txt']:
            txt = txt.replace('COMPLETED',str(p.step_num))
            txt = txt.replace('TOTAL',str(p.num_blocks))
            
            message = visual.TextStim(win,
                height = p.text_height,
                text=dedent(txt))
            message.draw()
            win.flip()
            keys = event.waitKeys(keyList  = ['space'])           
    

    ########################
    #### Visual Objects ####
    ########################
    
    #colors
    p.dot_colors = p.lch_to_rgb(p)
    
    #dotstims init
    dotstims, cue = init_stims(p, win)

    #get fixation cross and feedback info
    fixation, reward = get_basic_objects(win, p)
       
    ############################
    #### Set up Trial Order ####
    ############################
    p.correct_resp = []
    p.active_rule = []
    p.miniblock = []
    p.coherences = dict(color = [], motion = [], shape = [])
    if p.block_id == 'test':
        
        #Pseudorandomize miniblock structure
        p.miniblocks = []
        for i in range(p.num_block_reps):
            mini = list(p.miniblock_ids) #deepcopy
            np.random.shuffle(mini)
        
            #make sure no repititions
            if len(p.miniblocks) > 0:
                while mini[0] == p.miniblocks[-1]:
                    np.random.shuffle(mini)
            p.miniblocks.extend(mini)
        print(p.miniblocks)
    
    
        p.ntrials = p.ntrials_per_miniblock * len(p.miniblocks)

        #create random color, shape, motion patterns for all trials
        p.dimension_val = {}
        p.dimension_correct_resp = {}
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
        
            p.dimension_val[dimension], p.dimension_correct_resp[dimension] = zip(*resp)
     

        #create vector of correct response that implictly define "correct" rule
        for block_num, block in enumerate(p.miniblocks):
        
            #create list of 'active' rules according to each miniblock
            block_rules = block.split('_') #2 active rules in a block
            block_rules = block_rules * int(p.ntrials_per_miniblock/2)
            np.random.shuffle(block_rules)
        
            ########################
            #### TODO make coherences are balanced for 'active'rule ####
            ########################
        
            #create coherences within miniblocks
            for dimension in ['color','motion','shape']:
        
                coherence = np.linspace(p.coherence_floor[dimension],
                                        p.coherence_floor[dimension] + p.coherence_range[dimension],
                                        num=int(p.ntrials_per_miniblock/3))
                coherence = list(coherence)*3 #2 repeats
                np.random.shuffle(coherence)
                p.coherences[dimension].extend(list(coherence))
        
            #get correct responses
            for n,rule in enumerate(block_rules):
                trial_idx = block_num*p.ntrials_per_miniblock + n
                resp = p.dimension_correct_resp[rule][trial_idx]
            
                p.correct_resp.append(resp)
                p.active_rule.append(rule)
                p.miniblock.append(block)
    
    else:
        
        p.active_rule = [p.block_id] * p.n_train_trials
        p.ntrials = p.n_train_trials
    
        #create coherences
        num_coherences_per_training = int(p.n_train_trials / (p.ntrials_per_miniblock/3)) + 1 
        for i in range(num_coherences_per_training):
            for dimension in ['color','motion','shape']:

                coherence = np.linspace(p.coherence_floor[dimension],
                                        p.coherence_floor[dimension] + p.coherence_range[dimension],
                                        num=int(p.ntrials_per_miniblock/3))
                                    
                np.random.shuffle(coherence)
                p.coherences[dimension].extend(list(coherence))
        
        #create random color, shape, motion patterns for all trials
        p.dimension_val = {}
        p.dimension_correct_resp = {}
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
        
            p.dimension_val[dimension], p.dimension_correct_resp[dimension] = zip(*resp)
            
        p.correct_resp = p.dimension_correct_resp[p.block_id]
    
    ########################
    #### Run Experiment ####
    ########################
    print(p.coherences)
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
    p.correct = []
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
        color = p.dimension_val['color'][n]
        motion = p.dimension_val['motion'][n]
        shape = p.dimension_val['shape'][n]
        rule = p.active_rule[n]
        print(color, motion, shape, rule)
        


        keys = present_dots_record_keypress(p,
                                            win,
                                            dotstims,
                                            cue,
                                            clock,
                                            color, shape, motion,
                                            rule)
        
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
        p.correct.append(correct)
        
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
    out_f = op.join(p.outdir,p.sub + '_test_' + p.block_id + '_' + str(p.step_num) + '.pkl')
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
    p.set_subject_specific_params()
    
    ##################################
    #### Window Initialization ####
    ##################################
    
    # Create a window
    win = p.launch_window(p)        
    logging.console.setLevel(logging.WARNING)
    
    #hide mouse
    event.Mouse(visible = False)
                  
    ########################
    #### Task Blocks ####
    ########################
    p.randomize_test_blocks()
    p.blocks = np.repeat(p.blocks, p.num_test_within_blocks)
    p.num_blocks = len(p.blocks)
    print(p.blocks)
    for n,block in enumerate(p.blocks):
        p.step_num = n
        p.block_id = block
        c = experiment_module(p, win)
        
    core.quit()
   
    
if __name__ == "__main__":
   main(sys.argv[1:])    
   

