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
from trial_functions import check_abort, init_stims, present_dots_record_keypress, get_basic_objects, update_rule_names, setup_miniblocks_and_coherences
      

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
    #######################
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
    
    ########################
    #### Run Experiment ####
    ########################
    
    #set up trial structure
    setup_miniblocks_and_coherences(p)
    
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
    p.rew = []
    p.bank = 0
    win.recordFrameIntervals = True
    num_correct= 0
    num_errors = 0
    for n in range(p.ntrials):
        
        ############################
        ###dot stim/choice period###
        ############################
        
        #set up color coherences (before initializing dots)
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
        
        #give reward if active rule is rewarded
        rew = False
        if correct:
            if rule == p.rewarded_rule: #high reward rule
                if np.random.rand() <= p.p_rew_high: #coin flip
                    rew = True
            else: #low reward rule
                if np.random.rand() <= p.p_rew_low: #coin flip
                    rew = True
        p.rew.append(rew)
        
        ################
        ###iti period###
        ################
        
        if rew: #change fixation cross color
            draw_stim(win,
                        fixation,
                        p.fb_iti * win.framerate)
            #draw reward cue
            draw_stim(win,
                        reward,
                        p.fb_dur * win.framerate)
            draw_stim(win,
                        fixation,
                        p.iti * win.framerate)
        
        else: #no reward
            draw_stim(win,
                        fixation,
                        p.iti * win.framerate)
    

    print('errors',num_errors, num_errors/p.ntrials)
    print('mean_rt',np.nanmean(p.rt))
    print('\nOverall, %i frames were dropped.\n' % win.nDroppedFrames)

    #save data
    out_f = op.join(p.outdir,p.sub + '_reward_' + p.block_kind + '_' + str(p.step_num) + '.pkl')
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
    p.randomize_rewarded_rule()
    # p.set_subject_specific_params()
    print(p.rewarded_rule)
    
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

    #reward blocks
    for n in range(p.num_rew_blocks):
        p.num_blocks = p.num_rew_blocks
        p.block_kind = 'reward'
        p.step_num = n
        c = experiment_module(p, win)
        
    #break period
    txt = 'Great job. You will now take a break. Please relax for the next NUM minutes. After, you will have two more blocks. These blocks will not have rewards, but please try your hardest.'
    for break_min in range(p.break_dur):
        time_left = p.break_dur - break_min
        break_txt = txt.replace('NUM', str(time_left))
        message = visual.TextStim(win,
            height = p.text_height,
            text=break_txt)
        message.draw()
        win.flip()
        core.wait(60)
    
    #test blocks
    for n in range(p.num_test_blocks):
        p.p_rew_high = 0
        p.p_rew_low = 0
        p.block_kind = 'test'
        p.num_blocks = p.num_test_blocks
        p.step_num = n
        c = experiment_module(p, win)
        
    core.quit()
   
    
if __name__ == "__main__":
   main(sys.argv[1:])    
   

