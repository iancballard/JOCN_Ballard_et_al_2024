from psychopy.visual import ShapeStim, Polygon
from psychopy import core, visual, event, logging
import dots
import numpy as np
import pickle
import os.path as op
import glob

def check_abort(keys):
    if 'escape' in keys:
        core.quit()
        
def init_stims(p, win):
    #dots
    shape_list = ['cross','circle']
    color_list = ['green','pink']
    dotstims = {}
    for color_idx, color in enumerate(color_list):
        for shape in shape_list:
            dotstims[color + '_' + shape] = [
            dots.RandomDotMotion(win, ##high color coherence high shape coherence 
                                    color = p.dot_colors[color_idx],
                                    size = p.dot_size,
                                    shape = shape,
                                    density = p.dot_density * p.coherence['color'] * p.coherence['shape'] ,
                                    aperture = p.dot_aperture),
            dots.RandomDotMotion(win, ##low color coherence high shape coherence 
                                    color = p.dot_colors[1 - color_idx],
                                    size = p.dot_size,
                                    shape = shape,
                                    density = p.dot_density * (1 - p.coherence['color']) * p.coherence['shape'] ,
                                    aperture = p.dot_aperture),             
            dots.RandomDotMotion(win, ##high color coherence low shape coherence 
                                    color = p.dot_colors[color_idx],
                                    size = p.dot_size,
                                    shape = [x for x in shape_list if x != shape][0], #other shape
                                    density = p.dot_density * p.coherence['color'] * (1 - p.coherence['shape']),
                                    aperture = p.dot_aperture),
            dots.RandomDotMotion(win, ##low color coherence low shape coherence 
                                    color = p.dot_colors[1 - color_idx],
                                    size = p.dot_size,
                                    shape = [x for x in shape_list if x != shape][0], #other shape
                                    density = p.dot_density * (1 - p.coherence['color']) * (1 - p.coherence['shape']),
                                    aperture = p.dot_aperture)]
    
    #polygonal cue
    cue = Polygon(win,
                    radius=p.poly_radius,
                    lineColor=p.poly_color,
                    fillColor=p.poly_color,
                    lineWidth=p.poly_linewidth)
    
    
    return dotstims, cue
    

def present_dots_record_keypress(p, win, dotstims, cue, clock, color, shape, motion, rule):
    keys = False
    
    #polygonal cue
    cue.setEdges(p.cues[rule])
    cue.draw()

    #randomly initialize dot locations
    for ds in dotstims[color + '_' + shape]:
        ds.reset()
    win.flip()

    #loop through frames
    nframes = p.decision_dur * win.framerate
    for frameN in range(nframes): #update dot position
        #loop through 4 component dotstims
        for ds in dotstims[color + '_' + shape]:
            ds.update(p.motion_direction_map[motion],
                                        p.coherence['motion'])
            ds.draw()

        #draw cue
        cue.setEdges(p.cues[rule])
        cue.draw()

        win.flip()

        #detect keypresses
        if not keys: #only record first
            keys = event.getKeys(keyList = ['1','2'],
                                timeStamped = clock)  # get keys from event buffer
        
    return keys
     
def get_basic_objects(win, p):
    
    #fixation cross
    fixation = visual.TextStim(win,
        color = p.fixation_color,
        text='+')
        
    reward = visual.TextStim(win,
        color = p.rew_color,
        text='+')
    
    return fixation, reward

def update_rule_names(p):

    p.instruct_text['intro'] = [x.replace('RULE1',p.cue_map[p.cues['color']]) for x in p.instruct_text['intro']]
    p.instruct_text['intro'] = [x.replace('RULE2',p.cue_map[p.cues['motion']]) for x in p.instruct_text['intro']]
    p.instruct_text['intro'] = [x.replace('RULE3',p.cue_map[p.cues['shape']]) for x in p.instruct_text['intro']]
    

def setup_miniblocks_and_coherences(p):
    
    
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
    p.correct_resp = []
    p.active_rule = []
    p.miniblock = []
    p.coherences = dict(color = [], motion = [], shape = [])
    
    for block_num, block in enumerate(p.miniblocks):
        
        #create list of 'active' rules according to each miniblock
        block_rules = block.split('_') #2 active rules in a block
        block_rules = block_rules * int(p.ntrials_per_miniblock/2)
        np.random.shuffle(block_rules)
        
        ########################
        #make sure first rule of new miniblock is the rule that wasn't in old miniblock
        ########################
        
        #figure out identity of unique rule
        if block_num > 0:
            old_rules = p.miniblocks[block_num-1].split('_') #2 old rules
            unique_rule = list(set([x for x in block_rules if x not in old_rules]))[0]
        
            #reshuffle
            while block_rules[0] != unique_rule:
                np.random.shuffle(block_rules)
        
        ########################
        #####get correct responses ####
        ########################

        for n,rule in enumerate(block_rules):
            trial_idx = block_num*p.ntrials_per_miniblock + n
            resp = p.dimension_correct_resp[rule][trial_idx]
            
            p.correct_resp.append(resp)
            p.active_rule.append(rule)
            p.miniblock.append(block)

        ########################
        # create coherences within miniblocks
        ########################
        coh = {}
        for dimension in ['color','motion','shape']:
            coh[dimension] = np.zeros(len(block_rules))
    
            #index when rule is active
            active_times = np.array([x == dimension for x in block_rules])

            #coherence levels
            coherence = np.linspace(p.coherence_floor[dimension],
                                    p.coherence_floor[dimension] + p.coherence_range[dimension],
                                    num=p.n_coherence_levels)
            coherence = list(coherence)*2
            np.random.shuffle(coherence)

            if sum(active_times) > 0: #this means that rule is active in this miniblock
                coh[dimension][active_times] = coherence
    
                #shuffle again and set other trials
                np.random.shuffle(coherence)
                coh[dimension][~active_times] = coherence

            else: #this means this rule is not active, so set randomly
                coherence = coherence*2
                np.random.shuffle(coherence)
                coh[dimension] = coherence
            
            p.coherences[dimension].extend(list(coh[dimension]))
            
def set_subject_specific_params(p):
    
    p.coherence_floor = {}
    for rule in ['color','shape','motion']:
        
        #should be 7 blocks, but check for less just in case
        files = glob.glob(op.join(op.abspath('./data'), p.sub + '_training_' + rule + '*'))
        file_idx = [int(x.split('_')[-2]) for x in files]
        f = files[file_idx.index(max(file_idx))]
        print(f)
        
        with open(f, 'rb') as f:
            training_data = pickle.load(f)
            
        mean_coherence = np.mean(training_data.coherence_record[rule])
        
        if rule == 'motion':
            p.coherence_floor[rule] = np.round(max(mean_coherence, .02),2)
        else:
            p.coherence_floor[rule] = np.round(max(mean_coherence, .52),2) 
    
    print(p.coherence_floor)