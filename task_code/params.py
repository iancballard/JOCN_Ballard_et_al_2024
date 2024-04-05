from copy import deepcopy
import os.path as op
from textwrap import dedent
import seaborn as sns
import numpy as np

#base parameters for all runs
base = dict(
    decision_dur = 2,
    feedback_dur = .5,
    iti = 1,
    outdir = op.abspath('./data/'),
    window_color = '#0a0a0a',#'#545454',
    full_screen = True,
    test_refresh = True,
    coherence = dict(color = .8,
                           motion = .8,
                           shape = .8),
    coherence_floor = dict(color = .52,
                           motion = .02,
                           shape = .52),
    coherence_range = dict(color = .12,
                           motion = .12,
                           shape = .12),
    n_coherence_levels = 4,
    rew_color = '#997020',
    monitor_units = 'deg',
    fixation_color = -.2,
    text_height = .5,
)

#dot parameters
base.update(
    dot_size = .2,#.15, #degrees of visual angle
    dot_aperture = 6, #degrees of visual angle
    dot_density = 24,
    motion_direction_map = {'up':270, 'down':90},
    chroma = 50,
    lightnesses = [80, 80],
    hues = [160, 340],
)

#polygonal cue parameters
base.update(
    poly_radius=.3,
    poly_linewidth=1,
    poly_color=-.2,
    )

#feedback parameters
base.update(
    fb_circle_radius = 1,
    fb_text_size = .75,
    fb_left_loc = (-2.5,0),
    fb_right_loc = (2.5,0),
    too_slow_color = '#992020',
    fb_pos_color = '#997020',#'#E2A429',
    fb_neg_color = '#992020',#'#AA0707',
    fb_text_color = '#999999'
    )

train = deepcopy(base)

train.update(

    run_type = 'train',
    monitor_name = 'mbpro',
    ntrials = 40,
    training_blocks = ['shape','motion','color']*7,
    coherence_update = {'motion': .02,
                'color':.01,
                'shape':.01},
    coherence = {'motion': .6,
                'color':.8,
                'shape':.8},
    num_correct_down = np.repeat([3,3,3,3,3,3,2],3), #how many correct in a row before it gets harder
    feedback_dur = .75,
    decision_dur = 2,
    instruct_text = {
    'intro':
                    ["""
                    Welcome to the experiment.
                    Today you will be making decisions based on
                    arrays of colored, moving dots.
                    You can advance through the instructions by pressing the '1' key.
                    """,
                    """
                    In this part of the experiment, you will practice responding
                    based on the color, shape motion of the dots.
                    """,
                    """
                    You have 2 seconds to respond on each trial.
                    Try to respond quickly, but without sacrificing accuracy.
                    The central cross will flicker if you answer incorrectly.
                    """,
                    """
                    Each rule is associated with a shape. Try to learn these 
                    shapes because they will be important later on.
                    """,
                    """
                    Keep your eyes focused on the central shape.
                    Do not move your eyes around to look at the dots.
                    """],
    'break_txt':    ["""
                    You have completed COMPLETED out of TOTAL trials.
                    Next you will respond based on
                    
                    FEATURE.
                    
                    Take a short break if you'd like.
                    Press space to continue.
                    """],
    'motion':       ["""
                    In the following trials,
                    indicate the direction that most 
                    of the dots are moving in. 
                    Press '1' for up and '2' for down. 
                    """],
    'shape':       ["""
                    In the following trials,
                    indicate the most common shape.
                    Press '1' for circle and '2' for crosses. 
                    """],
    'color':        ["""
                    In the following trials,
                    indicate the most common
                    color of the dots. 
                    Press '1' for green and '2' for pink. 
                    """]},
    instruct_color = '#7E2BE0'
)

# retrain = deepcopy(train)
#
# retrain.update(
#     run_type = 'retrain',
#     training_blocks = ['shape','motion','color'],
#     coherence_update = {'motion': .01,
#                 'color':.01,
#                 'shape':.01},
#     num_correct_down = np.repeat([1],3),
# )

psychophys = deepcopy(train)

psychophys.update(
    run_type = 'psychophys',
    training_blocks = ['shape','motion','color']*2,
    ntrials_init = 8,
    ntrials_test = 60
)

switch = deepcopy(base)

switch.update(

    run_type = 'switch',
    monitor_name = 'mbpro',
    num_blocks = 6,
    miniblock_ids = ['color_shape','shape_motion','motion_color'],
    num_block_reps = 2, #6 blocks total
    ntrials_per_miniblock = 16, #=96 trials per block
    instruct_text = {
    'intro':
                    [
                    """
                    In this part of the experiment, you will practice responding
                    based on the color, motion and direction of the dots.
                    """
                    """
                    On each trial, a central shape will indicate how you should respond:
                    """
                    """                    
                    Color: RULE1
                    Motion: RULE2
                    Shape: RULE3    
                    """
                    """                   
                    You have 2 seconds to respond on each trial.
                    Try to respond quickly, but without sacrificing accuracy.
                    Press space to begin.
                    """],
    'break_txt':    ["""
                    You have completed COMPLETED out of TOTAL blocks.
                    Take a short break if you'd like.
                    Press space to continue.
                    """]
                },
    instruct_color = '#7E2BE0'
)


switch_train = deepcopy(switch)

switch_train.update(
    run_type = 'switch_train',
    num_blocks = 2,
    coherence_floor = dict(color = .8,
                           motion = .8,
                           shape = .8),
)

reward = deepcopy(switch)

reward.update(
    run_type = 'reward',
    num_rew_blocks = 6,
    num_test_blocks = 2,
    fb_iti = .3,
    fb_dur = .5,
    p_rew_high = .85,
    p_rew_low = .1,
    break_dur = 6, #6 minute break
)

