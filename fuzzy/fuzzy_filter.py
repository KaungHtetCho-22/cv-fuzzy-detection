import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Define fuzzy variables
confidence = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'confidence')
iou = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'iou')
trust = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'trust')

# Membership functions
confidence.automf(3)
iou.automf(3)
trust['low'] = fuzz.trimf(trust.universe, [0, 0, 0.5])
trust['medium'] = fuzz.trimf(trust.universe, [0.2, 0.5, 0.8])
trust['high'] = fuzz.trimf(trust.universe, [0.5, 1, 1])

# Rules
rule1 = ctrl.Rule(confidence['poor'] | iou['poor'], trust['low'])
rule2 = ctrl.Rule(confidence['average'] & iou['average'], trust['medium'])
rule3 = ctrl.Rule(confidence['good'] & iou['good'], trust['high'])

# Control system
trust_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
trust_sim = ctrl.ControlSystemSimulation(trust_ctrl)

def filter_detection(conf, iou_score):
    trust_sim.input['confidence'] = conf
    trust_sim.input['iou'] = iou_score
    trust_sim.compute()
    return trust_sim.output['trust']
