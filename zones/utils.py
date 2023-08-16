import numpy as np


def get_named_goal_vector():

    J = np.array([[0.93746383, 0.7257721 , 0.51970005, 0.55108805, 0.30759908,
        0.66486559, 0.24314127, 0.22265362, 0.12839532, 0.38265358,
        0.83084475, 0.20381052, 0.54570852, 0.97652514, 0.61484882,
        0.90669365, 0.36273377, 0.41514413, 0.69766517, 0.97104452,
        0.44642987, 0.68578095, 0.96322573, 0.34602538]])
    W = np.array([[0.04658744, 0.6132297 , 0.17750154, 0.7470412 , 0.79464227,
        0.34748171, 0.91121304, 0.45157131, 0.76549351, 0.97685069,
        0.1016588 , 0.5950235 , 0.41211833, 0.99821283, 0.46381131,
        0.14241008, 0.78194727, 0.2558423 , 0.63867183, 0.48801561,
        0.27415259, 0.07493459, 0.04021906, 0.86341445]])
    R = np.array([[0.56206119, 0.41351551, 0.06089021, 0.49175584, 0.37151972,
        0.78651471, 0.27800672, 0.6413668 , 0.90253003, 0.20905838,
        0.3364942 , 0.97637833, 0.78422454, 0.1374359 , 0.83589714,
        0.24933559, 0.32258336, 0.77494119, 0.19660498, 0.21618076,
        0.04049623, 0.74389934, 0.61721893, 0.07773024]])
    Y = np.array([[7.49026470e-04, 9.17719119e-01, 9.21465030e-01, 7.14922838e-01,
        5.21557865e-01, 5.71643425e-01, 6.04791609e-01, 7.90493410e-01,
        2.55587239e-01, 6.30061807e-01, 9.84225246e-01, 9.47544655e-01,
        8.30635599e-01, 5.48330893e-01, 2.76120757e-01, 4.76113220e-01,
        2.94327675e-01, 3.57143143e-01, 7.18415244e-01, 2.00506118e-01,
        3.12302434e-01, 5.29668800e-01, 7.45683743e-01, 4.74814309e-01]])

    return {'J': J, 'W': W, 'R': R, 'Y': Y}
