import numpy as np

symmetry_dict = {
    # identity: non trivial axis
    # to not have dev/null error
    0: ((1,0,0), 0.),
    # faces
    1: ((1,0,0), 90.),
    2: ((1,0,0), 180.),
    3: ((1,0,0), 270.),
    4: ((0,1,0), 90.),
    5: ((0,1,0), 180.),
    6: ((0,1,0), 270.),
    7: ((0,0,1), 90.),
    8: ((0,0,1), 180.),
    9: ((0,0,1), 270.),
    # edges
    10: ((0,1,1), 180.),
    11: ((1,0,1), 180.),
    12: ((1,1,0), 180.),
    13: ((0,-1,1), 180.),
    14: ((-1,0,1), 180.),
    15: ((-1,1,0), 180.), 
    # diagonals
    16: ((1,1,1), 120.),
    17: ((1,1,1), 240.),
    18: ((1,1,-1), 120.),
    19: ((1,1,-1), 240.),
    20: ((1,-1,1), 120.),
    21: ((1,-1,1), 240.),
    22: ((-1,1,1), 120.),
    23: ((-1,1,1), 240.),
}

def get_isomorphism_axes_angle(rng, batch_size):
    indices = rng.randint(low=0, high=24, size=(batch_size))

    # look up rotations
    axes = np.array([symmetry_dict[x][0] for x in indices], dtype=np.float32)
    angles = np.array([symmetry_dict[x][1] for x in indices], dtype=np.float32)

    return axes, angles
        
