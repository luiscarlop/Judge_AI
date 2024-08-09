MOVES_DICT = {
                1: 'SQUAT',
                2: 'BENCH PRESS',
                3: 'DEADLIFT',
}


# Get thresholds for beginner mode
def get_thresholds(move: int = 1):

    if move not in MOVES_DICT.keys():
        raise ValueError(f"Invalid move {move}. Choose from {MOVES_DICT.keys()}")
    
    if move == 1:

        _ANGLE_HIP_KNEE_VERT = {
                                'STAND' : (0,  10),
                                'TRANS'  : (15, 75),
                                'PASS'   : 80
                            }    

            
        thresholds = {
                        'HIP_KNEE_VERT': _ANGLE_HIP_KNEE_VERT,

                        'KNEE_THRESH'  : [15, 75, 80],

                        'OFFSET_THRESH'    : 70.0,
                        # 'INACTIVE_THRESH'  : 15.0,

                        'CNT_FRAME_THRESH' : 50
                                
                    }
        
    elif move == 2:
        
        raise NotImplementedError("Bench press thresholds not implemented yet")

    elif move == 3:
        _ANGLE_KNEE_HIP_VERT = {
                                'START' : 20,
                                'TRANS' : (0, 20),
                                'EXTEND'   : -5
                            }
        
        _ANGLE_HIP_KNEE_VERT = {
                                'START' : (15,  50),
                                'TRANS'  : (8, 10),
                                'EXTEND'   : 5
                            }      

            
        thresholds = {
                        'HIP_KNEE_VERT': _ANGLE_KNEE_HIP_VERT,
                        'KNEE_HIP_VERT': _ANGLE_HIP_KNEE_VERT,

                        'HIP_THRESH'  : [-5, 20],
                        'KNEE_THRESH'  : [5, 10, 20],

                        'OFFSET_THRESH'    : 70.0,
                        # 'INACTIVE_THRESH'  : 15.0,

                        'CNT_FRAME_THRESH' : 50
                                
                    }
                
    return thresholds
