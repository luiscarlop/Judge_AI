import time
import cv2
import numpy as np
from utils import find_angle, get_landmark_features, draw_text, draw_dotted_line


class FSM_squat:

    def __init__(self, thresholds, flip_frame=False, pre_trained=True):

        # Flip the frame or not.
        self.flip_frame = flip_frame

        # Thresholds for various angles and offsets.
        self.thresholds = thresholds

        # Font type.
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        # line type
        self.linetype = cv2.LINE_AA

        # set radius to draw arc
        self.radius = 20

        # Select model for pose estimation. True = Pre-trained model, False = Custom model.
        self.pre_trained = pre_trained

        # Colors in BGR format.
        self.COLORS = {
                        "blue": (255, 127, 0),
                        "red": (50, 50, 225),
                        "green": (127, 255, 0),
                        "orange": (0, 135, 255),
                        "light_green": (127, 233, 100),
                        "yellow": (0, 255, 255),
                        "magenta": (255, 0, 255),
                        "white": (255, 255, 255),
                        "cyan": (255, 255, 0),
                        "light_blue": (255, 204, 102),
                        "black": (0, 0, 0),  # añadido
        }

        # Dictionary to maintain the various landmark features.
        self.dict_features = {}
        self.left_features = {
                                "shoulder": 11,
                                "elbow": 13,
                                "wrist": 15,
                                "hip": 23,
                                "knee": 25,
                                "ankle": 27,
                                "foot": 31,
        }

        self.right_features = {
                                "shoulder": 12,
                                "elbow": 14,
                                "wrist": 16,
                                "hip": 24,
                                "knee": 26,
                                "ankle": 28,
                                "foot": 32,
        }

        self.dict_features["left"] = self.left_features
        self.dict_features["right"] = self.right_features
        self.dict_features["nose"] = 0

        # For tracking counters and sharing states in and out of callbacks.
        self.state_tracker = {
                                "state_seq": [],

                                # 'start_inactive_time': time.perf_counter(),
                                # 'start_inactive_time_front': time.perf_counter(),
                                # 'INACTIVE_TIME': 0.0,
                                # 'INACTIVE_TIME_FRONT': 0.0,

                                # 0 --> Bend Backwards, 1 --> Bend Forward, 2 --> Keep shin straight, 3 --> Deep squat
                                "DISPLAY_TEXT": np.full((4,), False),
                                "COUNT_FRAMES": np.zeros((4,), dtype=np.int64),

                                "VALID_SQUAT": False,
                                "LOWER_HIPS": False,
                                "NO LIFT": False,

                                "prev_state": None,
                                "curr_state": None,
                                
                                "SQUAT_COUNT": 0,
                                "IMPROPER_SQUAT": 0,
        }

        self.FEEDBACK_ID_MAP = {
                                0: ('BEND BACKWARDS', 215, (0, 153, 255)),
                                1: ('BEND FORWARD', 215, (0, 153, 255)),
                                2: ('KNEE FALLING OVER TOE', 170, (255, 80, 80)),
                                3: ('SQUAT TOO DEEP', 125, (255, 80, 80))
        }

        self.valid_squat = {
                            "state_seq": ['s1', 's2', 's3', 's2', 's1'],
        }
    
    def _get_state(self, knee_angle):
        """Get the state of the squat.


        Args:
            knee_angle (int): Angle between the line hip-knee and a vertical.

        Returns:
            _type_: _description_
        """

        knee = None

        if self.thresholds['HIP_KNEE_VERT']['STAND'][0] <= knee_angle <= self.thresholds['HIP_KNEE_VERT']['STAND'][1]:
            knee = 1
        elif self.thresholds['HIP_KNEE_VERT']['TRANS'][0] <= knee_angle <= self.thresholds['HIP_KNEE_VERT']['TRANS'][1]:
            knee = 2
        elif knee_angle >= self.thresholds['HIP_KNEE_VERT']['PASS']:
            knee = 3

        return f's{knee}' if knee else None         
    
    def _next_state(self, curr_state, knee_angle):

        next_state = None

        if curr_state == 's1':
            if knee_angle >= self.thresholds['HIP_KNEE_VERT']['STAND'][1]:
                next_state = 's2'
            else:
                next_state = 's1'

        elif curr_state == 's2':
            if knee_angle >= self.thresholds['HIP_KNEE_VERT']['TRANS'][1] and knee_angle >= self.thresholds['HIP_KNEE_VERT']['PASS']:
                next_state = 's3'
            else:
                next_state = 's2'
        elif curr_state == 's3':
            if knee_angle >= self.thresholds['HIP_KNEE_VERT']['PASS']:
                next_state = 's3'
            else:
                next_state = 's2'

        return next_state
    
    def _update_state_sequence(self, curr_state):

        """Update the state sequence.

        If the state is 's2' and 's3' is not in the sequence, then append 's2' to the sequence.
        If the state is 's3' and 's2' is in the sequence, then append 's3' to the sequence.
        """
        # if curr_state == 's1':
        #     if (('s1' not in self.state_tracker['state_seq']) and (len(self.state_tracker['state_seq']) == 0)) or \
        #         (('s1' in self.state_tracker['state_seq']) and (self.state_tracker['state_seq'][-1] == 's2')):
        #         self.state_tracker['state_seq'].append(curr_state)


        if curr_state == 's2':
            if (('s3' not in self.state_tracker['state_seq']) and (self.state_tracker['state_seq'].count('s2'))==0) or \
                    (('s3' in self.state_tracker['state_seq']) and (self.state_tracker['state_seq'].count('s2')==1)):
                        self.state_tracker['state_seq'].append(curr_state)
            

        elif curr_state == 's3':
            if (curr_state not in self.state_tracker['state_seq']) and 's2' in self.state_tracker['state_seq']: 
                self.state_tracker['state_seq'].append(curr_state)


    def _check_movement(self):
            
            """Check if the movement is correct or not.
    
            If the sequence of states is correct, then return True.
            """
            
            if self.state_tracker['state_seq'] == self.valid_squat['state_seq']:
                return True
            else:
                return False


    def _show_feedback(self, frame, c_frame, dict_maps, lift):
    
            """Show feedback to the user.
    
            If the movement is correct, then show 'GOOD LIFT'.
            If the movement is incorrect, then show 'NO REP'.
            """

            if lift:
                draw_text(
                            frame, 
                            "GOOD LIFT", 
                            pos=(50, 50), 
                            font=self.font, 
                            text_color=self.COLORS["white"], 
                            text_color_bg=self.COLORS["green"],
                )

            else:
                draw_text(
                            frame, 
                            "NO REP", 
                            pos=(50, 50), 
                            font=self.font, 
                            text_color=self.COLORS["white"], 
                            text_color_bg=self.COLORS["red"],
                )

            for idx in np.where(c_frame)[0]:
                draw_text(
                            frame,
                            dict_maps[idx][0],
                            pos=(30, dict_maps[idx][1]),
                            text_color=self.COLORS["black"],
                            font_scale=1,
                            text_color_bg=dict_maps[idx][2],
                )
            
            return frame

    
    def process(self, frame: np.array, pose):
        
        frame_height, frame_width, _ = frame.shape

        # Process the image
        keypoints = pose.process(frame)

        if keypoints.pose_landmarks:
            ps_lm = keypoints.pose_landmarks

            nose_coord = get_landmark_features(ps_lm.landmark, self.dict_features, 'nose', frame_width, frame_height)
            left_shldr_coord, left_elbow_coord, left_wrist_coord, left_hip_coord, left_knee_coord, left_ankle_coord, left_foot_coord = \
                                get_landmark_features(ps_lm.landmark, self.dict_features, 'left', frame_width, frame_height)
            right_shldr_coord, right_elbow_coord, right_wrist_coord, right_hip_coord, right_knee_coord, right_ankle_coord, right_foot_coord = \
                                get_landmark_features(ps_lm.landmark, self.dict_features, 'right', frame_width, frame_height)

            offset_angle = find_angle(left_shldr_coord, right_shldr_coord, nose_coord)
            # print(offset_angle,self.thresholds)

            # Camera is not aligned properly --> It is facing the user
            if offset_angle > self.thresholds['OFFSET_THRESH']:

                cv2.circle(frame, nose_coord, 7, self.COLORS['white'], -1)
                cv2.circle(frame, left_shldr_coord, 7, self.COLORS['yellow'], -1)
                cv2.circle(frame, right_shldr_coord, 7, self.COLORS['magenta'], -1)

                cv2.line(frame, left_shldr_coord, nose_coord, self.COLORS['light_blue'], 4, lineType=self.linetype)
                cv2.line(frame, right_shldr_coord, nose_coord, self.COLORS['light_blue'], 4, lineType=self.linetype)

                if self.flip_frame:
                    frame = cv2.flip(frame, 1)

                draw_text(
                    frame, 
                    'CAMERA NOT ALIGNED PROPERLY', 
                    pos=(30, frame_height-60),
                    text_color=self.COLORS["black"],
                    font_scale=1,
                    text_color_bg=self.COLORS["orange"],
                )

                draw_text(
                    frame, 
                    'OFFSET ANGLE: ' + str(offset_angle), 
                    pos=(30, frame_height-30),
                    text_color=self.COLORS["black"],
                    font_scale=1,
                    text_color_bg=self.COLORS["orange"],
                ) 

                self.state_tracker['prev_state'] =  None
                self.state_tracker['curr_state'] = None

                self.state_tracker['SQUAT_COUNT'] = 0
                self.state_tracker['IMPROPER_SQUAT'] = 0

            # Camera is aligned properly
            else: 
                
                # Calculate the distance between the shoulders and the hips.
                # This is used to determine if the user is standing right side or left side.
                dist_l_sh_hip = abs(left_foot_coord[1]- left_shldr_coord[1])
                dist_r_sh_hip = abs(right_foot_coord[1] - right_shldr_coord)[1]

                shldr_coord = None
                elbow_coord = None
                wrist_coord = None
                hip_coord = None
                knee_coord = None
                ankle_coord = None
                foot_coord = None

                if dist_l_sh_hip > dist_r_sh_hip:
                    shldr_coord = left_shldr_coord
                    elbow_coord = left_elbow_coord
                    wrist_coord = left_wrist_coord
                    hip_coord = left_hip_coord
                    knee_coord = left_knee_coord
                    ankle_coord = left_ankle_coord
                    foot_coord = left_foot_coord

                    multiplier = -1    # Left side

                else:
                    shldr_coord = right_shldr_coord
                    elbow_coord = right_elbow_coord
                    wrist_coord = right_wrist_coord
                    hip_coord = right_hip_coord
                    knee_coord = right_knee_coord
                    ankle_coord = right_ankle_coord
                    foot_coord = right_foot_coord

                    multiplier = 1    # Right side

                # ------------------- Vertical Angle calculation --------------
                
                hip_vertical_angle = find_angle(shldr_coord, np.array([hip_coord[0], 0]), hip_coord)
                cv2.ellipse(frame, hip_coord, (30, 30), 
                            angle = 0, startAngle = -90, endAngle = -90+multiplier*hip_vertical_angle, 
                            color = self.COLORS['white'], thickness = 3, lineType = self.linetype)

                draw_dotted_line(frame, hip_coord, start=hip_coord[1]-80, end=hip_coord[1]+20, line_color=self.COLORS['blue'])




                knee_vertical_angle = find_angle(hip_coord, np.array([knee_coord[0], 0]), knee_coord)
                cv2.ellipse(frame, knee_coord, (20, 20), 
                            angle = 0, startAngle = -90, endAngle = -90-multiplier*knee_vertical_angle, 
                            color = self.COLORS['white'], thickness = 3,  lineType = self.linetype)

                draw_dotted_line(frame, knee_coord, start=knee_coord[1]-50, end=knee_coord[1]+20, line_color=self.COLORS['blue'])



                ankle_vertical_angle = find_angle(knee_coord, np.array([ankle_coord[0], 0]), ankle_coord)
                cv2.ellipse(frame, ankle_coord, (30, 30),
                            angle = 0, startAngle = -90, endAngle = -90 + multiplier*ankle_vertical_angle,
                            color = self.COLORS['white'], thickness = 3,  lineType=self.linetype)

                draw_dotted_line(frame, ankle_coord, start=ankle_coord[1]-50, end=ankle_coord[1]+20, line_color=self.COLORS['blue'])

                # ------------------------------------------------------------

                # Join landmarks.
                cv2.line(frame, shldr_coord, elbow_coord, self.COLORS['light_blue'], 4, lineType=self.linetype)
                cv2.line(frame, wrist_coord, elbow_coord, self.COLORS['light_blue'], 4, lineType=self.linetype)
                cv2.line(frame, shldr_coord, hip_coord, self.COLORS['light_blue'], 4, lineType=self.linetype)
                cv2.line(frame, knee_coord, hip_coord, self.COLORS['light_blue'], 4,  lineType=self.linetype)
                cv2.line(frame, ankle_coord, knee_coord,self.COLORS['light_blue'], 4,  lineType=self.linetype)
                cv2.line(frame, ankle_coord, foot_coord, self.COLORS['light_blue'], 4,  lineType=self.linetype)
                
                # Plot landmark points
                cv2.circle(frame, shldr_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, elbow_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, wrist_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, hip_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, knee_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, ankle_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, foot_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)

                current_state = self._get_state(knee_vertical_angle)
                self.state_tracker['curr_state'] = current_state
                self._update_state_sequence(current_state)


                # -------------------------------------- COMPUTE COUNTERS --------------------------------------

                if current_state == 's1':
                    
                    # lift = self._check_movement()

                    if len(self.state_tracker['state_seq']) == 3 and not self.state_tracker['NO LIFT']:
                        if hip_vertical_angle < 10:
                            self.state_tracker['SQUAT_COUNT']+=1
                            self.state_tracker['VALID_SQUAT'] = True
                        else:
                            self.state_tracker['IMPROPER_SQUAT']+=1
                            self.state_tracker['VALID_SQUAT'] = False

                        
                    elif 's2' in self.state_tracker['state_seq'] and len(self.state_tracker['state_seq']) == 1:
                        self.state_tracker['IMPROPER_SQUAT']+=1
                        self.state_tracker['VALID_SQUAT'] = False
                        
                    
                    self.state_tracker['state_seq'] = []


                # ----------------------------------------------------------------------------------------------------

                else:
                    if self.thresholds['KNEE_THRESH'][0] < knee_vertical_angle < self.thresholds['KNEE_THRESH'][1] and \
                        self.state_tracker['state_seq'].count('s2')==1:
                        
                        self.state_tracker['NO LIFT'] = True

                    elif self.thresholds['KNEE_THRESH'][2] < knee_vertical_angle:
                        self.state_tracker['NO LIFT'] = False

                #- ---------------------------------------------------------------------------------------------------

                hip_text_coord_x = hip_coord[0] + 10
                knee_text_coord_x = knee_coord[0] + 15
                ankle_text_coord_x = ankle_coord[0] + 10

                if self.flip_frame:
                    frame = cv2.flip(frame, 1)
                    hip_text_coord_x = frame_width - hip_coord[0] + 10
                    knee_text_coord_x = frame_width - knee_coord[0] + 15
                    ankle_text_coord_x = frame_width - ankle_coord[0] + 10

                if 's3' in self.state_tracker['state_seq'] or current_state == 's1':
                    self.state_tracker['LOWER_HIPS'] = False

                self.state_tracker['COUNT_FRAMES'][self.state_tracker['DISPLAY_TEXT']]+=1

                # if self.state_tracker['SQUAT_COUNT'] > 0:
                    
                frame = self._show_feedback(frame, self.state_tracker['COUNT_FRAMES'], self.FEEDBACK_ID_MAP, self.state_tracker['VALID_SQUAT'])



                cv2.putText(frame, str(int(hip_vertical_angle)), (hip_text_coord_x, hip_coord[1]), self.font, 0.6, self.COLORS['light_green'], 2, lineType=self.linetype)
                cv2.putText(frame, str(int(knee_vertical_angle)), (knee_text_coord_x, knee_coord[1]+10), self.font, 0.6, self.COLORS['light_green'], 2, lineType=self.linetype)
                cv2.putText(frame, str(int(ankle_vertical_angle)), (ankle_text_coord_x, ankle_coord[1]), self.font, 0.6, self.COLORS['light_green'], 2, lineType=self.linetype)

            
                draw_text(
                    frame, 
                    "CORRECT: " + str(self.state_tracker['SQUAT_COUNT']), 
                    pos=(int(frame_width*0.68), 30),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=self.COLORS['light_green']
                )  
                

                draw_text(
                    frame, 
                    "INCORRECT: " + str(self.state_tracker['IMPROPER_SQUAT']), 
                    pos=(int(frame_width*0.68), 80),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=self.COLORS['red']
                    
                )  
                
                
                self.state_tracker['DISPLAY_TEXT'][self.state_tracker['COUNT_FRAMES'] > self.thresholds['CNT_FRAME_THRESH']] = False
                self.state_tracker['COUNT_FRAMES'][self.state_tracker['COUNT_FRAMES'] > self.thresholds['CNT_FRAME_THRESH']] = 0    
                self.state_tracker['prev_state'] = current_state

        
        else:

            # if self.flip_frame:
            #     frame = cv2.flip(frame, 1)

            # end_time = time.perf_counter()
            # self.state_tracker['INACTIVE_TIME'] += end_time - self.state_tracker['start_inactive_time']

            # display_inactivity = False

            # if self.state_tracker['INACTIVE_TIME'] >= self.thresholds['INACTIVE_THRESH']:
            self.state_tracker['SQUAT_COUNT'] = 0
            self.state_tracker['IMPROPER_SQUAT'] = 0
                # cv2.putText(frame, 'Resetting SQUAT_COUNT due to inactivity!!!', (10, frame_height - 25), self.font, 0.7, self.COLORS['blue'], 2)
                # display_inactivity = True

            # self.state_tracker['start_inactive_time'] = end_time

            draw_text(
                    frame, 
                    "CORRECT: " + str(self.state_tracker['SQUAT_COUNT']), 
                    pos=(int(frame_width*0.68), 30),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=self.COLORS['light_green']
                )  
                

            draw_text(
                    frame, 
                    "INCORRECT: " + str(self.state_tracker['IMPROPER_SQUAT']), 
                    pos=(int(frame_width*0.68), 80),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=self.COLORS['red']
                    
                )  
            
            
            # Reset all other state variables
            
            self.state_tracker['prev_state'] =  None
            self.state_tracker['curr_state'] = None
            # self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0
            self.state_tracker['INCORRECT_POSTURE'] = False
            self.state_tracker['DISPLAY_TEXT'] = np.full((5,), False)
            self.state_tracker['COUNT_FRAMES'] = np.zeros((5,), dtype=np.int64)
            
            
        
        return frame

                    
