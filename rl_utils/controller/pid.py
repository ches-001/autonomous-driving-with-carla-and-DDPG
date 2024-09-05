import numpy as np
from collections import deque

class _BasePIDController:
    def __init__(
            self, 
            kp: float=1.0,
            ki: float=1.0,
            kd: float=1.0,
            dt: float=0.03, 
            integral_horizon: int=50):
        
        self.kp = kp; self.ki = ki; self.kd = kd; self.dt = dt       
        self.errors = deque(maxlen=integral_horizon)

    def proportional(self) -> float:
        if len(self.errors) == 0:
            return 0.0
        return self.kp * self.errors[-1]
    
    def integral(self) -> float:
        if len(self.errors) < 2:
            return 0.0
        ie = np.sum(self.errors) * self.dt
        return self.ki * ie
    
    def derivative(self) -> float:
        if len(self.errors) < 2:
            return 0.0
        de = (self.errors[-1] - self.errors[-2]) / self.dt
        return self.kd * de
    
    def pid_control(self, e: float) -> float:
        raise NotImplementedError
    

class LongitudinalPIDController(_BasePIDController):
    
    def __init__(self, *args, **kwargs):
        super(LongitudinalPIDController, self).__init__(*args, **kwargs)

    def compute_error(self, velocity: np.ndarray, setpoint_speed: float) -> float:
        speed = 3.6 * np.sqrt((velocity**2).sum())
        e = (setpoint_speed - speed)
        self.errors.append(e)
        return e
    
    def pid_control(self) -> float:
        p = self.proportional()
        i = self.integral()
        d = self.derivative()
        u =  p + i + d
        u = np.clip(u, 0.0, 1.0)
        return u
    

class LateralPIDController(_BasePIDController):
    
    def __init__(self, *args, **kwargs):
        super(LateralPIDController, self).__init__(*args, **kwargs)

    def compute_error(self, state: np.ndarray, rotation: np.ndarray, setpoint: np.ndarray) -> float:
        v_begin = state
        # rotation[1] is yaw (in degrees)
        v_end = v_begin + np.asarray(
            [np.cos(np.deg2rad(rotation[1])), np.sin(np.deg2rad(rotation[1])), 0.0]
        )
        v_vec = np.array([v_end[0]-v_begin[0], v_end[1]-v_begin[1], 0.0])
        w_vec = np.array([setpoint[0]-v_begin[0], setpoint[1]-v_begin[1], 0.0])
        dot = np.dot(w_vec, v_vec) / (np.linalg.norm(w_vec) * np.linalg.norm(v_vec))
        dot = np.arccos(np.clip(dot, -1.0, 1.0))
        cross = np.cross(v_vec, w_vec)
        if cross[2] < 0:
            dot *= -1.0
        self.errors.append(dot)
        return dot
    
    def pid_control(self) -> float:
        p = self.proportional()
        i = self.integral() * self.dt
        d = self.derivative() / self.dt
        u =  p + i + d
        u = np.clip(u, -1.0, 1.0)
        return u