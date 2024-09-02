import carla
import cv2
import math
import numpy as np
from pygame.time import Clock
from ..utils import get_actor_display_name
from typing import List, Tuple, Optional


class CarlaEnvRender:
    # I am well aware that Carla PythonAPI offers the HUD class to handle rendering with pygames, but honestly
    # I felt like this was a good opportunity to get in touch with opencv (cv2) once again, so I wrote a bunch 
    # of the HUD features I needed for rendering with opencv and its awesome, so yeah...
    def __init__(self, world: carla.World, world_scale: Optional[float]=None):
        world_scale = world_scale or 1.0
        self._world = world
        self._world_map = self._world.get_map()
        self._world.on_tick(self.on_world_tick)
        self._client_clock = Clock()
        self._server_clock = Clock()
        self.world_scale = world_scale

        self.main_width = int(800 * world_scale)
        self.main_height = int(600 * world_scale)
        self.side_panel_width = int(230 * world_scale)

        self.side_panel_color = (10, 10, 10)
        self.text_color = (255, 255, 255)

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5 * world_scale
        self.thickness = int(1 * world_scale)
        self.alpha = 0.9

        self.progress_bar_height = int(5 * world_scale)
        self.progress_bar_width = self.side_panel_width - int(40 * world_scale)  # Margin of 20 on each side
        self.progress_color = (200, 255, 180)                                    # Green color for progress bar

        self.throttle_bar_x = self.main_width + int(20 * world_scale)            # X position of throttle bar
        self.throttle_bar_y = int(100 * world_scale)                             # Y position of throttle bar

        self.brake_bar_x = self.main_width + int(20 * world_scale)               # X position of brake bar
        self.brake_bar_y = int(150 * world_scale)                                # Y position of brake bar

        self.steer_bar_x = self.main_width + int(20 * world_scale)               # X position of steer bar
        self.steer_bar_y = int(200 * world_scale)                                # Y position of steer bar


    def on_world_tick(self, timestamp: int):
        self._server_clock.tick()
        self._server_fps = self._server_clock.get_fps()


    def wrap_long_text(
            self, 
            text: int, 
            font: str, 
            font_scale: float, 
            thickness: int, 
            max_width: int) -> List[str]:
        
        # Split the text into words
        words = text.split(' ')
        lines = []
        current_line = ""
        for word in words:
            test_line = current_line + word + " "
            text_size = cv2.getTextSize(
                test_line, font, font_scale, thickness)[0]
            
            if text_size[0] <= max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word + " "
        if current_line:
            lines.append(current_line)
        return lines


    def draw_side_panel(
            self, img: np.ndarray, 
            actor: carla.Actor, 
            display_text: List[Tuple[str, Tuple[int, int]]],
            terminal_reason: Optional[str]=None
        ):
        c = actor.get_control()
        throttle = c.throttle
        brake = c.brake
        steer = c.steer

        # Draw the side panel background
        overlay = img.copy()
        cv2.rectangle(
            overlay, 
            (self.main_width, 0), 
            (self.main_width + self.side_panel_width, self.main_height), 
            self.side_panel_color, -1)

        # Blend the overlay with the original image
        cv2.addWeighted(overlay, self.alpha, img, 1 - self.alpha, 0, img)

        # Draw the progress bar filled area (for throttle control)
        cv2.putText(
            img, "Throttle", 
            (self.throttle_bar_x, self.throttle_bar_y-int(15 * self.world_scale)), 
            self.font, self.font_scale, self.text_color, self.thickness
        )
        cv2.rectangle(img, (self.throttle_bar_x, self.throttle_bar_y), 
                    (self.throttle_bar_x + self.progress_bar_width, 
                     self.throttle_bar_y + self.progress_bar_height), 
                    (100, 100, 100), -1)
        filled_width = int(self.progress_bar_width * throttle)
        cv2.rectangle(img, (self.throttle_bar_x, self.throttle_bar_y), 
                    (self.throttle_bar_x + filled_width, 
                     self.throttle_bar_y + self.progress_bar_height), 
                    self.progress_color, -1)

        # Draw the progress bar filled area (for brake control)
        cv2.putText(
            img, "Brake", 
            (self.brake_bar_x, self.brake_bar_y-int(15 * self.world_scale)), 
            self.font, self.font_scale, self.text_color, self.thickness
        )
        cv2.rectangle(img, (self.brake_bar_x, self.brake_bar_y), 
                    (self.brake_bar_x + self.progress_bar_width, 
                     self.brake_bar_y + self.progress_bar_height), 
                    (100, 100, 100), -1)
        filled_width = int(self.progress_bar_width * brake)
        cv2.rectangle(img, (self.brake_bar_x, self.brake_bar_y), 
                    (self.brake_bar_x + filled_width, 
                     self.brake_bar_y + self.progress_bar_height), 
                    self.progress_color, -1)
        
        # Draw the progress bar filled area (for steer control)
        cv2.putText(
            img, "Steer", 
            (self.steer_bar_x, self.steer_bar_y-int(15 * self.world_scale)), 
            self.font, self.font_scale, self.text_color, self.thickness
        )
        cv2.rectangle(img, (self.steer_bar_x, self.steer_bar_y), 
                    (self.steer_bar_x + self.progress_bar_width, 
                     self.steer_bar_y + self.progress_bar_height), 
                    (100, 100, 100), -1)
        
        w = self.progress_bar_width // 2
        steer_slider_x1 = self.steer_bar_x + (w + round((steer * w))) - 2
        steer_slider_x2 = steer_slider_x1 + 4
        cv2.rectangle(img, (steer_slider_x1, self.steer_bar_y), 
                    (steer_slider_x2, self.steer_bar_y + self.progress_bar_height), 
                    self.progress_color, -1)
        
        for line, pos in display_text:
            cv2.putText(img, line, pos, self.font, self.font_scale, self.text_color, self.thickness)

        if terminal_reason and terminal_reason != "none":
            text = f"terminal reason: {terminal_reason}"
            lines = self.wrap_long_text(
                text, self.font, self.font_scale, self.thickness, 
                max_width=self.side_panel_width-int(20 * self.world_scale))
            
            thickness = self.thickness
            font_scale = self.font_scale
            y_offset = pos[1] + int(40 * self.world_scale)
            for line in lines:
                cv2.putText(
                    img, line, 
                    (self.main_width + int(20 * self.world_scale), y_offset),
                    self.font, font_scale, (200, 240, 100), thickness)
                y_offset += int(cv2.getTextSize(
                    line, self.font, font_scale, thickness)[0][1] * 1.5 * self.world_scale)


    def get_display_text(self, actor: carla.Actor) -> List[str]:
        t = actor.get_transform()
        v = actor.get_velocity()
        heading = 'N' if abs(t.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(t.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > t.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > t.rotation.yaw > -179.5 else ''
        vehicles = self._world.get_actors().filter('vehicle.*')
        x_start = self.main_width+int(20 * self.world_scale)
        y_start = int(30 * self.world_scale)
        y_space = int(20 * self.world_scale)
        info_text = [
            ('Server:  % 1.0f FPS' % self._server_fps, (x_start, y_start)),
            ('Client:  % 1.0f FPS' % self._client_clock.get_fps(), (x_start, y_start+y_space)),
            ('Vehicle: % 2s' % get_actor_display_name(actor, truncate=20), (x_start, y_start+y_space*10)),
            ('Map:     % 2s' % self._world_map.name, (x_start, y_start+y_space*12)),
            ('Speed:   % 1.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)), (x_start, y_start+y_space*14)),
            (u'Heading:% 1.0fdeg % 2s' % (t.rotation.yaw, heading), (x_start, y_start+y_space*16)),
            ('Pos:% 2s' % ('(% 2.1f, % 2.1f)' % (t.location.x, t.location.y)), (x_start, y_start+y_space*18)),
            ('Height:  % 1.0f m' % t.location.z, (x_start, y_start+y_space*20)),
        ]
        info_text += [
            ('Num vehicles: % 1d' % len(vehicles), (x_start, y_start+y_space*22))
        ]
        return info_text


    def render(
            self, 
            actor: carla.Actor, 
            spectator_img: np.ndarray, 
            obs_img: np.ndarray, 
            terminal_reason: Optional[str]=None
        ):
        img = spectator_img.copy()
        obs_img = obs_img.copy()

        spectator_h, spectator_w = img.shape[:2]
        new_obs_h, new_obs_w = spectator_h//4, spectator_w//4
        obs_img = cv2.resize(obs_img, dsize=(new_obs_w, new_obs_h))
        offset_w = 10
        offset_h = 10
        img[offset_h:new_obs_h+offset_h, offset_w:new_obs_w+offset_w, :] = obs_img

        self._client_clock.tick()
        display_text = self.get_display_text(actor)
        self.draw_side_panel(img, actor, display_text, terminal_reason)
        cv2.imshow("CarlaEnv", img)
        cv2.waitKey(1)


    def close(self):
        cv2.destroyWindow("CarlaEnv")