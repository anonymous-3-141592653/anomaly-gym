import numpy as np
import pygame

from .world import World


def get_angle(x1, x2=(0, 0)) -> float:
    return np.degrees(np.arctan2(x2[1] - x1[1], x2[0] - x1[0]))


def build_triangle(center_x, center_y, scale, rotation) -> list[tuple[float, float]]:
    # define the points in a uint space
    p1 = (-0.5, 0)
    p2 = (0.5, 0.5)
    p3 = (0.5, -0.5)

    # rotate
    ra = np.radians(rotation)
    rp1x = p1[0] * np.cos(ra) - p1[1] * np.sin(ra)
    rp1y = p1[0] * np.sin(ra) + p1[1] * np.cos(ra)
    rp2x = p2[0] * np.cos(ra) - p2[1] * np.sin(ra)
    rp2y = p2[0] * np.sin(ra) + p2[1] * np.cos(ra)
    rp3x = p3[0] * np.cos(ra) - p3[1] * np.sin(ra)
    rp3y = p3[0] * np.sin(ra) + p3[1] * np.cos(ra)
    rp1 = (rp1x, rp1y)
    rp2 = (rp2x, rp2y)
    rp3 = (rp3x, rp3y)

    # scale
    sp1 = [rp1[0] * scale, rp1[1] * scale]
    sp2 = [rp2[0] * scale, rp2[1] * scale]
    sp3 = [rp3[0] * scale, rp3[1] * scale]

    # offset
    sp1[0] += center_x
    sp1[1] += center_y
    sp2[0] += center_x
    sp2[1] += center_y
    sp3[0] += center_x
    sp3[1] += center_y

    return [tuple(sp1), tuple(sp2), tuple(sp3)]  # type:ignore


def translate_point(p: tuple[float, float], width: float, height: float, cam_range: float) -> list[float]:
    x, y = p
    y *= -1  # this makes the display mimic the old pyglet setup (ie. flips image)
    x = (x * width * 0.5) / cam_range
    y = (y * height * 0.5) / cam_range
    x += width // 2 + (cam_range - 1)
    y += height // 2 + (cam_range - 1)
    return [x, y]


class PygameRenderer:
    def __init__(self, width, height, cam_range=1.5) -> None:
        pygame.init()
        self.width = width
        self.height = height
        self.cam_range = cam_range
        self.game_font = pygame.font.Font(None, self.width // 30)
        self.screen = pygame.Surface([self.width, self.height])

    def render(self, world: World, render_mode: str | None = "rgb_array") -> np.ndarray:
        self.draw_world(world)

        if render_mode == "rgb_array":
            observation = np.array(pygame.surfarray.pixels3d(self.screen))
            return np.transpose(observation, axes=(1, 0, 2))

        elif render_mode == "human":
            raise NotImplementedError

        else:
            raise NotImplementedError

    def draw_world(self, world: World) -> None:
        # clear screen
        self.screen.fill((255, 255, 255))

        for e, entity in enumerate(world.entities):
            x, y = entity.state.p_pos

            if abs(x) > self.cam_range or abs(y) > self.cam_range:
                angle = get_angle(x1=(x, y))
                px = np.clip(x, -self.cam_range * 0.9, self.cam_range * 0.9)
                py = np.clip(y, -self.cam_range * 0.9, self.cam_range * 0.9)
                tri_points = build_triangle(center_x=px, center_y=py, scale=0.1, rotation=angle)
                tri_points = [
                    translate_point(_p, width=self.width, height=self.height, cam_range=self.cam_range)
                    for _p in tri_points
                ]
                pygame.draw.polygon(self.screen, entity.color, tri_points)

            else:
                px, py = translate_point((x, y), width=self.width, height=self.height, cam_range=self.cam_range)
                pygame.draw.circle(
                    self.screen, entity.color, (px, py), (entity.size * self.width * 0.5) / self.cam_range
                )
                text = self.game_font.render(f"{entity.symbol}_{e}", True, "#FFFFFF")
                self.screen.blit(text, text.get_rect(center=(px, py)))

        text = self.game_font.render(str(world.elapsed_simulation_steps), True, "#000000")
        px, py = translate_point((1.2, 1.2), width=self.width, height=self.height, cam_range=self.cam_range)
        self.screen.blit(text, text.get_rect(center=(px, py)))
