import random

import gymnasium as gym

from configs.config import ENV_CONFIG, TRAINING_CONFIG


class WeatherFrictionWrapper(gym.Wrapper):
    """
    Applies configurable low-friction road tiles to CarRacing-v3 after each reset.

    Gymnasium's CarRacing-v3 stores the generated centerline in ``track`` and
    the physical Box2D road tile bodies in ``road``. The car dynamics use each
    contacted tile's ``road_friction`` attribute, so this wrapper modifies that
    value directly and reports whether the car is touching any modified tile.
    """

    def __init__(self, env):
        super().__init__(env)
        self.weather_mode = TRAINING_CONFIG.get("weather_mode", "rain")
        self.slick_tile_probability = self._clamp_probability(
            TRAINING_CONFIG.get("slick_tile_probability", 0.30)
        )
        self.slick_friction_value = float(
            TRAINING_CONFIG.get("slick_friction_value", 0.15)
        )
        self.default_friction_value = 1.0
        self.slick_tiles = set()

    @staticmethod
    def _clamp_probability(value):
        try:
            probability = float(value)
        except (TypeError, ValueError):
            probability = 0.30
        return max(0.0, min(1.0, probability))

    @property
    def friction_enabled(self):
        configured_mode = TRAINING_CONFIG.get(
            "weather_mode",
            ENV_CONFIG.get("weather_mode", self.weather_mode),
        )
        return str(configured_mode).lower() != "dry"

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._apply_weather_friction()
        info = dict(info)
        info["weather_mode"] = TRAINING_CONFIG.get("weather_mode", self.weather_mode)
        info["slick_tile_count"] = len(self.slick_tiles)
        info["is_on_slick_surface"] = self._is_on_slick_surface()
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["weather_mode"] = TRAINING_CONFIG.get("weather_mode", self.weather_mode)
        info["slick_tile_count"] = len(self.slick_tiles)
        info["is_on_slick_surface"] = self._is_on_slick_surface()
        return observation, reward, terminated, truncated, info

    def _apply_weather_friction(self):
        self.slick_tiles.clear()

        road_tiles = getattr(self.env.unwrapped, "road", None)
        if not road_tiles:
            return

        for tile in road_tiles:
            self._set_tile_friction(tile, self.default_friction_value)
            setattr(tile, "is_slick_surface", False)

        if not self.friction_enabled:
            return

        for tile in road_tiles:
            if random.random() <= self.slick_tile_probability:
                self._set_tile_friction(tile, self.slick_friction_value)
                setattr(tile, "is_slick_surface", True)
                self.slick_tiles.add(tile)

    def _set_tile_friction(self, tile, friction_value):
        setattr(tile, "road_friction", friction_value)
        for fixture in getattr(tile, "fixtures", []):
            try:
                fixture.friction = friction_value
            except Exception:
                pass

    def _is_on_slick_surface(self):
        car = getattr(self.env.unwrapped, "car", None)
        if car is None:
            return False

        for wheel in getattr(car, "wheels", []):
            for tile in getattr(wheel, "tiles", set()):
                if tile in self.slick_tiles or getattr(tile, "is_slick_surface", False):
                    return True

        hull = getattr(car, "hull", None)
        for tile in getattr(hull, "tiles", set()):
            if tile in self.slick_tiles or getattr(tile, "is_slick_surface", False):
                return True

        return False
