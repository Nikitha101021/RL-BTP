import numpy as np


class PhysicsActionShield:
    def __init__(
        self,
        slick_speed_threshold=30.0,
        throttle_scale=0.20,
        brake_floor=0.40,
        steering_threshold=0.50,
        steering_scale=0.45,
        angular_velocity_threshold=2.50,
        spin_steering_scale=0.25,
    ):
        self.slick_speed_threshold = float(slick_speed_threshold)
        self.throttle_scale = float(throttle_scale)
        self.brake_floor = float(brake_floor)
        self.steering_threshold = float(steering_threshold)
        self.steering_scale = float(steering_scale)
        self.angular_velocity_threshold = float(angular_velocity_threshold)
        self.spin_steering_scale = float(spin_steering_scale)

    def apply(self, action, telemetry):
        safe_action = np.asarray(action, dtype=np.float32).copy()
        original_shape = safe_action.shape

        if safe_action.ndim == 2:
            flat_action = safe_action[0]
        else:
            flat_action = safe_action

        if flat_action.shape[0] < 3:
            return safe_action.reshape(original_shape)

        steering = float(flat_action[0])
        gas = float(flat_action[1])
        brake = float(flat_action[2])

        is_on_slick_surface = bool(telemetry.get("is_on_slick_surface", False))
        speed = float(telemetry.get("speed", telemetry.get("true_speed", 0.0)))
        angular_velocity = abs(float(telemetry.get("angular_velocity", 0.0)))

        if is_on_slick_surface:
            if speed > self.slick_speed_threshold:
                gas *= self.throttle_scale
                brake = max(brake, self.brake_floor)

            if abs(steering) > self.steering_threshold:
                steering *= self.steering_scale

            if angular_velocity > self.angular_velocity_threshold:
                steering *= self.spin_steering_scale
                gas *= self.throttle_scale
                brake = max(brake, self.brake_floor)

        flat_action[0] = np.clip(steering, -1.0, 1.0)
        flat_action[1] = np.clip(gas, 0.0, 1.0)
        flat_action[2] = np.clip(brake, 0.0, 1.0)

        return safe_action.reshape(original_shape)


def apply_physics_shield(action, telemetry_dict):
    return PhysicsActionShield().apply(action, telemetry_dict)
