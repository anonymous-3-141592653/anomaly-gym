import carla
import numpy as np


def vec2arr(loc: carla.Location) -> np.ndarray:
    return np.array([loc.x, loc.y, loc.z])


def get_rel_angle(target_transform, reference_transform):
    _vec = reference_transform.get_forward_vector()
    ref_vec = np.array([_vec.x, _vec.y])
    target_vec = np.array(
        [
            target_transform.location.x - reference_transform.location.x,
            target_transform.location.y - reference_transform.location.y,
        ]
    )
    norm_target = np.linalg.norm(target_vec)
    angle = np.degrees(np.arccos(np.clip(np.dot(ref_vec, target_vec) / norm_target, -1.0, 1.0)))

    return angle


def unit_vector(vector):
    """Returns the unit vector of the vector."""
    return vector / (np.linalg.norm(vector) + 1e-8)


def angle_between(v1, v2, degrees=False):
    angle = np.arccos(np.clip(np.dot(unit_vector(v1), unit_vector(v2)), -1.0, 1.0))
    return np.degrees(angle) if degrees else angle


def shortest_distance_to_line(a_point, line_point, line_vec):
    v_ap = a_point - line_point  # Find a Vector from a_point to line_point
    proj_AP_v = (np.dot(v_ap, line_vec) / np.dot(line_vec, line_vec)) * line_vec  # project v_ap onto line_vec
    distance = np.linalg.norm(v_ap - proj_AP_v)
    return distance


def veh_dist(veh1, veh2):
    veh1_location = vec2arr(veh1.get_transform().location)
    veh2_location = vec2arr(veh2.get_transform().location)
    d = np.linalg.norm(veh1_location - veh2_location)
    return d


def veh_speed(veh):
    return np.linalg.norm(vec2arr(veh.get_velocity())[:2])


def pprint_dict(dic):
    s = "| "
    for k, v in dic.items():
        if isinstance(v, np.ndarray):
            p = str(v.round(3))
        elif isinstance(v, float):
            p = round(v, 3)
        else:
            p = v
        s += f"{k}: {p} |"
    print(s)


def generate_trajectory(start_wp, spacing=1.0, n=100):
    wpt = start_wp
    trajectory = []
    for i in range(n):
        try:
            wpt = wpt.next(spacing)[0]
            trajectory.append(vec2arr(wpt.transform.location))
        except:
            break
    return trajectory


def distance_along_trajectory(veh_loc, trajectory, spacing=1.0):
    min_d = 1e10
    min_i = len(trajectory)
    for i, loc in enumerate(trajectory):
        d = np.linalg.norm(loc - vec2arr(veh_loc))
        if d < min_d:
            min_d = d
            min_i = i
    return min_i * spacing


def distance_along_road(veh, env_map, env_veh_list, max_distance=100):
    veh_transform = veh.get_transform()
    veh_location = veh_transform.location
    veh_wpt = env_map.get_waypoint(veh_location)

    trajectory = generate_trajectory(veh_wpt)
    distance_ahead = max_distance
    vehicle_ahead = None

    for other in env_veh_list:
        if other.id == veh.id:
            continue

        target_transform = other.get_transform()
        target_wpt = env_map.get_waypoint(target_transform.location, lane_type=carla.LaneType.Any)

        if veh_dist(veh, other) > distance_ahead:
            continue

        if get_rel_angle(target_transform, veh_transform) > 90:
            continue

        if target_wpt.lane_id == veh_wpt.lane_id:
            other_loc = other.get_transform().location
            d = distance_along_trajectory(other_loc, trajectory)
            if d < distance_ahead:
                distance_ahead = d
                vehicle_ahead = other

        else:
            continue

    return (distance_ahead, vehicle_ahead)
