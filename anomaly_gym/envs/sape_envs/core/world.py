from typing import List, Union

import numpy as np

from .entities import Agent, Entity, Goal, Hazard, Object


class World:
    """
    Implements the basic logic of the world.
    Contains one agent, one goal and several (possibly moving) objects.
    """

    def __init__(self, agent: Agent, objects: list[Object], hazards: list[Hazard], goal: Goal):
        # list of agent and entities
        self.agent = agent
        self.objects = objects
        self.goal = goal
        self.hazards = hazards

        # position dimensionality
        self.dim_p = 2
        # simulation timestep
        self.dt = 0.1
        # physical damping (like friction on moving objects)
        self.damping = 0.25
        # contact response parameters
        self.contact_force = 1e2
        self.contact_margin = 1e-3
        self.elapsed_simulation_steps = 0
        self._external_force_vector = np.array(0)

    @property
    def entities(self) -> List[Union[Agent, Object, Hazard, Goal]]:
        """
        return all entities in the world
        """
        return [self.agent, *self.objects, *self.hazards, self.goal]

    @property
    def actors(self) -> List[Union[Agent, Object]]:
        """
        return all acting (actively moving) entities in the world
        """
        return [self.agent, *self.scripted_objects]

    @property
    def policy_agent(self) -> Agent:
        """
        return agent controllable by external policy
        """
        return self.agent

    @property
    def scripted_objects(self) -> List[Object]:
        """
        return all objects controlled by world scripts
        """
        return [obj for obj in self.objects if obj.action_callback is not None]

    @property
    def static_objects(self) -> List[Object]:
        """
        return all static objects
        """
        return [obj for obj in self.objects if obj.action_callback is None]

    def reset_sim(self):
        self.elapsed_simulation_steps = 0

    def simulate_step(self) -> None:
        """
        update state of the world
        """
        self.elapsed_simulation_steps += 1
        for obj in self.scripted_objects:
            obj.action = obj.action_callback()
        self._step_env_force = self._gather_environment_force()
        self._step_act_force = self._gather_action_force()
        self._step_ext_force = self._gather_external_force()
        self._step_total_force = self._step_env_force + self._step_act_force + self._step_ext_force
        self._apply_force(self._step_total_force)

    def _gather_action_force(self) -> np.ndarray:
        """
        gather action forces coming from actors
        """
        p_force = np.zeros((len(self.entities), self.dim_p))
        for i, entity in enumerate(self.entities):
            if entity.action.u is not None:
                noise = np.random.randn(*entity.action.u.shape) * entity.action.u_noise
                p_force[i] = entity.action.u + noise
        return p_force

    def _gather_environment_force(self) -> np.ndarray:
        """
        gather physical forces acting on all entities
        with simple (but inefficient) collision response
        """
        p_force = np.zeros((len(self.entities), self.dim_p))
        for a, entity_a in enumerate(self.entities):
            for b, entity_b in enumerate(self.entities):
                if b <= a:
                    continue
                f_a, f_b = self._get_collision_force(entity_a, entity_b)
                p_force[a] += f_a
                p_force[b] += f_b
        return p_force

    def _gather_external_force(self) -> np.ndarray:
        return self._external_force_vector

    def _get_collision_force(self, entity_a: Entity, entity_b: Entity) -> tuple[float | np.ndarray, float | np.ndarray]:
        """
        get collision forces for any contact between two entities
        """
        if (not entity_a.collide) or (not entity_b.collide):
            return (np.zeros(self.dim_p), np.zeros(self.dim_p))  # not a collider
        if entity_a is entity_b:
            return (np.zeros(self.dim_p), np.zeros(self.dim_p))  # don't collide against itself
        # compute actual distance between entities
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        # minimum allowable distance
        dist_min = entity_a.size + entity_b.size
        if dist <= dist_min:
            # softmax penetration
            k = self.contact_margin
            penetration = np.logaddexp(0, -(dist - dist_min) / k) * k
            force = self.contact_force * delta_pos / dist * penetration
            force_a = +force if entity_a.movable else np.zeros_like(force)
            force_b = -force if entity_b.movable else np.zeros_like(force)
            return force_a, force_b
        else:
            return np.zeros_like(dist), np.zeros_like(dist)

    def _apply_force(self, p_force: np.ndarray) -> None:
        """
        simulate new physical state:
            - apply velocity damping -> reduces velocity by some factor every timestep
            - apply forces to entities resulting from own actions or external factors
            - limit vector velocities to max conform with max speed
        """
        for i, entity in enumerate(self.entities):
            if not entity.movable:
                continue
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
            if not np.allclose(p_force[i], np.zeros(self.dim_p)):
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
            if entity.max_speed is not None:
                speed = np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))
                if speed > entity.max_speed:
                    entity.state.p_vel = (
                        entity.state.p_vel
                        / np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))
                        * entity.max_speed
                    )
            entity.state.p_pos += entity.state.p_vel * self.dt
