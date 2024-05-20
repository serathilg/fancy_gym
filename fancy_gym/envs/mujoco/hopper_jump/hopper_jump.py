import os
from typing import Optional

import numpy as np
from gym.envs.mujoco.hopper_v4 import HopperEnv

MAX_EPISODE_STEPS_HOPPERJUMP = 250


class HopperJumpEnv(HopperEnv):
    """
    Initialization changes to normal Hopper:
    - terminate_when_unhealthy: True -> False
    - healthy_reward: 1.0 -> 2.0
    - healthy_z_range: (0.7, float('inf')) -> (0.5, float('inf'))
    - healthy_angle_range: (-0.2, 0.2) -> (-float('inf'), float('inf'))
    - exclude_current_positions_from_observation: True -> False
    """

    def __init__(
            self,
            xml_file='hopper_jump.xml',
            forward_reward_weight=1.0,
            ctrl_cost_weight=1e-3,
            healthy_reward=2.0,
            contact_weight=2.0,
            height_weight=10.0,
            dist_weight=3.0,
            terminate_when_unhealthy=False,
            healthy_state_range=(-100.0, 100.0),
            healthy_z_range=(0.5, float('inf')),
            healthy_angle_range=(-float('inf'), float('inf')),
            reset_noise_scale=5e-3,
            exclude_current_positions_from_observation=False,
            sparse=False,
    ):

        self.sparse = sparse
        self._height_weight = height_weight
        self._dist_weight = dist_weight
        self._contact_weight = contact_weight

        self.max_height = 0
        self.goal = np.zeros(3, )

        self._steps = 0
        self.contact_with_floor = False
        self.init_floor_contact = False
        self.has_left_floor = False
        self.contact_dist = None

        xml_file = os.path.join(os.path.dirname(__file__), "assets", xml_file)
        super().__init__(xml_file=xml_file,
                         forward_reward_weight=forward_reward_weight,
                         ctrl_cost_weight=ctrl_cost_weight,
                         healthy_reward=healthy_reward,
                         terminate_when_unhealthy=terminate_when_unhealthy,
                         healthy_state_range=healthy_state_range,
                         healthy_z_range=healthy_z_range,
                         healthy_angle_range=healthy_angle_range,
                         reset_noise_scale=reset_noise_scale,
                         exclude_current_positions_from_observation=exclude_current_positions_from_observation)

        # increase initial height
        self.init_qpos[1] = 1.5

    @property
    def exclude_current_positions_from_observation(self):
        return self._exclude_current_positions_from_observation

    def step(self, action):
        self._steps += 1

        self.do_simulation(action, self.frame_skip)

        height_after = self._get_torso_height()
        site_pos_after = self._get_foot_pos()
        self.max_height = max(height_after, self.max_height)

        has_floor_contact = self._is_floor_foot_contact() if not self.contact_with_floor else False

        if not self.init_floor_contact:
            self.init_floor_contact = has_floor_contact
        if self.init_floor_contact and not self.has_left_floor:
            self.has_left_floor = not has_floor_contact
        if not self.contact_with_floor and self.has_left_floor:
            self.contact_with_floor = has_floor_contact

        ctrl_cost = self.control_cost(action)
        costs = ctrl_cost
        done = self._steps >= MAX_EPISODE_STEPS_HOPPERJUMP

        goal_dist = np.linalg.norm(site_pos_after - self.goal)
        if self.contact_dist is None and self.contact_with_floor:
            self.contact_dist = goal_dist

        rewards = 0
        if not self.sparse or (self.sparse and self._steps >= MAX_EPISODE_STEPS_HOPPERJUMP):
            healthy_reward = self.healthy_reward
            distance_reward = -goal_dist * self._dist_weight
            height_reward = (self.max_height if self.sparse else height_after) * self._height_weight
            contact_reward = -(self.contact_dist or 5) * self._contact_weight
            rewards = self._forward_reward_weight * (distance_reward + height_reward + contact_reward + healthy_reward)

        observation = self._get_obs()
        reward = rewards - costs
        info = dict(
            height=height_after,
            x_pos=site_pos_after,
            max_height=self.max_height,
            goal=self.goal[:1],
            goal_dist=goal_dist,
            height_rew=self.max_height,
            healthy_reward=self.healthy_reward,
            healthy=self.is_healthy,
            contact_dist=self.contact_dist or 0
        )
        return observation, reward, done, info

    def _get_obs(self):
        goal_dist = self._get_foot_pos() - self.goal
        return np.concatenate((super()._get_obs(), goal_dist.copy(), self.goal[:1]))

    def reset_model(self):
        # super(HopperJumpEnv, self).reset_model()

        # self.goal = self.np_random.uniform(0.3, 1.35, 1)[0]
        self.goal = np.concatenate([self.np_random.uniform(0.3, 1.35, 1), np.zeros(2, )])
        # self.sim.model.body_pos[self.sim.model.body_name2id('goal_site_body')] = self.goal
        self.model.body('goal_site_body').pos[:] = np.copy(self.goal)
        self.max_height = 0
        self._steps = 0

        noise_low = np.zeros(self.model.nq)
        noise_low[3] = -0.5
        noise_low[4] = -0.2

        noise_high = np.zeros(self.model.nq)
        noise_high[5] = 0.785

        qpos = (
                self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nq) +
                self.init_qpos
        )
        qvel = (
            # self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nv) +
            self.init_qvel
        )

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        self.has_left_floor = False
        self.contact_with_floor = False
        self.init_floor_contact = False
        self.contact_dist = None

        return observation

    def _is_floor_foot_contact(self):
        # floor_geom_id = self.model.geom_name2id('floor')
        # foot_geom_id = self.model.geom_name2id('foot_geom')
        # TODO: do this properly over a sensor in the xml file, see dmc hopper
        floor_geom_id = self._mujoco_bindings.mj_name2id(self.model,
                                                         self._mujoco_bindings.mjtObj.mjOBJ_GEOM,
                                                         'floor')
        foot_geom_id = self._mujoco_bindings.mj_name2id(self.model,
                                                        self._mujoco_bindings.mjtObj.mjOBJ_GEOM,
                                                        'foot_geom')
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            collision = contact.geom1 == floor_geom_id and contact.geom2 == foot_geom_id
            collision_trans = contact.geom1 == foot_geom_id and contact.geom2 == floor_geom_id
            if collision or collision_trans:
                return True
        return False

    def _get_foot_pos(self) -> np.ndarray:
        return self.data.site("foot_site").xpos

    def _get_torso_height(self) -> float:
        return float(self.get_body_com("torso")[2])


# # TODO is that needed? if so test it
# class HopperJumpStepEnv(HopperJumpEnv):
#
#     def __init__(self,
#                  xml_file='hopper_jump.xml',
#                  forward_reward_weight=1.0,
#                  ctrl_cost_weight=1e-3,
#                  healthy_reward=1.0,
#                  height_weight=3,
#                  dist_weight=3,
#                  terminate_when_unhealthy=False,
#                  healthy_state_range=(-100.0, 100.0),
#                  healthy_z_range=(0.5, float('inf')),
#                  healthy_angle_range=(-float('inf'), float('inf')),
#                  reset_noise_scale=5e-3,
#                  exclude_current_positions_from_observation=False
#                  ):
#
#         self._height_weight = height_weight
#         self._dist_weight = dist_weight
#         super().__init__(xml_file, forward_reward_weight, ctrl_cost_weight, healthy_reward, terminate_when_unhealthy,
#                          healthy_state_range, healthy_z_range, healthy_angle_range, reset_noise_scale,
#                          exclude_current_positions_from_observation)
#
#     def step(self, action):
#         self._steps += 1
#
#         self.do_simulation(action, self.frame_skip)
#
#         height_after = self.get_body_com("torso")[2]
#         site_pos_after = self.data.site('foot_site').xpos.copy()
#         self.max_height = max(height_after, self.max_height)
#
#         ctrl_cost = self.control_cost(action)
#         healthy_reward = self.healthy_reward
#         height_reward = self._height_weight * height_after
#         goal_dist = np.linalg.norm(site_pos_after - np.array([self.goal, 0, 0]))
#         goal_dist_reward = -self._dist_weight * goal_dist
#         dist_reward = self._forward_reward_weight * (goal_dist_reward + height_reward)
#
#         rewards = dist_reward + healthy_reward
#         costs = ctrl_cost
#         done = False
#
#         # This is only for logging the distance to goal when first having the contact
#         has_floor_contact = self._is_floor_foot_contact() if not self.contact_with_floor else False
#
#         if not self.init_floor_contact:
#             self.init_floor_contact = has_floor_contact
#         if self.init_floor_contact and not self.has_left_floor:
#             self.has_left_floor = not has_floor_contact
#         if not self.contact_with_floor and self.has_left_floor:
#             self.contact_with_floor = has_floor_contact
#
#         if self.contact_dist is None and self.contact_with_floor:
#             self.contact_dist = goal_dist
#
#         ##############################################################
#
#         observation = self._get_obs()
#         reward = rewards - costs
#         info = {
#             'height': height_after,
#             'x_pos': site_pos_after,
#             'max_height': copy.copy(self.max_height),
#             'goal': copy.copy(self.goal),
#             'goal_dist': goal_dist,
#             'height_rew': height_reward,
#             'healthy_reward': healthy_reward,
#             'healthy': copy.copy(self.is_healthy),
#             'contact_dist': copy.copy(self.contact_dist) or 0
#         }
#         return observation, reward, done, info


class HopperJumpImmediateSparse(HopperJumpEnv):
    """
    Split sparse reward into immediate improvements/changes such that
    undiscounted episode return is identical (up to constants) to sparse
    reward.

    rewards:
    - dense healthy
    - contact distance immediately at contact
    - improvement of max height, i.e. max(0, current - previous_max_height)
    - dense goal distance change since last step (old_dist - new_dist)

    previous_max_height=0 at reset

    contact, max height and goal need a baseline to improve from, several options:

    max_height_from_min: bool
    - False: first step gets reward of height after first step (~init).
        much harder to reach new maximum because of initial fall.
    - True: no reward until height has improved once, i.e. after fall/
        first local minimum. reward differentiation even when jump height below init.

    delayed_contact_penalty: bool, contact_baseline: Optional[float]
    - True, float x: terminal reward -x if no second contact, else -dist at contact
    - False, float x: reward x-dist at second contact
    - True, None: x := first/fall contact distance,
        terminal reward -x if no second contact, else -dist at contact
    - False, None: x := first/fall contact distance,
        reward x-dist at second contact

    goal_baseline: Optional[float]
    Initial goal distance at reset from which changes are computed, affects first reward
    and summed reward over episode.
    - None: Distance of reset state to goal as baseline, summed episode reward is
        reset_dist - final_dist
    - float x: first step reward is x - first_dist (might be fairly negative), summed reward
        is x - final_dist (and hence equal to sparse reward -final_dist for x=0)


    healthy_delta: bool
    Whether to give absolute healthy state or change of healthiness
    - false: Dense healhty reward if currently healthy
    - true: Sparse healthy reward if healthy state has changed, reset as unhealthy such that
        summed undiscounted reward is final healthy status
    """

    def __init__(
        self,
        max_height_from_min: bool = False,
        delayed_contact_penalty: bool = True,
        contact_baseline: Optional[float] = 5.0,
        goal_baseline: Optional[float] = 0.0,
        healthy_delta: bool = True,
        xml_file="hopper_jump.xml",
        forward_reward_weight=1.0,
        ctrl_cost_weight=1e-3,
        healthy_reward=2.0,
        contact_weight=2.0,
        height_weight=10.0,
        dist_weight=3.0,
        terminate_when_unhealthy=False,
        healthy_state_range=(-100.0, 100.0),
        healthy_z_range=(0.5, float("inf")),
        healthy_angle_range=(-float("inf"), float("inf")),
        reset_noise_scale=5e-3,
        exclude_current_positions_from_observation=False,
    ):
        self.max_height_from_min = max_height_from_min
        self.delayed_contact_penalty = delayed_contact_penalty
        self.contact_baseline = contact_baseline
        self.goal_baseline = goal_baseline
        self.healthy_delta = healthy_delta

        self._prev_height: float = float("-inf")
        self._max_height_after_min: float = 0.0
        self._had_local_min_height: bool = False

        self._last_dist_of_init_contact: Optional[float] = None

        self._prev_goal_dist: float = float("-inf")

        self._prev_healthy_reward: float = 0.0

        super().__init__(
            xml_file=xml_file,
            forward_reward_weight=forward_reward_weight,
            ctrl_cost_weight=ctrl_cost_weight,
            healthy_reward=healthy_reward,
            contact_weight=contact_weight,
            height_weight=height_weight,
            dist_weight=dist_weight,
            terminate_when_unhealthy=terminate_when_unhealthy,
            healthy_state_range=healthy_state_range,
            healthy_z_range=healthy_z_range,
            healthy_angle_range=healthy_angle_range,
            reset_noise_scale=reset_noise_scale,
            exclude_current_positions_from_observation=exclude_current_positions_from_observation,
            sparse=True,
        )

    def step(self, action):
        self._steps += 1

        self.do_simulation(action, self.frame_skip)

        height_after = self._get_torso_height()
        site_pos_after = self._get_foot_pos()

        if height_after > self._prev_height and not self._had_local_min_height:
            # last step was first local minimum of height
            self._had_local_min_height = True
        self._prev_height = height_after

        height_improvement_after_min: float
        if self._had_local_min_height:
            # check if improvement of max height and update
            height_improvement_after_min = max(
                0.0, height_after - self._max_height_after_min
            )
            self._max_height_after_min = max(height_after, self._max_height_after_min)
        else:
            height_improvement_after_min = 0.0

        height_improvement = max(0.0, height_after - self.max_height)

        self.max_height = max(height_after, self.max_height)

        has_floor_contact = (
            self._is_floor_foot_contact() if not self.contact_with_floor else False
        )

        if not self.init_floor_contact:
            self.init_floor_contact = has_floor_contact
        if self.init_floor_contact and not self.has_left_floor:
            self.has_left_floor = not has_floor_contact
            if not self.has_left_floor:
                self._last_dist_of_init_contact = np.linalg.norm(
                    site_pos_after - self.goal
                )
        if not self.contact_with_floor and self.has_left_floor:
            self.contact_with_floor = has_floor_contact

        ctrl_cost = self.control_cost(action)
        costs = ctrl_cost
        done = self._steps >= MAX_EPISODE_STEPS_HOPPERJUMP

        if done and not self._had_local_min_height:
            # continuous fall, final height as reward
            height_improvement_after_min = height_after

        goal_dist = np.linalg.norm(site_pos_after - self.goal)
        goal_dist_reduction = self._prev_goal_dist - goal_dist
        self._prev_goal_dist = goal_dist

        second_contact_now = False
        if self.contact_dist is None and self.contact_with_floor:
            self.contact_dist = goal_dist
            second_contact_now = True

        # contact reward with delayed penalty
        if second_contact_now:
            # negative land distance as reward immediately
            delayed_contacts_dist_reduction = -self.contact_dist
        elif done and self.contact_dist is None:
            # never jumped (or not yet landed) -> baseline penalty
            # if we somehow manage to not even land once (headstand?) then large dist 100 penalty
            delayed_contacts_dist_reduction = -(
                self.contact_baseline or self._last_dist_of_init_contact or 100.0
            )
        else:
            delayed_contacts_dist_reduction = 0.0

        # contact reward with only improvement at second contact
        if second_contact_now:
            # use baseline if set, otherwise dist of first contact
            jump_start_dist = self.contact_baseline or self._last_dist_of_init_contact
            assert (
                jump_start_dist is not None
            ), "Impossible, second contact requires first contact."
            # reduction of distance as reward immediately
            direct_contacts_dist_reduction = jump_start_dist - self.contact_dist
        else:
            direct_contacts_dist_reduction = 0.0

        contacts_dist_reduction = (
            delayed_contacts_dist_reduction
            if self.delayed_contact_penalty
            else direct_contacts_dist_reduction
        )

        if self.healthy_delta:
            healthy_reward = self.healthy_reward - self._prev_healthy_reward
            self._prev_healthy_reward = self.healthy_reward
        else:
            # healthy reward is dense but other rewards sum to sparse, so scale
            # by MAX_EPISODE_STEPS_HOPPERJUMP to maintain same ratios.
            healthy_reward = self.healthy_reward / MAX_EPISODE_STEPS_HOPPERJUMP
        distance_reward = goal_dist_reduction * self._dist_weight
        height_reward = (
            height_improvement_after_min
            if self.max_height_from_min
            else height_improvement
        ) * self._height_weight
        contact_reward = contacts_dist_reduction * self._contact_weight
        rewards = self._forward_reward_weight * (
            distance_reward + height_reward + contact_reward + healthy_reward
        )

        observation = self._get_obs()
        reward = rewards - costs
        info = dict(
            height=height_after,
            x_pos=site_pos_after,
            max_height=self.max_height,
            max_height_after_min=self._max_height_after_min,
            goal=self.goal[:1],
            goal_dist=goal_dist,
            height_rew=height_reward,
            healthy_reward=healthy_reward,
            healthy=self.is_healthy,
            contact_dist=self.contact_dist or 0,
            contacts_dist_reduction=contacts_dist_reduction,
            delayed_contacts_dist_reduction=delayed_contacts_dist_reduction,
            direct_contacts_dist_reduction=direct_contacts_dist_reduction,
        )
        return observation, reward, done, info

    def reset_model(self):
        observation = super().reset_model()

        self._prev_height = self._get_torso_height()
        self._max_height_after_min = 0.0
        self._had_local_min_height = False

        self._last_dist_of_init_contact = None

        self._prev_goal_dist = self.goal_baseline or np.linalg.norm(
            self._get_foot_pos() - self.goal
        )

        self._prev_healthy_reward = 0.0

        return observation
