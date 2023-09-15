'''
Maintainer Kai Gao
Sequencing samplers
TODO other forms of samplers?
'''

from typing import List, Any
import copy
import numpy as np
import math
from typing import Union
import cv2
import anytree

from lgmcts.algorithm.region_sampler import Region2DSampler, SampleData, SampleStatus,\
    sample_distribution, ObjectData


class Sampler:
    """
    manipulating_object, aligning_object, direction
    """

    def __init__(self, obj_name, origin_name, direction, region: Region2DSampler):
        self.obj_name = obj_name
        self.origin_name = origin_name
        self.direction = direction
        self.region = region


class Node(object):
    """MCTS Node"""

    def __init__(
        self,
        node_id,
        region_sampler: Region2DSampler,
        object_states: dict,
        sampler_dict=[],
        parent=None,
        action_from_parent=None,
        updated_obj_id=None,
        UCB_scalar=1.0,
        num_sampling=1,
        obj_support_tree: anytree.Node = None,
        prior_dict={},
        reward_mode='same',
        reward_dict={},
        is_virtual=False,
        verbose=False,
        rng=None
    ) -> None:

        self.node_id = node_id
        self.region_sampler = region_sampler
        self.object_states = object_states
        self.sampler_dict = sampler_dict
        self.unvisited_actions = {
            obj_id: list(range(num_sampling)) for obj_id in self.generate_actions()
        }
        self.children = {}  # we need multiple samples, so for each sampler, we sample k poses.
        self.parent = parent
        self.action_from_parent = action_from_parent  # the object wanted to move
        self.updated_obj_id = updated_obj_id  # the actually moved obj for the intended action
        self.visited_time = 0
        self.total_reward = 0
        self.UCB_scalar = UCB_scalar
        self.num_sampling = num_sampling
        self.obj_support_tree = obj_support_tree
        self.prior_dict = prior_dict
        self.reward_mode = reward_mode
        self.reward_dict = reward_dict
        self.is_virtual = is_virtual
        self.verbose = verbose
        self.rng = rng

        self.segmentation = None  # segmentation of the workspace, will be generated only once when needed

    def generate_actions(self):
        """
        generate the list of actions for this node. 
        That is, what samplers can be sampled 
        without breaking the pattern ordering
        """
        no_sample_objs = set()  # objects that cannot be sampled because of ordering
        for obj_id, sampler in self.sampler_dict.items():  # check ordering
            ordered = sampler.sample_info.get("ordered", False)
            if ordered:
                prior_objs = sampler.obj_ids[:sampler.obj_ids.index(obj_id)]
                for prior_obj in prior_objs:
                    if prior_obj in self.sampler_dict:
                        no_sample_objs.add(obj_id)
                        break

        return [obj_id for obj_id in self.sampler_dict.keys() if obj_id not in no_sample_objs]

    def UCB(self):
        """Upper confidence bound"""
        assert len(self.unvisited_actions) == 0
        best_action = 0
        best_sample = 0
        best_reward = -float("inf")
        for action, children_list in self.children.items():
            for child_id, child in enumerate(children_list):
                reward = child.total_reward / child.visited_time + self.UCB_scalar * math.sqrt(
                    2 * math.log10(self.visited_time) / child.visited_time
                )

                if reward >= best_reward:
                    best_reward = reward
                    best_action = action
                    best_sample = child_id
        return self.children[best_action][best_sample]

    def expansion(self):
        """expand the MCTS tree"""
        sampler_id = self.rng.choice(list(self.unvisited_actions.keys()))
        trial_id = self.unvisited_actions[sampler_id].pop(0)
        if len(self.unvisited_actions[sampler_id]) == 0:
            del self.unvisited_actions[sampler_id]
        action = (sampler_id, trial_id)
        return self.action_parametriczation(action)

    def action_parametriczation(self, action):
        """
        an action is represented by (sampler_id, trial_id), which is trying to 
        execute the sampler of object sampler_id for the trial_id-th time
        The the sampler of object sampler_id should has no pattern goal dependency
        check collision and start dependency (graspability)
        If in collision, try moving an obstacle away
        If not graspable, try moving a leaf on the subtree away
        Params:
            action: (sampler_id, trial_id)
        Returns: 
            action: (sampler_id, trial_id), 
            moved_obj: moved obj (obj to goal or obstacle), or None
            new_position: pose, 
            solved_sampler_obj_id: sampler_id or float('inf')
            solution_quality: float, the quality of solution, 0 if no solution
        """
        solution_quality = 0
        # check graspability
        found_node = anytree.search.find(
            self.obj_support_tree, lambda node: node.name == action[0]
        )
        if found_node and len(found_node.children) > 0:
            # not graspable, move a leave on the subtree away
            # Search for all leaf nodes
            leaf_nodes = found_node.leaves
            moved_obj = self.rng.choice(leaf_nodes).name
            # add a sampler to move the obstacle away
            buffer_sampler = SampleData(
                pattern="line",
                obj_id=moved_obj,
                obj_ids=[moved_obj],
                obj_poses_pix={})
            success, _, (moved_obj, new_position), _ = self.sampling_function(
                self.region_sampler,
                self.object_states,
                buffer_sampler
            )
            solved_sampler_obj_id = float('inf')
            return action, moved_obj, new_position, solved_sampler_obj_id, solution_quality

        sampler = self.sampler_dict[action[0]]
        success, obs, (moved_obj, new_position), solution_quality = self.sampling_function(
            self.region_sampler,
            self.object_states,
            sampler,
        )
        solved_sampler_obj_id, _ = action
        if not success:  # fails to complete the sampling, do
            if obs is None:
                # fails but not because of collision (e.g., out of workspace)
                solved_sampler_obj_id = float('inf')
                moved_obj = None
            else:
                # add a sampler to move the obstacle away
                buffer_sampler = SampleData(
                    pattern="line",
                    obj_id=obs,
                    obj_ids=[obs],
                    obj_poses_pix={})
                success, _, (moved_obj, new_position), _ = self.sampling_function(
                    self.region_sampler,
                    self.object_states,
                    buffer_sampler
                )
                solved_sampler_obj_id = float('inf')
        return action, moved_obj, new_position, solved_sampler_obj_id, solution_quality

    def sampling_function(
        self,
        region: Region2DSampler,
        object_states: dict,
        sample_data: SampleData,
        verbose: bool = False,
    ):
        """
        sampling function
        If sampling succeeded, return True, None, (moved_obj_id, new_pose)
        If sampling failed, return False, obs_name, (None, None)
        Params:
            RegionSampler, obj_name, origin_name, direction
        Returns: 
            success, obs_name, action:(obj_name, new_pos), solution_quality
        """
        # measure the sampling quality, and add it to the reward
        sample_quality = 0

        obj_id = sample_data.obj_id

        # update region
        region.set_object_poses(obj_states=object_states)
        # keep track of sampled object poses
        sampled_obj_poses_pix = {}
        pattern_objs = sample_data.obj_ids  # objects involved in the sampling pattern
        objs_away_from_goal = list(self.sampler_dict.keys())  # pattern objects away from goal
        objs_at_goal = [
            pattern_obj for pattern_obj in pattern_objs
            if (pattern_obj != obj_id) and (pattern_obj not in objs_away_from_goal)
        ]  # pattern objects at goal
        sampled_obj_poses_pix = {
            obj: region._world2pix(object_states[obj][:3] + region.objects[obj].pos_offset)
            for obj in objs_at_goal}

        # update prior
        if sample_data.pattern in self.prior_dict:
            prior, pattern_info = self.prior_dict[sample_data.pattern].gen_prior(
                region.grid_size, region.rng,
                obj_id=sample_data.obj_id,
                obj_ids=sample_data.obj_ids,
                obj_poses_pix=sampled_obj_poses_pix,
                sample_info=sample_data.sample_info
            )
            # cv2.imshow("prior", prior)
            # cv2.waitKey(0)
            # the prior object is too close to the boundary so that no sampling is possible
            if np.sum(prior) <= 0:
                leaf_nodes = self.obj_support_tree.leaves
                leaf_objs = set([n.name for n in leaf_nodes])
                sampled_objs = set(list(sampled_obj_poses_pix.keys()))
                graspable_sampled_objs = sampled_objs.intersection(leaf_objs)
                if len(graspable_sampled_objs) == 0:
                    graspable_sampled_objs = leaf_objs
                obs = self.rng.choice(list(graspable_sampled_objs))
                return False, obs, (obj_id, None), 0
            if self.is_virtual:
                # scene is virtual, so don't need to consider other objects
                region.set_objects_as_virtual(objs_away_from_goal)
            # DEBUG
            # region.visualize(block=False)
            # sample
            valid_pose, _, samples_status, sample_info = region.sample(sample_data.obj_id, 1, prior, allow_outside=False, pattern_info=pattern_info)
            if self.is_virtual:
                # reset to real objects
                region.set_objects_as_real(objs_away_from_goal)
            if valid_pose.shape[0] > 0:
                valid_pose = valid_pose.reshape(-1)

                if 'sample_probs' in sample_info:
                    sample_quality = sample_info['sample_probs']*sample_info['free_volume']/np.max(prior)
        else:
            raise NotImplementedError

        success = samples_status == SampleStatus.SUCCESS

        # test
        # print(f"sample status: {samples_status.name}, valid_pose: {valid_pose}")

        if not success:  # find an obstacle
            if self.segmentation is None:
                self.segmentation = self.semantic_segmentation(region)
            leaf_nodes = self.obj_support_tree.leaves
            leaf_objs = [n.name for n in leaf_nodes]
            counter = 100
            while (counter > 0):
                counter -= 1
                sample_pix, sample_probs = sample_distribution(prob=prior, rng=region.rng, n_samples=1)  # (N, 2)
                obs_id = self.segmentation[sample_pix[0][0], sample_pix[0][1], 0]
                if (obs_id not in [-1, obj_id]) and (obs_id in leaf_objs):
                    break
            if counter <= 0:
                obs_id = None
        else:
            obs_id = None
        action = (obj_id, valid_pose)
        return success, obs_id, action, sample_quality

    def semantic_segmentation(self, region: Region2DSampler):
        # TODO: Merge this part into sampler
        # semetic segmentation of the workspace
        segmentation = -1.0 * np.ones((region.grid_size[0], region.grid_size[1], 3), dtype=np.float32)
        # objects
        for obj_id, obj_data in region.objects.items():
            region._put_mask(
                mask=obj_data.mask,
                pos=obj_data.pos,
                rot=obj_data.rot,
                region_map=segmentation,
                value=float(obj_id),
            )
        return segmentation


class MCTS(object):
    """
    Input:
    1. region_sampler: current arrangement of objects
    2. L: A list of samplers [SamplerData1, SamplerData2,...]
    3. 'UCB_scalar':(float) UCB coefficients

    Output:
    1. a sequence of actions (o_i, p_i)
    """

    def __init__(
        self,
        region_sampler: Region2DSampler,
        L: List[SampleData],
        UCB_scalar=1.0,
        obj_support_tree: anytree.Node = None,
        prior_dict={},
        reward_mode='same',  # 'same' or 'prop'
        n_samples=1,
        is_virtual=False,
        verbose: bool = False,
        seed=4
    ) -> None:
        self.rng = np.random.default_rng(seed=seed)
        self.settings = {
            "UCB_scalar": UCB_scalar,
            "prior_dict": prior_dict,
            "rng": self.rng,
            "num_sampling": n_samples,
            'reward_mode': reward_mode
        }
        self.region_sampler = region_sampler
        self.sampler_dict = {s.obj_id: s for s in L}
        self.obj_support_tree = obj_support_tree  # initial object support tree
        self.start_state = region_sampler.get_object_poses()

        # intialize MCTS tree
        self.root = Node(
            0,
            region_sampler=region_sampler,
            object_states=self.start_state,
            sampler_dict=self.sampler_dict,
            parent=None,
            action_from_parent=None,
            updated_obj_id=None,
            obj_support_tree=self.obj_support_tree,
            reward_dict={},
            is_virtual=is_virtual,
            verbose=verbose,
            **self.settings,
        )

        # outputs
        self.action_list = []  # (objID, pos)
        self.isfeasible = False
        self.num_iter = 0
        # config
        self.is_virtual = is_virtual
        self.verbose = verbose

    def reset(self):
        """reset the sampler planner"""
        self.root = Node(
            0,
            region_sampler=self.region_sampler,
            object_states=self.start_state,
            sampler_dict=self.sampler_dict,
            parent=None,
            action_from_parent=None,
            updated_obj_id=None,
            obj_support_tree=self.obj_support_tree,
            reward_dict={},
            is_virtual=self.is_virtual,
            verbose=self.verbose,
            **self.settings,
        )
        self.action_list = []
        self.isfeasible = False
        self.num_iter = 0

    def search(self, max_iter: int = 10000, log_step: int = 1000) -> bool:
        """search for a feasible plan"""
        num_iter = 0

        while num_iter < max_iter:
            if (num_iter % log_step) == 0:
                print(f"Searched {num_iter}/{max_iter} iterations")
            num_iter += 1
            current_node = self.selection()
            # an action in MCTS is represented by (sampler_id, trail_id),
            # the index is according to L and the num_sample children list
            # TODO: do K sampling at the same time @KAI
            action, moved_obj, new_position, solved_sampler_obj_id, solution_quality = current_node.expansion()
            if (new_position.shape[0] > 0):  # go to a new state
                new_node = self.move(
                    num_iter,
                    action,
                    moved_obj,
                    new_position,
                    solved_sampler_obj_id,
                    solution_quality,
                    current_node,
                )
            else:  # stay in the same state
                new_node = Node(
                    num_iter,
                    region_sampler=self.region_sampler,
                    object_states=current_node.object_states,
                    sampler_dict=current_node.sampler_dict,
                    parent=current_node,
                    action_from_parent=action,
                    updated_obj_id=moved_obj,
                    obj_support_tree=copy_tree(current_node.obj_support_tree),
                    reward_dict={k: v for k, v in current_node.reward_dict.items()},
                    is_virtual=self.is_virtual,
                    verbose=self.verbose,
                    **self.settings,
                )

            # log this action
            if action[0] not in current_node.children:
                current_node.children[action[0]] = []
            current_node.children[action[0]].append(new_node)

            # update reward
            # TODO: new reward function @KAI
            reward = self.reward_detection(new_node)
            self.back_propagation(new_node, reward)
            if reward == len(self.sampler_dict):
                self.isfeasible = True
                self.construct_plan(new_node)
                return True
        # if not found, construct the best plan we can get
        self.construct_best_plan()
        # recording
        self.num_iter = num_iter
        return False

    def selection(self):
        """select a node to expand"""
        current_node = self.root
        while len(current_node.unvisited_actions) == 0:
            current_node = current_node.UCB()
        return current_node

    def move(
        self,
        node_id,
        action,
        obj,
        target,
        solved_sampler_obj_id: Union[int, None],
        solution_quality,
        current_node: Node
    ):
        """move the object to the a position and generate new node"""
        new_object_states = {
            k: v if k != obj else target for k, v in current_node.object_states.items()
        }
        # print(f"id: {node_id}, obj_states: {new_object_states}, target: {target}")

        new_sampler_dict = {obj_id: sampler for obj_id, sampler in current_node.sampler_dict.items() if obj_id != solved_sampler_obj_id}

        # update reward dict
        new_reward_dict = {obj_id: reward for obj_id, reward in current_node.reward_dict.items()}
        if solved_sampler_obj_id != float("inf"):
            new_reward_dict[solved_sampler_obj_id] = solution_quality

        # If we are moving an obstacle, the moved object may be an object moved to goal,
        # we need to retrive the sampler to indicate that this sampler needs to be solved again
        if solved_sampler_obj_id == float("inf"):
            backtracked_node = current_node
            while backtracked_node is not None:
                if obj in backtracked_node.sampler_dict.keys():
                    new_sampler_dict[obj] = backtracked_node.sampler_dict[obj]
                    if obj in new_reward_dict:
                        del new_reward_dict[obj]
                    break
                backtracked_node = backtracked_node.parent

        # update support tree
        new_tree = copy_tree(current_node.obj_support_tree)
        # check whether the object is in the support tree
        found_node = anytree.search.find(new_tree, lambda node: node.name == obj)
        if found_node:
            # parent should be root, assuming that the new pose is on table
            found_node.parent = new_tree
            # it should have no children at this point since we are moving it
            assert len(found_node.children) == 0

        new_node = Node(
            node_id,
            region_sampler=self.region_sampler,
            object_states=new_object_states,
            sampler_dict=new_sampler_dict,
            parent=current_node,
            action_from_parent=action,
            updated_obj_id=obj,
            obj_support_tree=new_tree,
            reward_dict=new_reward_dict,
            is_virtual=self.is_virtual,
            verbose=self.verbose,
            **self.settings,
        )
        return new_node

    def reward_detection(self, node: Node):
        """reward detection"""
        if self.settings['reward_mode'] == 'same':
            return len(self.sampler_dict) - len(node.sampler_dict)
        elif self.settings['reward_mode'] == 'prop':
            return sum(list(0.5+0.5*node.reward_dict.values()))

    def back_propagation(self, node: Node, reward):
        """back propagation in MCTS"""
        current_node = node
        while current_node is not None:
            current_node.visited_time += 1
            current_node.total_reward += reward
            current_node = current_node.parent

    def construct_plan(self, node: Node):
        """
        action_list: [(obj_id, pos)]
        """
        self.action_list = []
        current_node = node
        while current_node.parent is not None:
            parent_node = current_node.parent
            moved_object = current_node.updated_obj_id
            # current_node.show_arrangement()
            if moved_object is not None:
                old_pose = parent_node.object_states[moved_object].reshape(-1).astype(np.float32)
                new_pose = current_node.object_states[moved_object].reshape(-1).astype(np.float32)
                self.action_list.append(
                    {
                        "obj_id": moved_object,
                        "old_pose": old_pose,
                        "new_pose": new_pose,
                    }
                )
            current_node = parent_node
        self.action_list.reverse()

    def construct_best_plan(self):
        """get the best plan"""
        # search the node which has leaset left samplers
        best_node = self.root
        # traverse the tree
        queue = [self.root]
        while len(queue) > 0:
            node = queue.pop(0)
            if len(node.sampler_dict) < len(best_node.sampler_dict):
                best_node = node
            children_list = list(node.children.values())
            flattened_list = [item for sublist in children_list for item in sublist]
            queue.extend(flattened_list)
        # construct the plan
        self.construct_plan(best_node)


# copy anytree
def copy_tree(node: anytree.Node):
    copied_node = anytree.Node(copy.deepcopy(node.name))

    for child in node.children:
        child_copy = copy_tree(child)
        child_copy.parent = copied_node

    return copied_node
