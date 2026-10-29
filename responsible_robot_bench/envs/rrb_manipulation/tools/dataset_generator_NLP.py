from multiprocessing import Process, Manager
from time import time, sleep

from pyrep.const import RenderMode
from os.path import join, dirname, abspath, isfile
import sys
from pathlib import Path
CURRENT_DIR = dirname(abspath(__file__))
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from rrb.core import TASKS
from rrb.planning.task_metadata import task_id
from rrb.collection import (annotate_native_episode, inventory_native_episodes,
                            publish_native_episode, variation_lock, write_json,
                            collection_exception, preserve_failed_collection,
                            recover_interrupted_native_episodes)

sys.path.insert(0, join(CURRENT_DIR, '..'))  # Use local amsolver rather than installed
from amsolver import ObservationConfig
from amsolver.demo import Demo
from amsolver.action_modes import ActionMode
from amsolver.backend.utils import task_file_to_task_class
from amsolver.environment import Environment
import amsolver.backend.task as task

import os
import json
import random
import hashlib
import pickle
from PIL import Image
from amsolver.backend import utils
from amsolver.backend.const import *
import numpy as np

from absl import app
from absl import flags

"""
Change to different folder for different split. 
If you want to generate unseen data, please set the colors and objects to the unseen collections in amsolver/constant.py
"""

FLAGS = flags.FLAGS

flags.DEFINE_string('save_path',
                    './rlbench_data/train',
                    'Where to save the demos.')
flags.DEFINE_string('queue_status_path', None,
                    'Optional separate queue status file for concurrent collectors.')
flags.DEFINE_list('tasks', list(TASKS.values()),
                  'RRB task IDs or module names. If empty, collect all 23 tasks.')
flags.DEFINE_list('image_size', [360, 360],
                  'The size of the images tp save.')
flags.DEFINE_enum('renderer',  'opengl', ['opengl', 'opengl3'],
                  'The renderer to use. opengl does not include shadows, '
                  'but is faster.')
flags.DEFINE_integer('processes', 8,
                     'The number of parallel processes during collection.')
flags.DEFINE_integer('episodes_per_task', 5,
                     'Target total complete episodes per task, including existing episodes.')
flags.DEFINE_integer('max_attempts', 10, 'Attempts per missing episode before trying the next task.')
flags.DEFINE_bool('retry_failed', True, 'Requeue native collection failures until targets are reached.')
flags.DEFINE_integer('max_retry_rounds', 0, 'Retry rounds after the first pass; zero means no round limit.')
flags.DEFINE_float('retry_delay', 30., 'Seconds between failed-task retry rounds.')
flags.DEFINE_integer('seed', None, 'Optional reproducible seed for new collection attempts.')
flags.DEFINE_integer('variations', -1,
                     'Number of variations to collect per task. -1 for all.')
flags.DEFINE_bool('save_configs', True,
                     'whether also save the config for replay.')


def check_and_make(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def save_configs(task_base, waypoint_sets, config, example_path):
    check_and_make(example_path)
    # Save the low-dimension data
    with open(os.path.join(example_path, "configs.pkl"), 'wb') as f:
        pickle.dump(config, f)
    task_base.save_model(os.path.join(example_path, "task_base.ttm"))
    waypoint_sets.save_model(os.path.join(example_path, "waypoint_sets.ttm"))

def save_demo(demo, example_path):

    # Save image data first, and then None the image data, and pickle
    left_shoulder_rgb_path = os.path.join(
        example_path, LEFT_SHOULDER_RGB_FOLDER)
    left_shoulder_depth_path = os.path.join(
        example_path, LEFT_SHOULDER_DEPTH_FOLDER)
    left_shoulder_mask_path = os.path.join(
        example_path, LEFT_SHOULDER_MASK_FOLDER)
    right_shoulder_rgb_path = os.path.join(
        example_path, RIGHT_SHOULDER_RGB_FOLDER)
    right_shoulder_depth_path = os.path.join(
        example_path, RIGHT_SHOULDER_DEPTH_FOLDER)
    right_shoulder_mask_path = os.path.join(
        example_path, RIGHT_SHOULDER_MASK_FOLDER)
    overhead_rgb_path = os.path.join(
        example_path, OVERHEAD_RGB_FOLDER)
    overhead_depth_path = os.path.join(
        example_path, OVERHEAD_DEPTH_FOLDER)
    overhead_mask_path = os.path.join(
        example_path, OVERHEAD_MASK_FOLDER)
    wrist_rgb_path = os.path.join(example_path, WRIST_RGB_FOLDER)
    wrist_depth_path = os.path.join(example_path, WRIST_DEPTH_FOLDER)
    wrist_mask_path = os.path.join(example_path, WRIST_MASK_FOLDER)
    front_rgb_path = os.path.join(example_path, FRONT_RGB_FOLDER)
    front_depth_path = os.path.join(example_path, FRONT_DEPTH_FOLDER)
    front_mask_path = os.path.join(example_path, FRONT_MASK_FOLDER)

    check_and_make(left_shoulder_rgb_path)
    check_and_make(left_shoulder_depth_path)
    check_and_make(left_shoulder_mask_path)
    check_and_make(right_shoulder_rgb_path)
    check_and_make(right_shoulder_depth_path)
    check_and_make(right_shoulder_mask_path)
    check_and_make(overhead_rgb_path)
    check_and_make(overhead_depth_path)
    check_and_make(overhead_mask_path)
    check_and_make(wrist_rgb_path)
    check_and_make(wrist_depth_path)
    check_and_make(wrist_mask_path)
    check_and_make(front_rgb_path)
    check_and_make(front_depth_path)
    check_and_make(front_mask_path)

    for i, obs in enumerate(demo):
        left_shoulder_rgb = Image.fromarray(obs.left_shoulder_rgb)
        left_shoulder_depth = utils.float_array_to_rgb_image(
            obs.left_shoulder_depth, scale_factor=DEPTH_SCALE)
        left_shoulder_mask = Image.fromarray(
            (obs.left_shoulder_mask * 255).astype(np.uint8))
        right_shoulder_rgb = Image.fromarray(obs.right_shoulder_rgb)
        right_shoulder_depth = utils.float_array_to_rgb_image(
            obs.right_shoulder_depth, scale_factor=DEPTH_SCALE)
        right_shoulder_mask = Image.fromarray(
            (obs.right_shoulder_mask * 255).astype(np.uint8))
        overhead_rgb = Image.fromarray(obs.overhead_rgb)
        overhead_depth = utils.float_array_to_rgb_image(
            obs.overhead_depth, scale_factor=DEPTH_SCALE)
        overhead_mask = Image.fromarray(
            (obs.overhead_mask * 255).astype(np.uint8))
        wrist_rgb = Image.fromarray(obs.wrist_rgb)
        wrist_depth = utils.float_array_to_rgb_image(
            obs.wrist_depth, scale_factor=DEPTH_SCALE)
        wrist_mask = Image.fromarray((obs.wrist_mask * 255).astype(np.uint8))
        front_rgb = Image.fromarray(obs.front_rgb)
        front_depth = utils.float_array_to_rgb_image(
            obs.front_depth, scale_factor=DEPTH_SCALE)
        front_mask = Image.fromarray((obs.front_mask * 255).astype(np.uint8))

        left_shoulder_rgb.save(
            os.path.join(left_shoulder_rgb_path, IMAGE_FORMAT % i))
        left_shoulder_depth.save(
            os.path.join(left_shoulder_depth_path, IMAGE_FORMAT % i))
        left_shoulder_mask.save(
            os.path.join(left_shoulder_mask_path, IMAGE_FORMAT % i))
        right_shoulder_rgb.save(
            os.path.join(right_shoulder_rgb_path, IMAGE_FORMAT % i))
        right_shoulder_depth.save(
            os.path.join(right_shoulder_depth_path, IMAGE_FORMAT % i))
        right_shoulder_mask.save(
            os.path.join(right_shoulder_mask_path, IMAGE_FORMAT % i))
        overhead_rgb.save(
            os.path.join(overhead_rgb_path, IMAGE_FORMAT % i))
        overhead_depth.save(
            os.path.join(overhead_depth_path, IMAGE_FORMAT % i))
        overhead_mask.save(
            os.path.join(overhead_mask_path, IMAGE_FORMAT % i))
        wrist_rgb.save(os.path.join(wrist_rgb_path, IMAGE_FORMAT % i))
        wrist_depth.save(os.path.join(wrist_depth_path, IMAGE_FORMAT % i))
        wrist_mask.save(os.path.join(wrist_mask_path, IMAGE_FORMAT % i))
        front_rgb.save(os.path.join(front_rgb_path, IMAGE_FORMAT % i))
        front_depth.save(os.path.join(front_depth_path, IMAGE_FORMAT % i))
        front_mask.save(os.path.join(front_mask_path, IMAGE_FORMAT % i))

        # We save the images separately, so set these to None for pickling.
        obs.left_shoulder_rgb = None
        obs.left_shoulder_depth = None
        obs.left_shoulder_point_cloud = None
        obs.left_shoulder_mask = None
        obs.right_shoulder_rgb = None
        obs.right_shoulder_depth = None
        obs.right_shoulder_point_cloud = None
        obs.right_shoulder_mask = None
        obs.overhead_rgb = None
        obs.overhead_depth = None
        obs.overhead_point_cloud = None
        obs.overhead_mask = None
        obs.wrist_rgb = None
        obs.wrist_depth = None
        obs.wrist_point_cloud = None
        obs.wrist_mask = None
        obs.front_rgb = None
        obs.front_depth = None
        obs.front_point_cloud = None
        obs.front_mask = None

    # Save the low-dimension data
    with open(os.path.join(example_path, LOW_DIM_PICKLE), 'wb') as f:
        pickle.dump(demo, f)


def collect_variation(task_env, variation_path):
    """Resume the original full-observation collector without overwriting data."""
    variation_path = Path(variation_path)
    episodes_path = variation_path / EPISODES_FOLDER
    episodes_path.mkdir(parents=True, exist_ok=True)
    with variation_lock(variation_path):
        recovered_writes = recover_interrupted_native_episodes(episodes_path)
        inventory = inventory_native_episodes(episodes_path, FLAGS.save_configs)
        completed = len(inventory['complete'])
        episode_index = inventory['next_index']
        target = FLAGS.episodes_per_task
        progress_path = variation_path / 'collection_status.json'
        try:
            previous = json.loads(progress_path.read_text()) if progress_path.exists() else {}
        except (OSError, ValueError):
            # Counts and allocated IDs are reconstructed from saved episodes.
            previous = {}
        previous_attempts = previous.get('attempts', [])
        failed_ids = [int(p.name[7:]) for p in (variation_path / 'failed_episodes').glob('attempt*')
                      if p.name[7:].isdigit()]
        next_attempt_id = max(previous.get('next_attempt_id', len(previous_attempts)),
                              max(failed_ids, default=-1) + 1)
        progress = dict(task=task_env.get_name(), target_total=target,
                        existing_complete=completed, completed=completed,
                        incomplete_existing=inventory['incomplete'], attempts=previous_attempts,
                        next_attempt_id=next_attempt_id, annotations=[],
                        recovered_writes=recovered_writes,
                        state='collecting')
        write_json(progress_path, progress)

        def annotate_saved(path, details):
            annotation = dict(path=str(path))
            try:
                if not hasattr(task_env._task, 'objects_list'):
                    # Existing episodes can be annotated before the first reset;
                    # initialize the task exactly as the first reset would.
                    task_env._scene.init_task()
                annotation.update(annotate_native_episode(path, task_env.get_name(), details, task_env._task.objects_list))
            except Exception as error:
                annotation.update(state='failed', raw_preserved=True,
                                  error_type=type(error).__name__, error=str(error))
                progress['state'] = 'annotation_failed'
                raise RuntimeError(f'YOLO annotation failed; raw episode preserved at {path}') from error
            finally:
                progress['annotations'].append(annotation)
                write_json(progress_path, progress)
                print(json.dumps(annotation), flush=True)

        for existing in inventory['complete']:
            annotate_saved(Path(existing['path']), existing)
        while completed < target:
            collected = False
            for attempt in range(FLAGS.max_attempts):
                attempt_id = progress['next_attempt_id']
                progress['next_attempt_id'] += 1
                write_json(progress_path, progress)
                if FLAGS.seed is not None:
                    token = f'{FLAGS.seed}:{task_env.get_name()}:{variation_path.name}:{episode_index}:{attempt_id}'
                    seed = int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], 'little')
                    np.random.seed(seed)
                    random.seed(seed)
                else:
                    seed = None
                event = dict(episode_index=episode_index, attempt=attempt + 1, attempt_id=attempt_id,
                             seed=seed, started_unix=time())
                demo = None
                recorded = []
                failure_kind = None
                stored_failure = None
                phase = 'native_collection'
                try:
                    # Preserve native scene generation, motion planning and outcomes.
                    returned = task_env.get_demos(amount=1, live_demos=True, max_attempts=1,
                                                  callable_each_step=recorded.append)
                    if not isinstance(returned, tuple) or len(returned) != 2:
                        raise ValueError('Native collector must return demonstrations and success labels')
                    demos, outcomes = returned
                    if len(demos) != 1 or len(outcomes) != 1:
                        raise ValueError('Expected one native demonstration and one outcome')
                    demo, success = demos[0], bool(outcomes[0])
                    event['task_success'] = success
                    event['terminal_safe'] = bool(demo[-1].is_safe)
                    if success:
                        phase = 'storage'
                        configs = task_env.read_config(demo.high_level_instructions) if FLAGS.save_configs else None
                        def save(destination):
                            save_demo(demo, str(destination))
                            if configs is not None:
                                save_configs(*configs, str(destination))
                        details = publish_native_episode(episodes_path, episode_index, save,
                                                         FLAGS.save_configs, metadata=event)
                        event.update(saved=True, frames=details['frames'], path=f'episodes/episode{episode_index}')
                        completed += 1
                        episode_index += 1
                        collected = True
                    else:
                        event.update(saved=False, reason='Native motion demonstration did not complete the goal')
                        failure_kind = 'goal_not_completed'
                except Exception as error:
                    stored_failure = getattr(error, 'failure_data', None)
                    event.update(saved=False, **collection_exception(error))
                    failure_kind = 'exception' if phase == 'native_collection' else 'storage_error'
                    if demo is None and recorded:
                        demo = Demo(recorded)
                        descriptions = getattr(getattr(task_env, '_scene', None), 'descriptions', None)
                        if descriptions is not None:
                            demo.high_level_instructions = descriptions
                        observed_safe = getattr(recorded[-1], 'is_safe', None)
                        event['last_observed_safe'] = None if observed_safe is None else bool(observed_safe)
                event['elapsed_seconds'] = time() - event['started_unix']
                if not collected:
                    event.update(failure_kind=failure_kind, truncated=failure_kind == 'exception')
                    event['failure_data'] = stored_failure or preserve_failed_collection(
                        variation_path, event, demo, save_demo, save_configs, task_env, FLAGS.save_configs)
                progress['attempts'].append(event)
                progress['completed'] = completed
                write_json(progress_path, progress)
                print(json.dumps(event), flush=True)
                if failure_kind == 'storage_error':
                    progress['state'] = 'storage_failed'
                    write_json(progress_path, progress)
                    raise RuntimeError(f'Native goal completed but storage failed; see {progress_path}')
                if collected:
                    # Outside the simulation retry block: annotation failure must
                    # never discard or resimulate an already saved success.
                    annotate_saved(episodes_path / f"episode{event['episode_index']}", details)
                    break
            if not collected:
                progress['state'] = 'attempts_exhausted'
                write_json(progress_path, progress)
                raise RuntimeError(f'{task_env.get_name()}: collected {completed}/{target}; see {progress_path}')
        progress['state'] = 'complete'
        write_json(progress_path, progress)
        return progress


def run(i, lock, task_index, variation_count, results, file_lock, tasks):
    """Each thread will choose one task and variation, and then gather
    all the episodes_per_task for that variation."""

    # Initialise each thread with random seed
    np.random.seed(None)
    num_tasks = len(tasks)

    img_size = list(map(int, FLAGS.image_size))

    obs_config = ObservationConfig()
    obs_config.set_all(True)
    obs_config.right_shoulder_camera.image_size = img_size
    obs_config.left_shoulder_camera.image_size = img_size
    obs_config.overhead_camera.image_size = img_size
    obs_config.wrist_camera.image_size = img_size
    obs_config.front_camera.image_size = img_size

    # Store depth as 0 - 1
    obs_config.right_shoulder_camera.depth_in_meters = False
    obs_config.left_shoulder_camera.depth_in_meters = False
    obs_config.overhead_camera.depth_in_meters = False
    obs_config.wrist_camera.depth_in_meters = False
    obs_config.front_camera.depth_in_meters = False

    # We want to save the masks as rgb encodings.
    obs_config.left_shoulder_camera.masks_as_one_channel = False
    obs_config.right_shoulder_camera.masks_as_one_channel = False
    obs_config.overhead_camera.masks_as_one_channel = False
    obs_config.wrist_camera.masks_as_one_channel = False
    obs_config.front_camera.masks_as_one_channel = False

    if FLAGS.renderer == 'opengl':
        obs_config.right_shoulder_camera.render_mode = RenderMode.OPENGL
        obs_config.left_shoulder_camera.render_mode = RenderMode.OPENGL
        obs_config.overhead_camera.render_mode = RenderMode.OPENGL
        obs_config.wrist_camera.render_mode = RenderMode.OPENGL
        obs_config.front_camera.render_mode = RenderMode.OPENGL

    # Create a virtual environment instance
    amsolver_env = Environment(
        action_mode=ActionMode(),
        obs_config=obs_config,
        headless=True) # set headless=False, if user want to visualize the simulator 
    amsolver_env.launch()

    task_env = None

    errors, retryable_tasks, fatal_tasks = [], [], []
    results[i] = dict(errors=[], retryable_tasks=[], fatal_tasks=[])

    while True:
        # Figure out what task/variation this thread is going to do
        with lock:

            if task_index.value >= num_tasks:
                print('Process', i, 'finished')
                break

            my_variation_count = variation_count.value
            t = tasks[task_index.value]
            task_env = amsolver_env.get_task(t)
            var_target = task_env.variation_count()
            if FLAGS.variations >= 0:
                var_target = np.minimum(FLAGS.variations, var_target)
            if my_variation_count >= var_target:
                # If we have reached the required number of variations for this
                # task, then move on to the next task.
                variation_count.value = my_variation_count = 0
                task_index.value += 1

            variation_count.value += 1
            if task_index.value >= num_tasks:
                print('Process', i, 'finished')
                break
            t = tasks[task_index.value]

        task_env = amsolver_env.get_task(t)
        task_env.set_variation(my_variation_count)  # Set Variation

        variation_path = os.path.join(
            FLAGS.save_path, task_env.get_name(),
            VARIATIONS_FOLDER % my_variation_count)

        check_and_make(variation_path)

        try:
            collect_variation(task_env, variation_path)
        except Exception as error:
            problem = f'Process {i}, {task_env.get_name()}, variation {my_variation_count}: {error}\n'
            print(problem, flush=True)
            errors.append(problem)
            status_path = Path(variation_path) / 'collection_status.json'
            try:
                status = json.loads(status_path.read_text()) if status_path.exists() else {}
            except (OSError, ValueError):
                status = {}
            if status.get('state') == 'attempts_exhausted':
                retryable_tasks.append(task_env.get_name())
            else:
                fatal_tasks.append(task_env.get_name())
        results[i] = dict(errors=errors, retryable_tasks=retryable_tasks, fatal_tasks=fatal_tasks)

    results[i] = dict(errors=errors, retryable_tasks=retryable_tasks, fatal_tasks=fatal_tasks)
    amsolver_env.shutdown()


def main(argv):
    if FLAGS.episodes_per_task < 1 or FLAGS.processes < 1 or FLAGS.max_attempts < 1:
        raise ValueError('episodes_per_task, processes and max_attempts must be positive')
    if FLAGS.max_retry_rounds < 0 or not np.isfinite(FLAGS.retry_delay) or FLAGS.retry_delay < 0:
        raise ValueError('Retry round count and delay must be nonnegative and finite')

    task_files = [TASKS[task_id(name)] for name in (FLAGS.tasks or TASKS)]
    tasks_path = Path(CURRENT_DIR).parent / 'vlm/tasks'
    for name in task_files:
        if not (tasks_path / (name + '.py')).is_file():
            raise ValueError('Missing registered RRB task module: ' + name)

    check_and_make(FLAGS.save_path)
    pending = task_files
    round_index = 0
    queue_path = (Path(FLAGS.queue_status_path) if getattr(FLAGS, 'queue_status_path', None)
                  else Path(FLAGS.save_path) / 'collection_queue.json')
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    with Manager() as manager:
        while pending:
            tasks = [task_file_to_task_class(t, parent_folder='vlm') for t in pending]
            result_dict = manager.dict()
            task_index, variation_count = manager.Value('i', 0), manager.Value('i', 0)
            lock, file_lock = manager.Lock(), manager.Lock()
            write_json(queue_path, dict(state='collecting', round=round_index, pending=pending,
                                        target_per_task=FLAGS.episodes_per_task))
            processes = [Process(target=run, args=(
                i, lock, task_index, variation_count, result_dict, file_lock, tasks))
                for i in range(min(FLAGS.processes, len(tasks)))]
            for process in processes:
                process.start()
            for process in processes:
                process.join()
            failures, retryable, fatal = [], set(), []
            for i, process in enumerate(processes):
                result = result_dict.get(i, {})
                if not result:
                    fatal.append(f'Worker {i} returned no collection report')
                failures.extend(result.get('errors', []))
                retryable.update(result.get('retryable_tasks', []))
                fatal.extend(result.get('fatal_tasks', []))
                if process.exitcode != 0:
                    fatal.append(f'Worker {i} exited with code {process.exitcode}')
            if not failures and not fatal:
                write_json(queue_path, dict(state='complete', round=round_index, pending=[],
                                            target_per_task=FLAGS.episodes_per_task))
                print('Data collection complete.', flush=True)
                return
            pending = [name for name in task_files if name in retryable]
            can_retry = (FLAGS.retry_failed and pending and not fatal
                         and (FLAGS.max_retry_rounds == 0 or round_index < FLAGS.max_retry_rounds))
            write_json(queue_path, dict(state='retry_wait' if can_retry else 'failed',
                                        round=round_index, pending=pending, errors=failures,
                                        fatal=fatal, target_per_task=FLAGS.episodes_per_task))
            if not can_retry:
                raise RuntimeError('Collection incomplete: ' + '; '.join(failures + fatal))
            print(f'Retrying {len(pending)} unfinished tasks after {FLAGS.retry_delay:g} seconds.', flush=True)
            sleep(FLAGS.retry_delay)
            round_index += 1


if __name__ == '__main__':
    # Render data for different Variation Episode
  app.run(main)
