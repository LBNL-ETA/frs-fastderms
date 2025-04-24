"""
Part of the FAST-DERMS Flexible Resource Scheduler Demonstration
"""

from pathlib import Path

import os
import json
import subprocess
import sys
import logging
import time

import datetime as dt


class Demo_Handler(object):
    def __init__(self, logger: logging.Logger = None):
        self.processes = []
        self.TIMER_LVL = 31
        logging.addLevelName(self.TIMER_LVL, "TIMER")
        if logger is None:
            self.logger = logging.getLogger("Demo Handler")
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger = logger
        self.input_topic = ""
        self.automator_setup = False
        self.s0 = time.time()
        self.s1 = self.s0
        self.task_name = "task"
        self.task_running = False
        self.error_detected = False

    def add_process(self, path_to_app, simulation_id, request, app_config):
        args = [
            "python",
            path_to_app,
            str(simulation_id),
            json.dumps(request),
            json.dumps(app_config),
        ]
        proc = subprocess.Popen(
            args, env={**os.environ, "PYTHONPATH": ":".join(sys.path)}
        )
        self.processes.append(proc)
        return proc

    def clean_processes(self, simulation_obj=None):
        if self.add_process:
            self.logger.warning("Killing all existing processes / Apps")
        remaining_processes = []
        for proc in self.processes:
            try:
                self.logger.warning(f"Killing process: {proc.pid} {proc.args}")
                proc.kill()
                proc.wait()
            except Exception as e:
                self.logger.error(f"Something went wrong killing: {proc.pid}")
                self.logger.error(e)
                remaining_processes.append(proc)
        self.processes = remaining_processes

        if simulation_obj is not None:
            try:
                simulation_obj.stop()
                self.logger.warning(
                    f"Simulation with simulation_id: {self.IOs._simulation_id} has finished at {dt.datetime.now()}"
                )
            except Exception as e:
                self.logger.error(f"Error stopping simulation: {e}")
                pass

    def reset(self):
        self.clean_processes()
        self.automator_setup = False
        self.s0 = time.time()
        self.task_name = "task"
        self.task_running = False
        self.error_detected = False

    def clean_folder(
        self, path_to_folder, list_of_files_to_keep=[], list_of_files_to_delete=None
    ):
        # Check folder exists
        try:
            path_to_folder = Path(path_to_folder)
            if not path_to_folder.exists():
                print(f"Folder {path_to_folder} does not exist, let's create it")
                path_to_folder.mkdir(parents=True)
        except Exception as e:
            print(f"Error: {e}")

        if list_of_files_to_delete is None:
            files_to_delete = [f for f in path_to_folder.iterdir() if f.is_file()]
        else:
            files_to_delete = list_of_files_to_delete

        files_to_delete = [
            f for f in files_to_delete if f.name not in list_of_files_to_keep
        ]

        if not files_to_delete:
            print(f"Folder {path_to_folder}/: No files to delete")
        else:
            print(f"In {path_to_folder}/:")
            for file in files_to_delete:
                try:
                    print(f"DELETING: {file.name}")
                    os.remove(file)
                except OSError as e:
                    print(f"CANNOT FIND: {e.filename}")

    def setup_automator(self, IOmodule, **kw_args):
        try:
            self.IOs = IOmodule
            input_topic = kw_args.get("input_topic", None)
            if input_topic is None:
                self.logger.warning("Input Topic not provided, using default")
                input_topic = self.IOs.service_topic("automation", "input")
            if input_topic != self.input_topic:
                self.input_topic = input_topic
                self.logger.info(
                    f"Setting up Automator with input_topic: {input_topic}"
                )

                _ = self.IOs._gapps.subscribe(self.input_topic, self.on_message)
            else:
                self.logger.debug(
                    f"Automator already setupo for topic: {self.input_topic}"
                )
            self.automator_setup = True
        except Exception as e:
            self.logger.error(f"Error: {e}")

    def send_command(self, topic, command: str, opts: dict = {}):
        if not self.automator_setup:
            self.logger.error("Automator not setup, cannot send command")
            return
        self.logger.info(f"Sending command: {command}")
        message = {"command": command}
        if opts:
            message.update(opts)
            self.logger.debug(f"Command options: {opts}")
        self.IOs.send_gapps_message(topic, message)

    def on_message(self, headers, message):
        if not self.automator_setup:
            self.logger.error("Automator not setup, cannot receive command")
            return
        self.logger.debug(f'Received message on topic: {headers["destination"]}')
        self.logger.debug(f"{message}")
        if self.input_topic is None or self.input_topic in headers["destination"]:
            command = message["command"]
            if command == "start_task":
                task_name = message.get("task_name", "task")
                self.start_task(task_name=task_name)
            elif command == "stop_task":
                self.stop_task()
            elif command == "error":
                self.error()

    def running(self):
        if not self.automator_setup:
            self.logger.error("Automator not setup, cannot receive command")
            return False
        return self.task_running

    def all_good(self):
        if not self.automator_setup:
            self.logger.error("Automator not setup, cannot receive command")
            return False

        return not self.error_detected

    def start_task(self, task_name: str = "task"):
        if not self.automator_setup:
            self.logger.error("Automator not setup, cannot receive command")
            return
        self.s1 = time.time()
        self.task_name = task_name
        self.task_running = True
        self.logger.info(f"Starting {task_name}")

    def stop_task(self, *args, **kw_args):
        if not self.automator_setup:
            self.logger.error("Automator not setup, cannot receive command")
            return
        if len(args) > 0:
            message = str(args[0])
            self.logger.warning(f"Stopping {self.task_name} with message: {message}")
        bypass_stats = kw_args.get("bypass_stats", False)
        if not bypass_stats:
            s2 = time.time()
            self.logger.log(
                self.TIMER_LVL, f"Completed {self.task_name} in {(s2 - self.s1):.2f} s."
            )
            self.logger.info(f"Total time since start: {(s2 - self.s0):.2f} s.")
            time.sleep(1)
        self.task_name = ""
        self.task_running = False

    def error(self):
        if not self.automator_setup:
            self.logger.error("Automator not setup, cannot receive command")
            return
        self.logger.error(f"Error Detected during {self.task_name}. Raising Exception.")
        self.error_detected = True
        self.task_running = False

    def wait_for_task(self):
        if not self.automator_setup:
            self.logger.error("Automator not setup, cannot receive command")
            return
        while self.task_running:
            time.sleep(1)
        else:
            if self.error_detected:
                raise Exception("Error Detected during task execution.")
