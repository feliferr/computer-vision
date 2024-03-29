
#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
 
"""Setup.py module for the workflow's worker utilities.
All the workflow related code is gathered in a package that will be built as a
source distribution, staged in the staging area for the workflow being run and
then installed in the workers when they start running.
This behavior is triggered by specifying the --setup_file command line option
when running the workflow for remote execution.
"""
 
# pytype: skip-file
 
from __future__ import absolute_import
from __future__ import print_function
 
import subprocess
from distutils.command.build import build as _build  # type: ignore
 
import setuptools
from setuptools import find_packages, setup
 
 
# This class handles the pip install mechanism.
class build(_build):  # pylint: disable=invalid-name
    """A build command class that will be invoked during package install.
  The package built using the current setup.py will be staged and later
  installed in the worker using `pip install package'. This class will be
  instantiated during install for this specific scenario and will trigger
  running the custom commands specified.
  """
 
    sub_commands = _build.sub_commands + [("CustomCommands", None)]
 
 
# Some custom command to run during setup. The command is not essential for this
# workflow. It is used here as an example. Each command will spawn a child
# process. Typically, these commands will include steps to install non-Python
# packages. For instance, to install a C++-based library libjpeg62 the following
# two commands will have to be added:
#
#     ['apt-get', 'update'],
#     ['apt-get', '--assume-yes', 'install', 'libjpeg62'],
#
# First, note that there is no need to use the sudo command because the setup
# script runs with appropriate access.
# Second, if apt-get tool is used then the first command needs to be 'apt-get
# update' so the tool refreshes itself and initializes links to download
# repositories.  Without this initial step the other apt-get install commands
# will fail with package not found errors. Note also --assume-yes option which
# shortcuts the interactive confirmation.
#
# Note that in this example custom commands will run after installing required
# packages. If you have a PyPI package that depends on one of the custom
# commands, move installation of the dependent package to the list of custom
# commands, e.g.:
#
#     ['pip', 'install', 'my_package'],
#
# TODO(BEAM-3237): Output from the custom commands are missing from the logs.
# The output of custom commands (including failures) will be logged in the
# worker-startup log.
CUSTOM_COMMANDS = [
    ['apt-get', 'update'],
    ['apt-get', '--assume-yes', 'install', 'libsm6','libxext6', 'libxrender1'],
]
 
 
class CustomCommands(setuptools.Command):
    """A setuptools Command class able to run arbitrary commands."""
 
    def initialize_options(self):
        pass
 
    def finalize_options(self):
        pass
 
    def RunCustomCommand(self, command_list):
        print("Running command: %s" % command_list)
        p = subprocess.Popen(
            command_list,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        stdout_data, _ = p.communicate()
        print("Command output: %s" % stdout_data)
        if p.returncode != 0:
            raise RuntimeError(
                "Command %s failed: exit code: %s" % (command_list, p.returncode)
            )
 
    def run(self):
        for command in CUSTOM_COMMANDS:
            self.RunCustomCommand(command)
 
setup(
    name='action_recognition',
    version='0.0.1',
    description='dataflow video action recognition',
    packages=find_packages(),
    install_requires=[
        "apache-beam[gcp]",
        "click",
        "gcloud",
        "Pillow",
        "pandas==1.1.5",
        "scikit-learn==0.24.2",
        "torch==1.9.0",
        "torchvision==0.10.0",
        "tqdm==4.62.1",
        "albumentations==1.0.3",
        "imutils==0.5.4",
        "scenedetect==0.5.1.1",
        "six==1.14.0",
        "google-cloud-storage",
        "google-auth-oauthlib",
        "google-auth",
    ],
    cmdclass={
        # Command class instantiated and run during pip install scenarios.
        'build': build,
        'CustomCommands': CustomCommands,
    }
)
