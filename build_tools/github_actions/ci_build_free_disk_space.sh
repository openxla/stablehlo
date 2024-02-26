#!/usr/bin/env bash
# Copyright 2024 The StableHLO Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script is used to free up disk space on the CI system.
# Specifically for github actions there is a bunch of wasted space for pre-installed
# software that we don't need. This script is used to remove those packages and free
# up that space.
# Inspiration from:
# https://github.com/jlumbroso/free-disk-space/blob/main/action.yml
# https://github.com/SeleniumHQ/selenium/blob/trunk/scripts/github-actions/free-disk-space.sh

echo "Freeing up disk space on GitHub runner"

echo "Listing 100 largest packages"
dpkg-query -Wf '${Installed-Size}\t${Package}\n' | sort -n | tail -n 100

# Print available space on all mounts
df -H

echo "Removing large packages"
sudo apt-get remove -y '^dotnet-.*'
sudo apt-get remove -y '^aspnetcore-.*'
sudo apt-get remove -y '^llvm-.*'
sudo apt-get remove -y 'php.*'
sudo apt-get remove -y 'haskell.*'
sudo apt-get remove -y '^mongodb-.*'
sudo apt-get remove -y '^mysql-.*'
sudo apt-get remove -y azure-cli google-cloud-sdk hhvm \
                       powershell mono-devel libgl1-mesa-dri \
                       google-chrome-stable firefox powershell \
                       libgl1-mesa-dri
sudo apt-get autoremove -y
sudo apt-get clean
df -H

echo "Removing large directories"

sudo rm -rf /usr/share/dotnet/
sudo rm -rf /usr/local/graalvm/
sudo rm -rf /usr/local/.ghcup/
sudo rm -rf /usr/local/share/powershell
sudo rm -rf /usr/local/share/chromium
sudo rm -rf /usr/local/lib/android
sudo rm -rf /usr/local/lib/node_modules
sudo rm -rf /opt/ghc
sudo rm -rf /usr/local/.ghcup

df -H

echo "Removing docker images"
sudo docker image prune --all --force

df -H
