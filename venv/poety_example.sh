# Poetry
# So let's check out another tool called Poetry, 
# and we'll do the same things, start a project, install some dependencies, and run our code. 

# Here we are at its home page, and here in the documentation there's a line of code that I can use to install it. 
# https://python-poetry.org/
# https://python-poetry.org/docs/#installing-with-the-official-installer
[user]@-3D05SQ3:~/python/managing-python-packages-virtual-environments/demos$ curl -sSL https://install.python-poetry.org | python3 -
Retrieving Poetry metadata

# Welcome to Poetry!

This will download and install the latest version of Poetry,
a dependency and package manager for Python.

It will add the `poetry` command to Poetry's bin directory, located at:

/home/[user]/.local/bin

You can uninstall at any time by executing this script with the --uninstall option,
and these changes will be reverted.

Installing Poetry (1.4.1): Done

Poetry (1.4.1) is installed now. Great!

You can test that everything is set up by executing:

`poetry --version`

[user]@-3D05SQ3:~/python/managing-python-packages-virtual-environments/demos$ poetry --version
Poetry (version 1.4.1)

[user]@-3D05SQ3:~/python/managing-python-packages-virtual-environments/demos$ poetry self update
Updating Poetry version ...

Using version ^1.4.1 for poetry

Updating dependencies
Resolving dependencies... Downloading https://files.pythonhosted.org/packages/ae/2a/7ad62b2c56e71c6330fc35cfd5813376e788Resolving dependencies... Downloading https://files.pythonhosted.org/packages/c8/22/9460e311f340cb62d26a38c419b1381b8593Resolving dependencies... Downloading https://files.pythonhosted.org/packages/e5/ca/1172b6638d52f2d6caa2dd262ec4c811ba59Resolving dependencies... Downloading https://files.pythonhosted.org/packages/e5/ca/1172b6638d52f2d6caa2dd262ec4c811ba59Resolving dependencies... Downloading https://files.pythonhosted.org/packages/e5/ca/1172b6638d52f2d6caa2dd262ec4c811ba59Resolving dependencies... Downloading https://files.pythonhosted.org/packages/ac/ab/a19748648244e7012cf3d46c65a2b84de94cResolving dependencies... Downloading https://files.pythonhosted.org/packages/22/a6/858897256d0deac81a172289110f31629fc4Resolving dependencies... Downloading https://files.pythonhosted.org/packages/ed/35/a31aed2993e398f6b09a790a181a7927eb14Resolving dependencies... Downloading https://files.pythonhosted.org/packages/fc/34/3030de6f1370931b9dbb4dad48f6ab1015abResolving dependencies... Downloading https://files.pythonhosted.org/packages/71/4c/3db2b8021bd6f2f0ceb0e088d6b2d4914767Resolving dependencies... Downloading https://files.pythonhosted.org/packages/76/cb/6bbd2b10170ed991cf64e8c8b85e01f2fb38Resolving dependencies... Downloading https://files.pythonhosted.org/packages/85/01/e2678ee4e0d7eed4fd6be9e5b043fff9d22dResolving dependencies... (11.3s)

Writing lock file

No dependencies to install or update
[user]@-3D05SQ3:~/python/managing-python-packages-virtual-environments/demos$ poetry --version
Poetry (version 1.4.1)


# So I'm just going to copy/paste that into my terminal, and what you immediately notice is that, 
# in my opinion at least, Poetry really has a friendly user interface. 
# It immediately gives some info about what the installer does and it added some code in my .profile actually, so I don't even have to do that myself. 

# By default, Poetry is installed into a platform and user-specific directory:
# ~/Library/Application Support/pypoetry on MacOS.
# ~/.local/share/pypoetry on Linux/Unix.
# %APPDATA%\pypoetry on Windows.

# If this directory is not present in your $PATH, you can add it in order to invoke Poetry as poetry.
# Alternatively, the full path to the poetry binary can always be used:
# ~/Library/Application Support/pypoetry/venv/bin/poetry on MacOS.
# ~/.local/share/pypoetry/venv/bin/poetry on Linux/Unix.
# %APPDATA%\pypoetry\venv\Scripts\poetry on Windows.
# $POETRY_HOME/venv/bin/poetry if $POETRY_HOME is set.

[user]@-3D05SQ3:~/python/managing-python-packages-virtual-environments/demos$ env |grep 'HOME'
HOME=/home/[user]
WORKON_HOME=/home/[user]/python/virtualenvs
PROJECT_HOME=/home/[user]/python/panTestProjects

[user]@-3D05SQ3:~/python/managing-python-packages-virtual-environments/demos$ $PATH
-bash: /home/[user]/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/mnt/c/Program: No such file or directory


[user]@-3D05SQ3:~/python/managing-python-packages-virtual-environments/demos$ $HOME
-bash: /home/[user]: Is a directory
[user]@-3D05SQ3:~/python/managing-python-packages-virtual-environments/demos$ cd /home/[user]
[user]@-3D05SQ3:~$ ls
python
[user]@-3D05SQ3:~$ ls -a
.   .bash_history  .bashrc  .config     .local       .profile         .sudo_as_admin_successful  python
..  .bash_logout   .cache   .landscape  .motd_shown  .python_history  .virtualenvs
[user]@-3D05SQ3:~$ cd ./local
-bash: cd: ./local: No such file or directory
[user]@-3D05SQ3:~$ cd .local/
[user]@-3D05SQ3:~/.local$ ls
bin  lib  share
[user]@-3D05SQ3:~/.local$ cd bin
[user]@-3D05SQ3:~/.local/bin$ ls
f2py   f2py3.8  pip   pip3.8  pipenv-resolver  pybabel     virtualenv-clone      virtualenvwrapper_lazy.sh
f2py3  pbr      pip3  pipenv  poetry           virtualenv  virtualenvwrapper.sh
[user]@-3D05SQ3:~/.local/bin$ ls  -l
total 48
-rwxr-xr-x 1 [user] [user]   216 Jan 20 22:14 f2py
-rwxr-xr-x 1 [user] [user]   216 Jan 20 22:14 f2py3
-rwxr-xr-x 1 [user] [user]   216 Jan 20 22:14 f2py3.8
-rwxr-xr-x 1 [user] [user]   211 Feb  9 09:24 pbr
-rwxr-xr-x 1 [user] [user]   221 Feb  2 12:49 pip
-rwxr-xr-x 1 [user] [user]   221 Feb  2 12:49 pip3
-rwxr-xr-x 1 [user] [user]   221 Feb  2 12:49 pip3.8
-rwxr-xr-x 1 [user] [user]   203 Feb 15 12:57 pipenv
-rwxr-xr-x 1 [user] [user]   214 Feb 15 12:57 pipenv-resolver
lrwxrwxrwx 1 [user] [user]    49 Mar 27 09:23 poetry -> /home/[user]/.local/share/pypoetry/venv/bin/poetry
-rwxr-xr-x 1 [user] [user]   222 Feb  3 09:44 pybabel
-rwxr-xr-x 1 [user] [user]   238 Feb  9 09:24 virtualenv
-rwxr-xr-x 1 [user] [user]   214 Feb  9 09:24 virtualenv-clone
-rwxr-xr-x 1 [user] [user] 41703 Feb  9 09:24 virtualenvwrapper.sh
-rwxr-xr-x 1 [user] [user]  2210 Feb  9 09:24 virtualenvwrapper_lazy.sh
[user]@-3D05SQ3:~/.local/bin$ cd  /home/[user]/.local/share/pypoetry/venv
[user]@-3D05SQ3:~/.local/share/pypoetry/venv$

# And now I can start working with Poetry. Let's start a new project. Let's call my project myproject. So, this creates a new folder. Let's take a look. 
[user]@-3D05SQ3://home/[user]/python/panTestProjects$ ls
new_project  py3_8_numpy1_22_pandas1_4
[user]@-3D05SQ3://home/[user]/python/panTestProjects$ poetry new poetry_new_project
Created package poetry_new_project in poetry_new_project
[user]@-3D05SQ3://home/[user]/python/panTestProjects$ ls
new_project  poetry_new_project  py3_8_numpy1_22_pandas1_4
[user]@-3D05SQ3://home/[user]/python/panTestProjects$

# And Poetry has generated a whole project skeleton for me. 
# There's an empty README file, 
# a myproject folder, which is actually a Python package with an empty init file in there, so that's meant to hold the code from my project, 
# a folder for unit tests, 
# and the pyproject.toml, which is where my dependencies will go. 

# But let's start by installing our requirements. And slightly surprisingly, I have to use the command poetry add instead of install. 
# Apart from that, it works pretty straightforward, and again we see that the output is very user friendly. 
# Now, let me just copy our familiar script into the package folder, 
# and to run our code, we use the same approach as with Pipenv, which is to say Poetry run Python, but the difference is that the location of the script is now slightly different because it's inside the package. 
# Or, by the way, we can start the shell with an active environment, and because in this case we actually have a package, I can run Python and import the module from the package. And this, again, runs our code. 
# Very well. Let me exit Python here and exit the shell. 
# Again, I'm not using deactivate here, but exit, 

[user]@D05SQ3://home/panxi/python/panTestProjects/poetry_new_project$ poetry add babel
Creating virtualenv poetry-new-project-GIbPIhWT-py3.8 in /home/panxi/.cache/pypoetry/virtualenvs
Using version ^2.12.1 for babel

Updating dependencies
Resolving dependencies... Downloading https://files.pythonhosted.org/packages/a3/21/0ffac8dacd94d20f4d1ceaaa91cf28ad94ecResolving dependencies... Downloading https://files.pythonhosted.org/packages/a3/21/0ffac8dacd94d20f4d1ceaaa91cf28ad94ecResolving dependencies... Downloading https://files.pythonhosted.org/packages/a3/21/0ffac8dacd94d20f4d1ceaaa91cf28ad94ecResolving dependencies... (1.2s)

Writing lock file

Package operations: 2 installs, 0 updates, 0 removals

  • Installing pytz (2023.2)
  • Installing babel (2.12.1)
[user]@D05SQ3://home/panxi/python/panTestProjects/poetry_new_project$

[user]@D05SQ3://home/panxi/python/panTestProjects/poetry_new_project$ cp /home/panxi/python/managing-python-packages-virtual-environments/demos/babel_demo.py poetry_new_project/
[user]@D05SQ3://home/panxi/python/panTestProjects/poetry_new_project$ ls
README.md  poetry.lock  poetry_new_project  pyproject.toml  tests
[user]@D05SQ3://home/panxi/python/panTestProjects/poetry_new_project$ poetry run python poetry_new_project/babel_demo.py
In the Netherlands we write 12,345,678 as 12.345.678
[user]@D05SQ3://home/panxi/python/panTestProjects/poetry_new_project$ poetry shell
Spawning shell within /home/panxi/.cache/pypoetry/virtualenvs/poetry-new-project-GIbPIhWT-py3.8
. /home/panxi/.cache/pypoetry/virtualenvs/poetry-new-project-GIbPIhWT-py3.8/bin/activate
[user]@D05SQ3://home/panxi/python/panTestProjects/poetry_new_project$ . /home/panxi/.cache/pypoetry/virtualenvs/poetry-new-project-GIbPIhWT-py3.8/bin/activate
(poetry-new-project-py3.8) [user]@D05SQ3://home/panxi/python/panTestProjects/poetry_new_project$ python
Python 3.8.10 (default, Nov 14 2022, 12:59:47)
[GCC 9.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> from poetry_new_project import babel_demo
In the Netherlands we write 12,345,678 as 12.345.678
>>> exit
Use exit() or Ctrl-D (i.e. EOF) to exit
>>> exit()
(poetry-new-project-py3.8) [user]@D05SQ3://home/panxi/python/panTestProjects/poetry_new_project$ exit
exit
[user]@D05SQ3://home/panxi/python/panTestProjects/poetry_new_project$

# and now taking a look at the dependencies file by project.toml, this is actually quite similar to the Pipfile. 
# We see a section for dependencies and one for development time dependencies, and there's a separate section, tool.poetry, which lets you fill in some metadata about your project. 
# There's also a build‑system section, which is there because poetry will actually create a package for you that you can upload to PyPI, something that Pipenv doesn't do. 
# By the way, we have a poetry.lock file too to make the build deterministic and repeatable, just like with pipfile.lock. 

[user]@D05SQ3://home/panxi/python/panTestProjects/poetry_new_project$ ls
README.md  poetry.lock  poetry_new_project  pyproject.toml  tests
[user]@D05SQ3://home/panxi/python/panTestProjects/poetry_new_project$ cat pyproject.toml
[tool.poetry]
name = "poetry-new-project"
version = "0.1.0"
description = ""
authors = ["Your Name <you@example.com>"]
readme = "README.md"
packages = [{include = "poetry_new_project"}]

[tool.poetry.dependencies]
python = "^3.8"
babel = "^2.12.1"


[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
[user]@D05SQ3://home/panxi/python/panTestProjects/poetry_new_project$ ls
README.md  poetry.lock  poetry_new_project  pyproject.toml  tests
[user]@D05SQ3://home/panxi/python/panTestProjects/poetry_new_project$ cat poetry.lock
# This file is automatically @generated by Poetry 1.4.1 and should not be changed by hand.

[[package]]
name = "babel"
version = "2.12.1"
description = "Internationalization utilities"
category = "main"
optional = false
python-versions = ">=3.7"
files = [
    {file = "Babel-2.12.1-py3-none-any.whl", hash = "sha256:b4246fb7677d3b98f501a39d43396d3cafdc8eadb045f4a31be01863f655c610"},
    {file = "Babel-2.12.1.tar.gz", hash = "sha256:cc2d99999cd01d44420ae725a21c9e3711b3aadc7976d6147f622d8581963455"},
]

[package.dependencies]
pytz = {version = ">=2015.7", markers = "python_version < \"3.9\""}

[[package]]
name = "pytz"
version = "2023.2"
description = "World timezone definitions, modern and historical"
category = "main"
optional = false
python-versions = "*"
files = [
    {file = "pytz-2023.2-py2.py3-none-any.whl", hash = "sha256:8a8baaf1e237175b02f5c751eea67168043a749c843989e2b3015aa1ad9db68b"},
    {file = "pytz-2023.2.tar.gz", hash = "sha256:a27dcf612c05d2ebde626f7d506555f10dfc815b3eddccfaadfc7d99b11c9a07"},
]

[metadata]
lock-version = "2.0"
python-versions = "^3.8"
content-hash = "48334098cdfdbc0af08eddfbb5d850f47eb51160f59e0c445f02508430c03ab0"
[user]@D05SQ3://home/panxi/python/panTestProjects/poetry_new_project$

# Now, creating a project for a different Python version is not as easy with Poetry. 
# The best way to make this work is with a separate tool called pyenv, and that is beyond the scope of this course. 
# So, that should give you a bit of an impression of both Pipenv and Poetry.

# I really think they're quite similar, although they have different ideas about things like what a project is and how to do packaging.