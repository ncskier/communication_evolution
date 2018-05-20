# Communication Evolution Project
Created by Brandon Walker

## Run Command Line:
`python3 evolution.py run [project_name] [one] [draw]`

`python3 evolution.py cont [project_name] [generation] [one] [draw]`

`python3 evolution.py vis [project_name] [generation] [one] [one_agent]`

`[one]` (bool) specifies if the simulation should be run with all agents using one nn.

`[one_agent]` (int) specifies only visualizing the simulation from one agent.

`[draw]` (bool) specifies whether the simulation should output a visualization (the world size is not currently initialized correctly when this option is used)

Unfortunately the parameters like population size, world size, number of generations, neural network type, etc. are all hardcoded and must be changed manually in the source code.

Apologies; the project is a bit messy right now.