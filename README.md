# Evolutionary-Trading-Bot

Steps to reproduce results:

Clone repo and cd to project directory.  
Create and activate a virtual environment and download packages.

```console
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

# Run a tournament

```console
py tournament.py [size num_parents num_iterations mutation_probability dir]
```

After the tournament, the best strategies will be saved in a new directory named dir in the results directory as pkl files.

# Generate results

```console
py results.py
```

This generates a summary of the results in the results directory as csv. By default it searches for directories [sortino50, sortino100, sortino200, portfolio50, portfolio100, portfolio200] that contain the pkl files.

# Graph trade patterns

```console
py graph.py [train | test]
```

This generates the graphs of the trading patterns on the train or test data for the random strategy, simple strategy and the best strategy observed in the pkl files. Assumes the directories [sortino50, sortino100, sortino200, portfolio50, portfolio100, portfolio200] are present in the results directory.
