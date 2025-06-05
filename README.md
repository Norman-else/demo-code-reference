# demo-code-reference

This repository contains a small demo for evaluating different inventory
replenishment algorithms. The system generates simulated demand data,
executes multiple algorithms, and produces a simple SVG bar chart showing
service level (percentage of days without stockouts) for each algorithm.

## Requirements

The code uses only the Python standard library, so no additional
installations are necessary.

## Running the Demo

```bash
python3 main.py
```

Running the script will create a file `service_level.svg` in the project
folder and print metrics for each algorithm to the console.

