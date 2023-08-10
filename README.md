# Matheuristic-for-CVRP-SDSW

This repository contains the Python code and data used for publication "Vehicle routing with stochastic demand, service and waiting times - The case of food bank collection problems" by Meike Reusken, Gilbert Laporte, Sonja U. K. Rohmer, and Frans Cruijssen.

## Instructions
The folder 'Data' contains the real-life data from Food Bank Zevenaar (Gelderland, the Netherlands), Food Bank Eemsdelta (Groningen, the Netherlands) and Moisson Montreal (Québec, Canada) as well as the random data proposed by Solomon (1987) that was used for this study.
The folder 'Output' is the location to which results are stored. 

To operate the matheuristic on the random data, run: Random_instances.py. 
To operate the matheuristic on the real-life data, run: Real_instances.py.

The python scripts in this repository contain the following content.

| Script name | Content |
|:-----------------|:-----------------|
| Fixed_inputs.py | Setting of the fixed parameters |
| Functions.py| Definitions of functions for all subproblems |
| main.py| Uses the above two scripts and combines all subproblems into the matheuristic including iterative procedures |
| Random_instances.py | Execution of the matheuristic on 27 random instances |
| Real_instances.py | Execution of the matheuristic on nine real-life food bank instances |

For questions, contact m.c.d.reusken_1@tilburguniversity.edu.
