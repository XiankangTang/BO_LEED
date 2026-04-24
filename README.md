This repository provides a reference implementation and links related to the research paper:

"Physics-informed automated surface reconstructing via low-energy electron diffraction based on Bayesian optimization"​ (arXiv:2604.04578).


This work introduces a novel framework for automating the quantitative analysis of Low-Energy Electron Diffraction (LEED) I(V) curves. The method integrates physics-based multiple scattering forward models directly into a trust-region Bayesian Optimization (BO) loop. This approach simultaneously optimizes structural and experimental parameters, enabling efficient, autonomous exploration of complex, non-convex parameter spaces for accurate surface structure determination.

Key Components

1. Citation of the Main Research Paper

If you use the methodology or findings from this research, please cite the following paper:

Physics-informed automated surface reconstructing via low-energy electron diffraction based on Bayesian optimization​

Xiankang Tang, Ruiwen Xie, Jan P. Hofmann, Hongbin Zhang

arXiv:2604.04578 [physics.comp-ph](2026)

https://arxiv.org/abs/2604.04578

2. Code Source

The underlying computational package for quantitative LEED analysis is ViPErLEED:

Repository:​ viperleed/viperleed
on GitHub

Description:​ ViPErLEED is an open-source Python package for quantitative LEED-I(V) analysis. It provides tools for measurement processing, calculation of theoretical I(V) curves, and optimization of structural models to fit experimental data.

Related Repositories:

https://github.com/viperleed/viperleed

3. Source of Example Data

The input structures, experimental I(V) curves, and optimization parameters for the two benchmark systems are sourced from the official ViPErLEED documentation:

Source: viperleed.calc documentation – "Examples" section

Ag(100)-(1×1): Used as a simple test case for structure optimization.

α-Fe₂O₃(1-102)-(1×1): Used as a complex oxide surface example.

URL: https://www.viperleed.org/stable/content/viperleed_calc.html

The data includes POSCAR(structure), EXPBEAMS.csv(experimental I(V)), and DISPLACEMENTS(parameter search ranges) files, which are essential for reproducing the Bayesian optimization results.
