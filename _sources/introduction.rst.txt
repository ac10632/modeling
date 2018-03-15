=============================
Python Modeling Tools
=============================

:Author:
   William Alexander

   worksprogress1@gmail.com


Introduction
############

These tools are ones that I have found to be useful over the years.  In making the move to Python from
SAS and R, it seemed like a good time to assemble them in a single spot.  A lot of these exist elsewhere in other
modules available in Python.  However, there's nothing quite like customizing what one uses everyday to suit one's exact needs.


The functionality provided in this module is:

- Fitting Generalized Linear Models (glm).
- Fitting Generalized Additive Models (gam).
- Models are specified in a syntax similar to R.
- The model formula can include specifications for linear splines and categorical variables.
- A separate class that handles the creation of the design matrix.
- Diagnostic plots.

Package contents:

- module data_class:

  - class DataClass
  - class DataError
  - class ModelSpecificationError
    
- module functions:
  
  - function linear_splines_basis1
  - function linear_splines_basis2
  - function categorical_to_design
  - function ks_calculate
  - function decile_plot
    
- module glm: class glm
- module gam: class gam  
  
  
:Example Imports:

   - from modeling_tools.functions import decile_plot
   - from modeling_tools.data_class import DataClass
   - from modeling_tools.glm import glm

The code is available `here <https://github.com/ac10632/modeling>`_.

It may be installed via pip.  For example, on Ubuntu:

pip3 install git+https://git@github.com/ac10632/modeling

