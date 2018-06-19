.. pySPC documentation master file, created by
   sphinx-quickstart on Wed Jun 13 14:31:25 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pySPC's documentation!
=================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:


MVC pattern
===========
pySPC uses a Model View Controller `(MVC) pattern  <https://en.wikipedia.org/wiki/Model%E2%80%93view%E2%80%93controller>`_

The Model (which is contained in the repertory named core) is the heart of the software.

It is *totally independant*.

You can (and sometimes you better should) use it separately, typically with scripts (using it a jupyter notebook is a very efficient way to document ).

The Model and the view are here for the Graphical User Interface (GUI).

The Core (Model in MVC)
-----------------------

The GUI (View and Controller in MVC)
------------------------------------

API documentation
=================


The Core
--------

Inherintance graph :

.. inheritance-diagram:: Measurement.Measurements FCS.CorrelationMeasurement FCS.FCSMeasurements
   :parts: 2



.. autoclass:: Data.Data
    :members:

.. autoclass:: Experiment.Experiment
    :members:

.. autoclass:: Results.Results
    :members:

.. autoclass:: ExpParam.Experiment_param
    :members:

.. autoclass:: Measurement.Measurements
    :members:

.. autoclass:: FCS.CorrelationMeasurement
    :members:

.. autoclass:: FCS.FCSMeasurements
    :members:





The Controller
--------------

.. autoclass:: Controller.Controller
    :members:




The View
--------


Tests
=====

.. math::

  W^{3\beta}_{\delta_1 \rho_1 \sigma_2} \approx U^{3\beta}_{\delta_1 \rho_1}

.. note::

   Note test

.. warning::
    warning test



TODO
====

.. todolist::

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
