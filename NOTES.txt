
Purpose
=======

Program to take the retinal classification CSV from the Zooniverse
export and turn it into a CSV format that is useful for comparisons.

How it works
============

The program export_converter.py can be run as::

   python export_converter.py PATH_TO_EXPORT_CSV_FILE


(You will need to have `pandas <http://pandas.pydata.org/>`_
installed.)


This will produce the following file outputs for each of the tasks:
 - fovea_data.csv
 - optic_nerve_box_data.csv
 - cup_disk_data.csv 
 - notch_haemorrhage_marks.csv
 
The program works (see main function) in the following steps:
 - Load the Zooniverse CSV export data with Pandas
 - Filter to the required workflows as specified in VALID_WORKFLOWS.
 - Setup 'accumulators' to process the data. Each accumulator is
   called to setup (using the whole dataframe), on each row with a
   user and subject key and to finish.
 - The accumulators typically reach into the dataframe and parse the
   embedded annotation row in the handle_row() call. See for example
   AccumulateOpticNerveBox.handle_row().

