# Modifications do Decision Trees to tackle class imbalacement #

## About ##

This project was developed in the context of the Machine Learning I course from the Bachelor's in Computer Science of the Faculty of Sciences of the University of Porto.

This project was developed by:

* Carla Henriques
* Jaime Cruz Ferreira
* Matheus Borges

## The idea ##

Taking a simple decision tree implementation, the idea was to tackle the poor performance in cases where the dataset presents a severe class imbalance.

The idea was to calculate the chosen class in the leaves taking into account the presence of that class in the original dataset. Three functions were used and a linear function presented the best results.

### Example: ###

If class 1 has 90% presence in the original dataset and class 2 has 10% presence, if in a leaf there are 10 examples -> 8 class 1/ 2 class 2, the decision would be class 2.
Using the linear function, class 2 would have a weight of 90 (conversely class 1's weight is 10), and 8 * 10 < 2 * 90.

