#!/usr/bin/env python3
import numpy as np 

A = np.array([[56.0, 0.0, 4.4, 68.0],   # Calories For Carbs
             [1.2, 104.0, 52.0, 8.0],   # Calories For Fat
             [1.8, 135.0, 99.0, 0.9]])  # Calories For Proteins
##           Apple, Beef, Eggs, Potatoes   Per 100g
total_cals = A.sum(axis = 0)
percentage = 100 * A /total_cals.reshape(1,A.shape[1])
