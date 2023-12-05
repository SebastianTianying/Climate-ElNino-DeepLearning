# Climate-ElNino-DeepLearning

CS269 Group Project 

Authors: Allen Cheung, Garrick Su, Tianying Zhu 

CS@UCLA

The primary goal of this work is to develop a robust machine learning model capable of predicting
the total number of severe storms and other weather-related disasters, such as flash floods, winter
storms, or excessive heat, during upcoming months in the contiguous United States.

With the world bracing for a strong El Niño year, understanding its potential effects is crucial
for a variety of stakeholders. From policy-makers gearing up for disaster management to marine
biologists studying potential disruptions in marine life, timely and accurate predictions can aid in
better preparation and mitigation strategies.

This study will focus specifically on incorporating historical data of past El Niño events and their
corresponding magnitude index (ENSO Index), ERA5 [7] weather data, as well as Storm Events data
in the U.S to output severe weather predictions with up to a 12-month lead time.
Our intended contributions are as follows:
• Aggregate relevant features from ERA5, storm event data, and ENSO Indices to create a
single dataset for severe weather event prediction in the contiguous United States
• Explore the efficacy of fine-tuning foundational climate models for the task of severe weather
event prediction compared to developing a specialized deep learning architecture
• Predict the number of extreme weather-related disasters in the upcoming months across the
contiguous United States with a spatial resolution of 0.5 degrees
• Evaluate the impact of including ENSO indices as an additional input feature, performing a
comprehensive ablation study with different combinations of ENSO indices


