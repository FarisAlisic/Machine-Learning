This is a Machine Learning project in which we use 
Linear Regression built from scratch to learn and 
predict the phone price data.

The dataset I used has been retrived from: 
https://www.kaggle.com/datasets/rkiattisak/mobile-phone-price

Kaggle is a popular website with all kinds of datasets,
that are usualy used for machine learning.

The initial dataset was not clean and it had duplicates,
therefore I cleaned it and made a new dataset with all
integers and one string column indicating the Brand (which
had a big effect to the price so it had to stay), the brand
column was processed in a special way.

Results:

Training R² Score: 0.8316301314599689
Testing R² Score: 0.8619881666103291

The model preformed well despite the small and not detailed 
enough dataset. The main part i was interested about was 
the user test part. The inputs can be not real life inputs 
and the model still scores well therefore I think it is a 
success. However in the future I will build up on this Project
with different models and more advanced models, but the first
improvement I would like to do is find a better dataset because 
this had flaws.
