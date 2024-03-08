---
title:  Model Deployment
subtitle: Production
author: Jesús Calderón
---


# Introduction

## Agenda

::::::{.columns}
:::{.column}

**6.1 Model Deployment and Prediction Service **
    
+ ML Deployment Myths and Anti-Patterns
+ Batch Prediction vs Online Prediction

:::
:::{.column}

**6.2 Explainability Methods**

+ Partial Dependence Plots
+ Permutation Importance
+ Shap Values

:::
::::::


## Slides, Notebooks, and Code

::::::{.columns}
:::{.column}

+ These notes are based on Chapter 7 of [*Designing Machine Learning Systems*](https://huyenchip.com/books/), by [Chip Huyen](https://huyenchip.com/).

:::
:::{.column}

**Notebooks**

+ `./notebooks/production_5_model_development.ipynb`


**Code**

+ `./src/credit_experiment_*.py`

:::
::::::

# Our Reference Architecture

## The Flock Reference Architecture

![Agrawal et al (2019)](../img/flock_ref_arhitecture.png)


# Deployment

## Deployment

Make it available to user. Putting it in front of users.
RMarkDown for reports. Will give graphs and code. PowerBI is more interactive. Pushing results to database.
We've seend how to take a Panadas Dataframe and push it directly to a database.

::::::{.columns}
:::{.column}

+ Deploying a model is to make it useable by allowing users to interact with it through an app or by using its results for a purpose in a data product (BI visuals, reports, data views).

Have a development environment closely resemble the production env. Bring your dev as close to prod env as possible.
If you linux in stage, should have in prod. This will minimize discrepencies and makes your deployments more robust. We are catering to the production environment that we want to meet.
Dockerize Spark Image and build clusters using containers. 
+ Deployment is a transition of development to a production environment. 


:::
:::{.column}

+ There is a wide range of production environments, from BI to live applications serving millions of users.
+ Engaging with users in formal or informal feedback conversations is helpful, although only sometimes possible.


:::
::::::


## Deployment Myths and Anti-Patterns

::::::{.columns}
:::{.column}

**1. You only deploy one or two ML models at a time**
Have a branching strategy. Never work off main.
Use tags in your commits. 
+ Infrastructure should support many models, not only a few.
+ Many models can interact, and we also need a way of mapping these interactions.
+ Ride sharing app: 

    - 10 models: ride demand, driver availability, estimated time of arrival, dynamic pricing, fraud, churn, etc.
    - 20 countries.


:::
:::{.column}

Section 7 of Jupyter notebooks, tests have been left for my use. 
**2. If we don't do anything, model performance stays the same**

+ Software does not age like fine wine.
+ Data distribution shifts: when the data distribution in the trained model differs from the distribution during testing.

:::
::::::


## Deployment Myths and Anti-Patterns (cont.)

::::::{.columns}
:::{.column}


**3. You won't need to update your models as much**

+ Model performance decays over time.
+ Deploy should be easy:

    - The development environment should resemble the production environment as closely as possible.

Production environment should be provisioned with code. Parquet files send to cold storage. 
    - Infrastructure should be easier to rebuild than to repair.
Keep it slim. Add changes little by little. Test and commit it. You will have incremental changes that continue to work. 
    - Small incremental and frequent changes.


:::
:::{.column}

**4. Most ML engineers don't need to worry about scale**

+ Scale means different things to different applications.
If volumous data, take smaller batches. Choose tools that allow you to work in batches. Pargquet, Spark, Dask. Machine learning library specifically designed for Dask. Up to 50M observations works with Dask.
+ Number of users, availability, speed or volume of data.

:::
::::::

## Batch Prediction Vs Online Prediction

::::::{.columns}
:::{.column}


**Online Prediction**

- Predictions are generated and returned as soon as requests for these predictions arrive.
- Also known as on-demand prediction.
Restful has too do with characteristics of state. HTTP protocol implements asll the verbs and actions to run API. WIll return JSON file containing label and additional information about prodicitons. FastAPIs. Python libraries to create a harness for API. Flask. Django.
- Traditionally, requests are made to a prediction service via a RESTful API.
- When requests are made via HTTP, online prediction is known as *synchronous prediction*. 


:::
:::{.column}

**Batch Prediction**

On a time schedule. Every morning triggers a predictions.
- Predictions are generated periodically or whenever triggered.
Stored in memory or brokerage service in the cloud.
- Predictions are stored in SQL tables or in memory. They are later retrieved as needed.
Asynchronous doesn't happen in real time.
- Batch prediction is also known as asynchronous prediction.

:::
::::::


## Model Prediction Service
Prediction Service is what we produce. 
::::::{.columns}
:::{.column}
Some that is producitng a response. Has a listener listening for requests. 
Three types of model prediction or inference service:

Simplest batch predictions. 
+ Batch prediction: uses only batch features.
+ Online prediction that uses only batch features (e.g., precomputed embeddings).
+ Online streaming prediction: uses batch features and streaming features.

:::
:::{.column}

![Batch Prediction (based on Huyen 2021)](./img/batch_prediction.png)

:::
::::::


## Model Prediction Service (cont.)

::::::{.columns}
:::{.column}

![Online Prediction (based on Huyen 2021)](./img/online_prediction.png)
:::
:::{.column}

![Streaming Prediction (based on Huyen 2021)](./img/streaming_prediction.png)

:::
::::::


## References

+ Agrawal, A. et al. "Cloudy with a high chance of DBMS: A 10-year prediction for Enterprise-Grade ML." arXiv preprint arXiv:1909.00084 (2019).
+ Huyen, Chip. "Designing machine learning systems." O'Reilly Media, Inc.(2021).
