# Data Science Ethics Checklist

[![Deon badge](https://img.shields.io/badge/ethics%20checklist-deon-brightgreen.svg?style=popout-square)](http://deon.drivendata.org/)

## A. Data Collection
 - [x] **A.1 Informed consent**: If there are human subjects, have they given informed consent, where subjects affirmatively opt-in and have a clear understanding of the data uses to which they consent?
 * No, our research does not include any data involving human subjects.

 - [x] **A.2 Collection bias**: Have we considered sources of bias that could be introduced during data collection and survey design and taken steps to mitigate those?
 * When working with raster and vector data, an issue may arise when calculating values in raster data using vector boundaries. Specifically, there are cells that lie on the boundary line (Figure 1). To address this, we utilized the Shapely package in Python, which allows us to classify cells based on their overlap with the boundary. Cells that have 50% or more of their area inside the boundary are considered inside, while those with less than 50% are considered outside. Consequently, we excluded the values of cells classified as outside from our dataset. for further analysis we could consider incorporating Dasymetric method.
 

![Figure 1. raster and vector data](https://jo-wilkin.github.io/GEOG0030/coursebook/images/w9/rtovector.png)

 - [x] **A.3 Limit PII exposure**: Have we considered ways to minimize exposure of personally identifiable information (PII) for example through anonymization or not collecting information that isn't relevant for analysis?
 * The data utilized in our study were already deidentified or aggregated at a higher level to ensure privacy and confidentiality.

 - [x] **A.4 Downstream bias mitigation**: Have we considered ways to enable testing downstream results for biased outcomes (e.g., collecting data on protected group status like race or gender)?
 *  While our collected data doesn't include gender or race information directly, we plan looking at whether the modelshowed consistency across geographic regions (census tracts in this case). The fairness of the model across these regions can be assessed by calculating the true positive rate across census tracts to see if significant differences exist.

## B. Data Storage
 - [x] **B.1 Data security**: Do we have a plan to protect and secure data (e.g., encryption at rest and in transit, access controls on internal users and third parties, access logs, and up-to-date software)?
 * We plan to upload data in a priviately shared repository among the investigator

 - [x] **B.2 Right to be forgotten**: Do we have a mechanism through which an individual can request their personal information be removed?
 * This may not be applied to our research since did not have data that has individual data

 - [x] **B.3 Data retention plan**: Is there a schedule or plan to delete the data after it is no longer needed?
 * No, because the input and output of our project is aiming on annual average values, these informatin will be useful as historical records when new datasets come up. 

## C. Analysis
 - [x] **C.1 Missing perspectives**: Have we sought to address blindspots in the analysis through engagement with relevant stakeholders (e.g., checking assumptions and discussing implications with affected communities and subject matter experts)?
 * We thoroughly reviewed the metadata of each dataset to ensure a clear understanding of the data creation process and the meaning of each individual variable.

 - [x] **C.2 Dataset bias**: Have we examined the data for possible sources of bias and taken steps to mitigate or address these biases (e.g., stereotype perpetuation, confirmation bias, imbalanced classes, or omitted confounding variables)?
 * Missing data: At first, we didn't take building specific information (building year for example) into account. This could import biases. Then we added building specific datasets in census tract level.
 * Correlation: For the first version of processed datasets, there are 7 features, where there is a comparatively high correlation among them. Then we applied PCA dimension reduction on these features to keep core information and reduce correlation at the same time.

 - [x] **C.3 Honest representation**: Are our visualizations, summary statistics, and reports designed to honestly represent the underlying data?
 * Yes

 - [x] **C.4 Privacy in analysis**: Have we ensured that data with PII are not used or displayed unless necessary for the analysis?
 * Yes

 - [x] **C.5 Auditability**: Is the process of generating the analysis well documented and reproducible if we discover issues in the future?
 * Yes

## D. Modeling
 - [x] **D.1 Proxy discrimination**: Have we ensured that the model does not rely on variables or proxies for variables that are unfairly discriminatory?
 * Yes
 - [x] **D.2 Fairness across groups**: Have we tested model results for fairness with respect to different affected groups (e.g., tested for disparate error rates)?
 * We have a plan to import features including race and gender distribution in census tract level to implement a Geographic Disparity Analysis on the correlation between model outputs and the race and gender distribution.
 - [x] **D.3 Metric selection**: Have we considered the effects of optimizing for our defined metrics and considered additional metrics?
 * Yes. We use 2 metrics, mean squared error (MSE) as loss, R square (R2) as accuracy to evaluate the performances of models. Plus, the geographic similarity of ground truth and predicted results is algo presented.
 - [x] **D.4 Explainability**: Can we explain in understandable terms a decision the model made in cases where a justification is needed?
 * Yes. The outputs of the prediction models represent the predicted census tract level annual averaged value of economic loss in buildings due to wildfire in Texas.
 - [x] **D.5 Communicate bias**: Have we communicated the shortcomings, limitations, and biases of the model to relevant stakeholders in ways that can be generally understood?
 * No

## E. Deployment
 - [x] **E.1 Redress**: Have we discussed with our organization a plan for response if users are harmed by the results (e.g., how does the data science team evaluate these cases and update analysis and models to prevent future harm)?
 * Yes. We would determine at which step among data acquisition, data processing, model running, and model deployment the problem occurs, and then update it at the corresponding stage.
 - [x] **E.2 Roll back**: Is there a way to turn off or roll back the model in production if necessary?
 * Yes. We tracted model versions on Github. So we can roll back if necessary.
 - [x] **E.3 Concept drift**: Do we test and monitor for concept drift to ensure the model remains fair over time?
 * No.
 - [x] **E.4 Unintended use**: Have we taken steps to identify and prevent unintended uses and abuse of the model and do we have a plan to monitor these once the model is deployed?
 * No.

*Data Science Ethics Checklist generated with [deon](http://deon.drivendata.org).*