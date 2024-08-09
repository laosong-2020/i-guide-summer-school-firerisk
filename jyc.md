# Aggregation method
- How to aggregate the variable? (mean, median, sum, average) ?
    - conclusion
        - The aggregation methods should align with how the expected annual loss is calculated, focusing on **cumulative impact for variables related to risk and exposure (sums/means)** and **average conditions for environmental factors (means)**.
        
        ### 1. Risk Factors
        
        Risk factors are variables that directly relate to the potential severity and intensity of a wildfire. These typically include variables related to fire behavior and intensity.
        
        - **Burn Probability (BP):**
            - **Aggregation:** Mean
            - **Rationale:** Represents the average probability of the tract being burn by a large fire.
        - **Conditional Flame Length (CFL):**
            - **Aggregation:** Mean
            - **Rationale:** Provides a sense of average fire intensity.
        - **Flame Length Exceedance Probability - 4 ft (FLEP4) and 8 ft (FLEP8):**
            - **Aggregation:** Mean
            - **Rationale:** Offers an average probability of encountering significant fire intensity.
        - **Conditional Risk to Potential Structures (cRPS):**
            - **Aggregation:** Mean
            - **Rationale:** understand the average risk level across the entire block group.
        
        ### 2. Exposure Factors
        
        Exposure factors involve elements related to the vulnerability of human and structural assets to wildfire impacts.
        
        - **Building-Count, Housing-Unit-Count:**
            - **Aggregation:** Sum
            - **Rationale:** Reflects the total number of structures at risk within each block group.
        - **Building Density, Housing-Unit-Density:**
            - **Aggregation:** Mean
            - **Rationale:** Average density provides insights into the spatial distribution of structures, affecting exposure and risk.
        - **Building Coverage:**
            - **Aggregation:** Mean
            - **Rationale:** Indicates the average percentage of land covered by buildings, impacting potential exposure and loss.
        - **Housing Unit Exposure (HUExposure):**
            - **Aggregation:** Sum
            - **Rationale:** Provides the total number of housing units potentially exposed, aligning with how losses are calculated.
        - **Population-Density**
            - **Aggregation:** Mean
            - **Rationale:** Average density provides insights into the population distribution.
        - **Exposure Type:**
            - **Aggregation:** Mean
            - **Rationale:** Captures the average exposure type within each block group.
        
        ### 3. Environmental Factors
        
        Environmental factors influence the conditions that affect fire spread and behavior, such as climate and vegetation conditions.
        
        - **Precipitation:**
            - **Aggregation:** Mean
            - **Rationale:** Average precipitation provides a representative measure of moisture availability, impacting fuel conditions and fire behavior.
        - **MaxTemperature, MinTemperature, MeanTemperature:**
            - **Aggregation:** Mean
            - **Rationale:** Represents average temperature conditions.
        - **Potential Evapotranspiration:**
            - **Aggregation:** Mean
            - **Rationale:** Reflects average moisture loss conditions, impacting fuel moisture and fire potential.
    - Details on how the x and y calculated
        - target feature
            
            ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/cb0dfb90-df60-4e9d-906b-98dc4a714221/60014085-67d8-4de4-848d-70adcc277197/Untitled.png)
            
            ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/cb0dfb90-df60-4e9d-906b-98dc4a714221/14e95bb3-4cad-4915-9e04-2a0d3ff21845/Untitled.png)
            
            ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/cb0dfb90-df60-4e9d-906b-98dc4a714221/3b615e02-6384-4d6d-8ca3-941e8d4ead76/Untitled.png)
            
            Exposure is typically calculated at the Census block level and then aggregated to the Census tract and county level by **summing the Census block exposure values** within the parent Census tract or parent county.
            
            The annualized frequency value represents the **area-weighted BP** (due to a large fire) of a location in a given year. The Census block inherits the hazard occurrence count from the fishnet cell that encompasses it, or, if a Census block intersects multiple fishnet cells, an **area-weighted count** is calculated
            
            The HLR is the representative percentage of a location’s hazard type exposure that experiences loss due to a hazard occurrence or the average rate of loss associated with the hazard occurrence. In concept, it is the **average of the loss ratios associated with past hazard occurrences** and is used to estimate the potential impact of a future hazard occurrence. 
            
        - Xs
            
            Building Count: Building Count is a 30-m raster representing the count of buildings in the building footprint dataset located within each 30-m pixel.
            
            Building Density: Building Density is a 30-m raster representing the density of buildings in the building footprint dataset (buildings per square kilometer [km²]).
            
            Building Coverage: Building Coverage is a 30-m raster depicting the percentage of habitable land area covered by building footprints.
            
            Housing Unit Density (HUDen): HUDen is a 30-m raster of housing-unit density (housing units/km²).
            
            Housing Unit Exposure (HUExposure): HUExposure is a 30-m raster that represents the expected number of housing units within a pixel potentially exposed to wildfire in a year. This is a long-term annual average and not intended to represent the actual number of housing units exposed in any specific year.
            
            Conditional Risk to Potential Structures (cRPS): The potential consequences of fire to a home at a given location, if a fire occurs there and if a home were located there. Referred to as Wildfire Consequence in the Wildfire Risk to Communities web application.
            
            Exposure Type: Exposure is the spatial coincidence of wildfire likelihood and intensity with communities. Generate the Exposure Type raster by applying the smoothing process described above for burn probability using the LANDFIRE 2.2.0 fuels data as the primary input. Assign a value of one to all burnable pixels and a value of 0 in all nonburnable pixels in the original 30-m resolution LANDFIRE data. Then apply the spatial smoothing used for BP (three iterative 510-m focal means) to spread values into otherwise non-burnable areas, using the same steps described above to handle small patches of burnable vegetation and other land cover types. if the underlying land cover is considered burnable in the LANDFIRE fuel, the Exposure Type is “direct” (pixel value of 1). The exposure type is “indirect” (pixel value between 0 and 1) if two conditions are met: 1) the land cover is nonburnable urban, agricultural, or bare ground, and 2) the smoothed BP > 0. Finally, the exposure type is “nonexposed” (pixel value of 0) if the underlying land cover is nonburnable and the upsampled BP = 0.
            
            Burn Probability (BP): The probability of an area being burned by a large fire. Referred to as Wildfire Likelihood in the Wildfire Risk to Communities web application.
            
            Conditional Flame Length (CFL): The mean flame length for a fire burning in the direction of maximum spread (headfire) at a given location if a fire were to occur; an average measure of wildfire intensity.
            
            Flame Length Exceedance Probability - 4 ft (FLEP4): The conditional probability that flame length at a pixel will exceed 4 feet if a fire occurs; indicates the potential for moderate to high wildfire intensity.
            
            Flame Length Exceedance Probability - 8 ft (FLEP8): the conditional probability that flame length at a pixel will exceed 8 feet if a fire occurs; indicates the potential for high wildfire intensity.
            

## Dasymetric mapping

weighted average calculation

- **Calculate Proportionate Area:**
    - For each census tract, calculate the proportion of the tract covered by each class of wildfire factors. This involves overlaying the raster and using spatial analysis tools to determine the area of each factors within each tract.
- **Weighted Average Calculation:**
    - Calculate a weighted average of the wildfire likelihood for each census tract. This can be done by multiplying each likelihood value by the proportion of the census tract it occupies, summing these products, and then dividing by the total area of the census tract:
    Weighted Likelihood=Total Tract Area∑(Likelihood Value×Area Covered)
        
        ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/cb0dfb90-df60-4e9d-906b-98dc4a714221/058490ea-4b95-4760-b922-e159f77fde77/Untitled.png)
        

https://github.com/USEPA/Dasymetric-Toolbox-OpenSource/tree/master

https://essd.copernicus.org/articles/14/2833/2022/#section3


# Gradient Boost, XGBoost
## Gradient Boost

Gradient boosting is a machine learning ensemble technique. It combines the predictions result of multiple weak learner model, specifically decision tree. By building a new model on the errors or residuals of previous model, gradient boosting optimizes the model performance sequentially. 

- How Gradient boosting works
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/cb0dfb90-df60-4e9d-906b-98dc4a714221/c05ed080-a0a5-4dbb-9f6d-fb2bc4952f39/Untitled.png)
    
    The ensemble consists of M trees. Tree1 is trained using the feature matrix *X* and the labels *y*. The predictions labeled *y1(hat)* are used to determine the training set residual errors *r1*. Tree2 is then trained using the feature matrix *X* and the residual errors *r1* of Tree1 as labels. The predicted results *r1(hat)* are then used to determine the residual *r2*. The process is repeated until all the M trees forming the ensemble are trained. 
    

1. A loss function to be optimized
    1. The loss function used depends on the type of problem being solved. Regression task uses a squared error and classification task uses logarithmic loss.
2. A weak learner to make predictions
    1. Decision trees are used as the weak learner in gradient boosting. It is common to constrain the weak learners in some ways such as maximum number of layers, nodes, splits or leaf nodes. 
3. Additive model to add weak learner to minimize the loss function
    1. Gradient descent is used to minimize the loss

**Pros:**

1. **High Accuracy**: Gradient boosting can produce highly accurate predictive models by combining many weak learners (e.g., decision trees).
2. **Flexibility**: It can be applied to both regression and classification problems and is adaptable to different loss functions.
3. **Feature Importance**: The algorithm provides insights into feature importance, helping to understand which features contribute most to predictions.
4. **Handles Non-Linear Relationships**: Can capture complex non-linear patterns in the data due to its ensemble nature.

**Cons:**

1. **Training Time**: Gradient boosting can be slow to train, especially with large datasets, due to its sequential nature.
2. **Overfitting**: The algorithm can overfit if not properly regularized or if the trees are too deep.
3. **Parameter Tuning**: Requires careful tuning of multiple hyperparameters, such as learning rate, number of trees, and tree depth.
4. **Interpretability**: Models can be difficult to interpret, especially with many trees.

## XGBoost

**XGBoost (Extreme Gradient Boosting)** is an advanced implementation of gradient boosting that focuses on speed and performance. It was developed to address some limitations of the traditional gradient boosting algorithm. Here are some key improvements and features of XGBoost:

1. **Regularization**: XGBoost includes L1 (Lasso) and L2 (Ridge) regularization terms in its objective function, which helps prevent overfitting and improves model generalization.
2. **Parallelization**: The algorithm is designed to be parallelized, allowing it to take full advantage of multi-core processors and making it much faster than traditional gradient boosting implementations.
3. **Tree Pruning**: XGBoost uses a more effective tree pruning algorithm called "max depth" rather than the "depth-first" approach used in standard gradient boosting. This leads to more efficient and compact trees.
4. **Scalability**: XGBoost is highly scalable and can handle large datasets with millions of examples and features.

**Pros:**

1. **Speed and Performance**: XGBoost is optimized for speed and performance through parallel and distributed computing, making it significantly faster than traditional gradient boosting.
2. **Regularization**: Incorporates L1 (Lasso) and L2 (Ridge) regularization, which helps prevent overfitting and improves model generalization.
3. **Handling Missing Values**: Can automatically handle missing data, making preprocessing easier.
4. **Built-in Cross-Validation**: Supports built-in cross-validation, allowing for better model evaluation and hyperparameter tuning.
5. **Scalability**: Highly scalable and can efficiently handle large datasets.
6. **Tree Pruning**: Uses a more effective tree pruning algorithm that improves model efficiency and reduces overfitting.

**Cons:**

1. **Complexity**: The algorithm is more complex, with additional parameters to tune, which can make it challenging to configure optimally.
2. **Resource Intensive**: Despite optimizations, XGBoost can still be resource-intensive, requiring significant memory and computation power, especially with very large datasets.
3. **Sensitivity to Hyperparameters**: Like gradient boosting, XGBoost's performance is highly dependent on proper hyperparameter tuning, which can be time-consuming.
4. **Interpretability**: While feature importance is provided, the model itself can be difficult to interpret due to its ensemble nature.

https://www.datacamp.com/tutorial/guide-to-the-gradient-boosting-algorithm

https://machinelearningmastery.com/gentle-introduction-gradient-boosting-algorithm-machine-learning/

https://www.analyticsvidhya.com/blog/2021/09/gradient-boosting-algorithm-a-complete-guide-for-beginners/