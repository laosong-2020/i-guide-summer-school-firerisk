import os
import geopandas as gpd
import pandas as pd

# from datasets import Dataset

from config import BASE_DIR, DATASET_DIR

def describe_row(row):
    description = (
        f"In the census tract described here, the exposure type value is {row['Exposure-type']}, "
        f"indicating how close the housing units are to burnable vegetation, with higher values indicating closer proximity. "
        f"The annual burn probability is {row['Burn-Probability']}. The risk to potential structures is valued at {row['Risk-2-Potential-Structures']}. "
        f"The flame length exceedance probability for flames over 8 feet is {row['Flame-Length-Over-8-Feet']}, and for flames over 4 feet is {row['Flame-Length-Over-4-Feet']}. "
        f"The conditional risk to potential structures is {row['Conditional-Risk-2-Potential-Structures']}, and the conditional flame length is {row['Conditional-Flame-Length']} feet. "
        f"The population density in this area is {row['Population-Density']} people per square kilometer, with a total population count of {row['Population-Count']}. "
        f"The housing unit exposure, representing the expected number of housing units potentially exposed to wildfire annually, is {row['Housing-Unit-Exposure']}. "
        f"The housing unit density is {row['Housing-Unit-Density']} units per square kilometer, and the total number of housing units is {row['Housing-Unit-Count']}. "
        f"The building density is {row['Building-Density']} buildings per square kilometer, covering {row['Building-Cover']}% of habitable land area, with a total building count of {row['Building-Count']}. "
        f"The potential evapotranspiration in this area is {row['Potential-Evapotranspiration']} mm. The mean temperature during the summer of 2023 was {row['Mean-Temperature']}°C, "
        f"with a minimum temperature of {row['Min-Temperature']}°C and a maximum temperature of {row['Max-Temperature']}°C. The average precipitation in 2023 was {row['Precipitation']} mm."
    )
    return {
        "text": description,
        "Wildfire-Hazard-Potential-index": row["Wildfire-Hazard-Potential-index"],
        "Housing-Unit-Risk": row["Housing-Unit-Risk"],
        "Housing-Unit-Impact": row["Housing-Unit-Impact"],
    }

if __name__ == "__main__":
    df = pd.read_csv(os.path.join(DATASET_DIR, "dataset.csv"))
    print(df.shape)
    df_cleaned = df.dropna()
    print(df_cleaned.shape)
    dict_list = []

    for _, row in df_cleaned.iterrows():
        #print(row)
        dict_list.append(describe_row(row))
    
    df_bert = pd.DataFrame(dict_list, index=None)

    df_bert.to_csv(f"{DATASET_DIR}/dataset_bert.csv", index=False)