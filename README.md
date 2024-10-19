<div align="left">
  <img src="https://github.com/user-attachments/assets/3f727ef5-2d2c-45ec-ab62-3e4a049e2168" alt="Titanic Gif" width="700"/>
</div>

<div align="left">
  <a href="https://colab.research.google.com/drive/1itTfyj5bdfKmYyCkwf01IpkzQuB4Nxm7?usp=sharing" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"></a>
   <a href="https://www.kaggle.com/competitions/titanic/leaderboard" target="_blank">
  <img src="https://img.shields.io/badge/Kaggle-Top%2010%25-4CAF50" alt="Kaggle Top 10%"></a>
  <a href="https://github.com/Dalageo/ML-TitanicShipwreck/blob/main/LICENSE" target="_blank">
    <img src="https://img.shields.io/badge/License-Apache%202.0-D22128" alt="License: Apache License 2.0"></a>
  <img src="https://img.shields.io/github/stars/Dalageo/ML-TitanicShipwreck?style=social" alt="GitHub stars">
</div>

# Exploring the World's Most Renowned Shipwreck 🚢

In 1912, the Titanic set off on its first voyage across the Atlantic Ocean, carrying passengers ranging from the wealthy elite to emigrants seeking a new life. Tragically, the ship collided with an iceberg and sank, resulting in the loss of over 1,500 lives. This disaster not only shook the world but also sparked discussions about maritime safety and the social dynamics of the time.

This repository explores the factors affecting passenger survival on the Titanic and aims to build a predictive model to estimate survival probabilities based on available passenger characteristics. The available dataset contains a detailed records of the passengers aboard, including information such as age, gender, passenger class, fare paid, and survival outcome. However, some key data points are missing, particularly in features like age and cabin, which poses challenges for building accurate predictive models. 

In this project, two different approaches are explored and compared based on model performance:

- **1. Removing Missing Data**: This method involves deleting rows with missing values to clean the dataset. While it ensures that the remaining data is complete, it reduces the number of observations available for analysis.

- **2. Filling Missing Data**: This approach fills in missing values in an effort to retain more data and potentially enhance the model's performance.

Overall, more robust models (Random Forest, XGBoost) were achieved using the second approach, which involved filling in missing values. A version of the developed model was also submitted to Kaggle’s [Titanic-Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic) competition, where it ranked in the top 9.38% (1316 out of 14036). 

<div align="center">
<img src="https://github.com/user-attachments/assets/0b991de5-3238-4b7f-96bd-99216d136574" alt="Kaggle" width = "800" height = "150"/>
</div>

Given that the true survival status of Titanic passengers is publicly available, some higher-ranked entries likely used manually crafted labels to achieve near-perfect accuracies. Therefore, the actual position of the provided model could be higher if all competitors strictly followed the competition rules. You can also find the Kaggle's notebook [here](https://www.kaggle.com/code/dalageo/exploring-the-world-s-most-renowned-shipwreck).

*It's important to mention that the score shown in the above image (0.78947) was achieved through a **slightly** modified ensemble model and different parameter tuning compared to the provided notebook (0.78468). These exact details are not shared here to encourage independent experimentation and to prevent you from overfitting.* 😜


## Data Description

The Titanic dataset used in this project is divided into two main files: `train.csv` and `test.csv`. Below is a brief description of each file:

- **`train.csv`**: This is the primary training dataset containing labeled data used to train the model. It includes 891 records and 12 columns, with the `Survived` column indicating whether a passenger survived (1) or not (0). This dataset is used to build and validate the machine learning model.
  
- **`test.csv`**: This is the test dataset that contains 418 records and 11 columns. It does **not** have the `Survived` column. The goal is to predict `Survived` using a model trained on the provided training data.

*On the competition's data, you will also find the `gender_submission.csv` file, which is an example submission file (not the true labels) provided by [Kaggle](https://www.kaggle.com/). This file shows the expected format of the predictions, containing only the `PassengerId` and `Survived` columns.*

The following table provides a detailed description of the columns found in `train.csv` and `test.csv`:

| Column Name    | Data Type   | Description                                                                 |
|----------------|-------------|-----------------------------------------------------------------------------|
| `PassengerId`  | Integer     | Unique identifier for each passenger                                        |
| `Survived`     | Integer     | Survival status (0 = No, 1 = Yes)                                           |
| `Pclass`       | Integer     | Passenger class (1 = 1st, 2 = 2nd, 3 = 3rd)                                 |
| `Name`         | String      | Name of the passenger                                                       |
| `Sex`          | String      | Gender of the passenger (`male`, `female`).                                 |
| `Age`          | Float       | Age of the passenger                                                        |
| `SibSp`        | Integer     | Number of siblings/spouses aboard the Titanic                               |
| `Parch`        | Integer     | Number of parents/children aboard the Titanic                               |
| `Ticket`       | String      | Ticket number                                                               |
| `Fare`         | Float       | Passenger fare                                                              |
| `Cabin`        | String      | Cabin number                                                                |  
| `Embarked`     | String      | Port of embarkation (`C` = Cherbourg; `Q` = Queenstown; `S` = Southampton)  |


## Setup Instructions

### <img src="https://upload.wikimedia.org/wikipedia/commons/d/d0/Google_Colaboratory_SVG_Logo.svg" alt="Google Colab Logo" width="15" height = "18"/> **Google Colab Setup**

1. **Download the required dataset from**:
   - **[Kaggle - Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/data)**

2. **Upload the `train.csv` and `test.csv` files to your own Google Drive in your preferred folder structure.**

3. **Update the file paths in the notebook to reflect your own Google Drive paths.**  

4. **Run the notebook cells as instructed to reproduce the results.**

---

### <img src="https://github.com/user-attachments/assets/8d36d1a5-e9b1-40d1-97c9-3d4ca49e9c95" alt="Local PC" width="18" height = "16" /> **Local Environment Setup**

1. **Download the required dataset from**:
    - **[Kaggle - Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/data)**

2. **Clone the repository**:
   ```sh
   git clone https://github.com/Dalageo/ML-TitanicShipwreck.git

3. **Navigate to the cloned directory**:
   ```sh
   cd ML-TitanicShipwreck
  
4. **Open the `Exploring the World's Most Renowned Shipwreck.ipynb` using your preferred Jupyter-compatible environment (e.g., [Jupyter Notebook](https://jupyter.org/), [VS Code](https://code.visualstudio.com/), or [PyCharm](https://www.jetbrains.com/pycharm/))**
   
5. **Update file paths for `train.csv` and `test.csv` as needed.**
   
6. **Run the cells sequentially to reproduce the results.**


## Acknowledgments

The dataset used in this project is provided by [Kaggle](https://kaggle.com/competitions/titanic) as part of the [Titanic-Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic) competition. Special thanks to [Kaggle's](https://www.kaggle.com/) data science community, and Will Cukierski for making this dataset available for educational and research purposes.


## License

This work is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). It was chosen to comply with the competition rules, which require the use of an [Open Source Initiative (OSI)](https://opensource.org/) approved license that permits commercial use while promoting open collaboration.

<div align="center">
<a href="https://www.apache.org/licenses/LICENSE-2.0">
  <img src="https://github.com/user-attachments/assets/bcf30286-f8b7-488a-8300-ec2464090c33" alt="Apache License 2.0" width="220" height="120">
</a>
</div>






