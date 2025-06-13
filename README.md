# FYP
Final Year Project


# Model Workflow Overview

This project organizes the machine learning workflow into two clear stages: **model training** and **result plotting**. Intermediate model files are saved and reused, enabling easy evaluation and visualization of results.

All steps are implemented and demonstrated in Jupyter notebooks.

---

## 🔧 Model Training (`*_generate.ipynb`)

To train models, run the corresponding `*_generate.ipynb` notebook for your task type

### How to use:

1. Open the desired notebook.
2. Specify the file paths for saving model parameters.
3. Adjust any parameters as needed.
4. Run all cells to:
   - Train the model.
   - Save the trained model parameters to disk.

---

## 📊 Results Visualization (`*_results.ipynb`)

After training and saving the models, you can evaluate and visualize performance by running the corresponding `*_results.ipynb` notebook. This notebook will:

- Load the saved model parameters.
- Calculate relevant accuracy or performance metrics.
- Generate and display plots to illustrate the results.