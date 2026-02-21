# Task 1: CNN Classification with Comprehensive Analysis

**Author:** Dr. Anas M. Ali  
**Email:** aaboessa@psu.edu.sa  
**Affiliation:** Researcher, Prince Sultan University  

## 1. Project Overview & Idea Description
This project tackles the binary classification of pneumonia from chest X-ray images using the PneumoniaMNIST dataset, a subset of the MedMNIST v2 benchmark. 

**The Challenge:** The dataset suffers from a significant class imbalance between the "Normal" and "Pneumonia" classes, which often causes standard deep learning models to overfit to the majority class (Pneumonia) and exhibit poor specificity.

**The Solution (Class Decomposition):** Instead of using standard oversampling or undersampling, I implemented a **Class Decomposition** strategy using K-Means clustering. 
1. **Feature Extraction:** A pre-trained visual language model (`openai/clip-vit-base-patch32`) was used to extract high-level semantic embeddings from all images in the majority "Pneumonia" class.
2. **Clustering:** K-Means clustering was applied to these embeddings to split the "Pneumonia" class into two distinct sub-types (`Pneumonia_Type_1` and `Pneumonia_Type_2`). [Image of K-Means clustering algorithm]
3. **Training Strategy:** The problem was transformed from a binary classification task to a 3-class classification task during training. This forced the model to learn finer-grained features within the pneumonia class rather than just generalizing it as a single block. During inference, the probabilities of the two pneumonia sub-types are summed to map the prediction back to the original binary label.

### Data Distribution Before and After Decomposition
By decomposing the majority class, the dataset distribution becomes much more balanced, preventing the model from collapsing into majority-class predictions.

**Before Decomposition (Binary):**
![Distribution Before](distribution_before.png)

**After Decomposition (3-Class):**
![Distribution After](distribution_after.png)

---

## 2. Repository Structure

```text
Task_1/
├── requirements.txt                   # Python dependencies required to run the scripts
├── README.md                          # Main repository overview
├── task1_classification_report.md     # This detailed methodology and results report
├── notebooks/
│   ├── Download_and_Analysis_dataset.ipynb # Data exploration and visualization
│   └── Tutorial.ipynb                 # End-to-end demonstration notebook
├── task1_classification/
│   ├── train_decomposition.py         # Script for CLIP feature extraction, clustering, and training
│   └── evaluate_decomp.py             # Script for testing the model and generating metrics
├── models/
│   └── best_model.pth                 # Saved weights for the trained MaxViT-T model
└── reports/                           # Generated evaluation outputs and figures
    ├── accuracy_curve.png
    ├── loss_curve.png
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── distribution_before.png
    ├── distribution_after.png
    ├── failure_cases.png
    └── metrics_summary.txt
