# %%
from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("yujunzhou/LabSafety_Bench", "MCQ")
# %%
ds["QA"]
# %%
print("Loading CBRN safety dataset...")
#ds = load_dataset("yujunzhou/LabSafety_Bench", "MCQ")
data_subset = ds["QA"][:5]  # Start with 5 questions
for item in data_subset:
    print(item)
questions = [item["Question"] for item in data_subset]
correct_answers = [item["Correct Answer"] for item in data_subset]
# %%
