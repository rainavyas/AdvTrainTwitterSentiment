# Objective

NLP classification of twitter tweets into one of six emotions: love, joy, fear, anger, surprise, sadness.
The dataset is described in https://www.kaggle.com/praveengovi/emotions-dataset-for-nlp?select=train.txt

Training dataset is augmented with synonym substitution based adversarial examples.


# Requirements

python3.4 or above

## Necessary Packages (installation with PyPI)

pip install torch, torchvision

pip install transformers

pip install nltk


# Training

Fork the repository (and clone).

Generate adversarial examples using _generate_adv_examples.py_

Run the _adv_train.py_ scripts with desired arguments in your terminal. For example, to train an ELECTRA-based classifier:

_python ./adv_train.py adversarial_examples_dir electra_trained.th electra --B=8 --lr=0.00001 --epochs=2_

# Experimental Results

| Training | Test Accuracy (%) |
| ----------------- | :-----------------: |
Without Adversarial Training| 93.3 |
With Adversarial Training | - |


### Training Details

- Initialise encoder with _ELECTRA_
- Batch Size = 8
- Epochs = 2
- Learning Rate = 1e-5
- 6 adversarial examples per data point, using N={1,2,3,4,5,6} substitutions
