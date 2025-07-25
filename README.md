# Character Prediction Program

Implementation of an n-gram character prediction program.
Given a string of characters, the program attempts to guess the next character.  
For demonstration, this repository includes a basic model that generates random guesses.

EDIT: now supports multiple langauages other than English 

---

## Input Format

`example/input.txt` contains sample input data.  
Each line in this file is a separate string for which the program should guess the next character.

---

## Output Format

`example/pred.txt` shows what the expected output format should look like.  
Each line in this file contains predictions for the corresponding line in `example/input.txt`.  
This example produces 3 guesses per input line.

---

## Program Structure

The main program is located in `src/myprogram.py`, which supports both training and prediction workflows.

### Training Mode

During training, your implementation may:

1. Load training data  
2. Train a model  
3. Save model checkpoints (e.g., to `work/model.checkpoint`)

To run training:

```bash
python3 src/myprogram.py train --work_dir work
