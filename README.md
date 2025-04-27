# Refining Human Pose Estimation
[Website](https://sarahtj.github.io/website/)

This project uses Vision Transformers to improve upon the state-of-the-art Human Pose Estimation model, ViTPose. There are two parts to this project. Part One: Improving the ViTPose backbone by incorporating a Feature Pyramid Network. Part Two: Predicting n number of future human poses, taking in a sequence of past human poses. 

Authors: Niva Ranavat, Sarah Jamil, Adithya Raman, Jacob Klinger


# Future Frame Prediction Part:

## Project Setup Instructions

Follow the steps below to set up the project using a Python virtual environment.

---

### Step 1: Clone the Repository

```bash
git clone https://github.com/nranavat1/Refined_Human_Pose_Estimation.git
cd Refined_Human_Pose_Estimation
```


### Step 2: Create a Virtual Environment
You will need Python 3.9 or higher for this project. 

```bash
python3 -m venv venv
```

### Step 3: Activate the Virtual Environment
On macOS/Linux:

```bash
source venv/bin/activate
```


On Windows:

```bash
venv\Scripts\activate
```


### Step 4: Install the Requirements

```bash
pip install numpy pandas matplotlib tensorflow torch torchvision opencv-python scikit-learn seaborn transformers
```

### Step 5: Download Required Datasets (3D Poses in the Wild)

This project part uses the 3D Poses in the Wild(3DPW) dataset. Below are instructions to install the dataset. 


#### Instructions

**Step 1:** Go to the official 3D Poses in The Wild dataset website:  
[3D Poses in The Wild Dataset]([https://virtualhumans.mpi-inf.mpg.de/3DPW/](https://virtualhumans.mpi-inf.mpg.de/3DPW/evaluation.html))

**Step 2:** Scroll down to the **Download** section, accept the License and download the following:
- [readme_and_demo.zip](https://virtualhumans.mpi-inf.mpg.de/3DPW/readme_and_demo.zip)
- [imageFiles.zip](https://virtualhumans.mpi-inf.mpg.de/3DPW/imageFiles.zip)
- [sequenceFiles.zip](https://virtualhumans.mpi-inf.mpg.de/3DPW/sequenceFiles.zip)


**Step 3:** Extract the downloaded `.zip` files.

**Step 4:** Create a folder in your project called `data` and move the extracted folders into it.  
  1. **Navigate to the project folder**  
   Open a terminal and `cd` into the project root directory.

  2. **Create a `data` directory**  
   This is where the datasets will be stored.

   ```bash
   mkdir -p data
   cd data
   ```

### Step 6: Run the code

The first file will load the image files and sequence files. The second file will run the model. By default, this will run the already trained model that we built on the testing data in the dataset.

```bash
cd ./pose_prediction
python3 extract_images.py
python3 train_test.py
```

# Improving ViTPose:

Repeat Steps 1 - 4



